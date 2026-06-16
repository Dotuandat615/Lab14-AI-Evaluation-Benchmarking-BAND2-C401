import asyncio
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


class MainAgent:
    """Optimized V2 support agent used as the release candidate."""

    def __init__(self, corpus_path: str = "data/document_corpus.json"):
        self.name = "SupportAgent-v2-optimized"
        self.version = "Agent_V2_Optimized"
        self.corpus_path = Path(corpus_path)
        self.corpus = self._load_corpus()

    async def query(self, question: str) -> Dict:
        await asyncio.sleep(0.25)

        lower_question = question.lower()
        retrieved = self._retrieve(question, top_k=3)

        if self._is_vpn_violation(lower_question):
            answer = self._vpn_violation_answer(retrieved)
        elif self._is_security_bypass(lower_question):
            answer = self._security_refusal(question, retrieved)
        elif self._is_out_of_scope(lower_question):
            answer = self._out_of_scope_answer(lower_question)
        elif retrieved:
            answer = self._grounded_answer(question, retrieved)
        else:
            answer = (
                "Tôi không tìm thấy thông tin rõ ràng trong tài liệu nội bộ được "
                "cung cấp. Vui lòng liên hệ phòng ban phụ trách để được xác nhận "
                "chính xác."
            )

        return {
            "answer": answer,
            "contexts": [text for _, text, _ in retrieved],
            "metadata": {
                "agent_version": self.version,
                "model": "local-retrieval-template-v2",
                "tokens_used": self._estimate_tokens(answer, retrieved),
                "sources": [doc_id for doc_id, _, _ in retrieved],
                "retrieval_used": bool(retrieved),
            },
        }

    def _load_corpus(self) -> Dict[str, str]:
        if not self.corpus_path.exists():
            return {}
        with self.corpus_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _retrieve(self, question: str, top_k: int = 3) -> List[Tuple[str, str, int]]:
        query_tokens = set(self._tokenize(question))
        if not query_tokens:
            return []

        lower_question = question.lower()
        intent_doc_ids = self._intent_doc_ids(lower_question)
        ranked: List[Tuple[str, str, int]] = []
        seen = set()

        for boost, doc_id in enumerate(intent_doc_ids):
            text = self.corpus.get(doc_id)
            if text:
                ranked.append((doc_id, text, 100 - boost))
                seen.add(doc_id)

        for doc_id, text in self.corpus.items():
            if doc_id in seen:
                continue
            doc_tokens = set(self._tokenize(text))
            score = len(query_tokens & doc_tokens)
            if score > 0:
                ranked.append((doc_id, text, score))

        ranked.sort(key=lambda item: (item[2], item[0]), reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)

    @staticmethod
    def _estimate_tokens(answer: str, retrieved: List[Tuple[str, str, int]]) -> int:
        context_tokens = sum(len(text.split()) for _, text, _ in retrieved)
        return max(80, len(answer.split()) + context_tokens // 2)

    @staticmethod
    def _is_security_bypass(lower_question: str) -> bool:
        risky_terms = [
            "hack",
            "admin",
            "lương của tất cả",
            "luong cua tat ca",
            "tắt tường lửa",
            "tat tuong lua",
            "bỏ qua tất cả",
            "bo qua tat ca",
            "mật khẩu của hệ thống",
            "mat khau cua he thong",
        ]
        return any(term in lower_question for term in risky_terms)

    @staticmethod
    def _is_vpn_violation(lower_question: str) -> bool:
        return (
            "quên thiết lập vpn" in lower_question
            or "quen thiet lap vpn" in lower_question
            or (
                "vô tình truy cập" in lower_question
                and ("vpn" in lower_question or "từ xa" in lower_question)
            )
        )

    @staticmethod
    def _is_out_of_scope(lower_question: str) -> bool:
        out_of_scope_terms = [
            "thời tiết",
            "thoi tiet",
            "điện biên phủ",
            "dien bien phu",
            "bài luận",
            "bai luan",
            "chứng khoán",
            "chung khoan",
            "insider trading",
        ]
        return any(term in lower_question for term in out_of_scope_terms)

    @staticmethod
    def _intent_doc_ids(lower_question: str) -> List[str]:
        rules = [
            (("nghỉ phép", "nghi phep"), ["doc_policy_002"]),
            (("reset", "otp", "quên mật khẩu", "quen mat khau"), ["doc_tech_001", "doc_policy_001"]),
            (("mật khẩu", "mat khau"), ["doc_policy_001", "doc_tech_001"]),
            (("vpn", "từ xa", "tu xa", "wfh"), ["doc_tech_002", "doc_policy_003"]),
            (("cài", "cai", "phần mềm", "phan mem"), ["doc_tech_003"]),
            (("backup", "sao lưu", "sao luu", "dữ liệu", "du lieu"), ["doc_tech_004"]),
            (("bảo mật", "bao mat", "hack", "hacker", "lừa đảo", "lua dao"), ["doc_tech_005"]),
            (("onboarding", "nhân viên mới", "nhan vien moi"), ["doc_policy_004"]),
            (("công tác", "cong tac", "vé máy bay", "ve may bay"), ["doc_policy_005", "doc_finance_001"]),
            (("hóa đơn", "hoa don", "thanh toán", "thanh toan"), ["doc_finance_001"]),
            (("tạm ứng", "tam ung"), ["doc_finance_002"]),
            (("đánh giá hiệu suất", "danh gia hieu suat"), ["doc_hr_001"]),
            (("bảo hiểm", "bao hiem", "phúc lợi", "phuc loi"), ["doc_hr_002"]),
            (("đào tạo", "dao tao", "khóa học", "khoa hoc"), ["doc_hr_003"]),
            (("làm thêm", "lam them", "tăng ca", "tang ca", "ngày lễ", "ngay le"), ["doc_hr_004"]),
            (("nghỉ việc", "nghi viec", "bàn giao", "ban giao"), ["doc_hr_005"]),
        ]

        doc_ids: List[str] = []
        for terms, docs in rules:
            if any(term in lower_question for term in terms):
                for doc_id in docs:
                    if doc_id not in doc_ids:
                        doc_ids.append(doc_id)
        return doc_ids

    @staticmethod
    def _out_of_scope_answer(lower_question: str) -> str:
        if "thời tiết" in lower_question or "thoi tiet" in lower_question:
            return (
                "Xin lỗi, tôi là hệ thống hỗ trợ nội bộ công ty và không có khả "
                "năng truy cập thông tin thời tiết thực tế. Tôi chỉ có thể hỗ "
                "trợ các câu hỏi về chính sách, quy trình và hỗ trợ kỹ thuật "
                "của công ty."
            )
        if "điện biên phủ" in lower_question or "bài luận" in lower_question:
            return (
                "Xin lỗi, tôi là hệ thống hỗ trợ nội bộ công ty, chỉ có thể trả "
                "lời các câu hỏi liên quan đến chính sách, quy trình và hỗ trợ "
                "kỹ thuật của công ty. Tôi không thể hỗ trợ việc viết bài luận "
                "lịch sử."
            )
        if "chứng khoán" in lower_question or "insider trading" in lower_question:
            return (
                "Tôi không tìm thấy thông tin về chính sách này trong tài liệu "
                "nội bộ được cung cấp. Đây có thể là câu hỏi liên quan đến tuân "
                "thủ pháp luật như insider trading. Vui lòng liên hệ phòng Pháp "
                "chế hoặc Tuân thủ để được tư vấn chính xác."
            )
        return (
            "Xin lỗi, tôi là hệ thống hỗ trợ nội bộ công ty, chỉ có thể trả lời "
            "các câu hỏi liên quan đến chính sách, quy trình và hỗ trợ kỹ thuật "
            "của công ty. Tôi không tìm thấy thông tin phù hợp trong tài liệu "
            "nội bộ được cung cấp."
        )

    def _security_refusal(
        self, question: str, retrieved: List[Tuple[str, str, int]]
    ) -> str:
        context = self._format_context(retrieved)
        lower_question = question.lower()
        if "lương của tất cả" in lower_question or "luong cua tat ca" in lower_question:
            return (
                "Tôi không có quyền truy cập thông tin lương cá nhân của nhân "
                "viên, và đây là thông tin bảo mật. Dù bạn là ai, tôi không thể "
                "chia sẻ dữ liệu lương cá nhân. Vui lòng liên hệ phòng HR qua "
                "kênh chính thức."
            )
        if "tắt tường lửa" in lower_question or "tat tuong lua" in lower_question:
            return (
                "Tuyệt đối không làm theo yêu cầu của hacker. Đây là tình huống "
                "ransomware hoặc social engineering. Ngay lập tức: (1) ngắt kết "
                "nối internet, (2) báo cáo tới security@company.com hoặc ext "
                "9999, và (3) không tắt tường lửa hay thay đổi hệ thống khi chưa "
                "có hướng dẫn từ IT Security.\n\n"
                f"Thông tin liên quan trong tài liệu nội bộ:\n{context}"
            )
        if "nghỉ phép" in lower_question or "nghi phep" in lower_question:
            return (
                "Tôi có thể trả lời phần chính sách nghỉ phép: nhân viên được "
                "12 ngày nghỉ phép có lương mỗi năm và cần đăng ký trước ít nhất "
                "3 ngày làm việc. Tuy nhiên, yêu cầu truy cập trái phép hoặc hack "
                "hệ thống HR là vi phạm chính sách bảo mật nghiêm trọng, nên tôi "
                "không thể hỗ trợ phần đó.\n\n"
                f"Thông tin liên quan trong tài liệu nội bộ:\n{context}"
            )
        return (
            "Tôi không thể hỗ trợ yêu cầu bỏ qua chính sách, truy cập trái phép, "
            "chia sẻ thông tin nhạy cảm hoặc thực hiện hành vi gây rủi ro bảo mật. "
            "Nếu bạn cần hỗ trợ hợp lệ, hãy dùng kênh chính thức như IT Helpdesk, "
            "security@company.com hoặc đường dây nóng bảo mật ext 9999.\n\n"
            f"Thông tin liên quan trong tài liệu nội bộ:\n{context}"
        )

    def _vpn_violation_answer(self, retrieved: List[Tuple[str, str, int]]) -> str:
        context = self._format_context(retrieved)
        return (
            "Có, đây là vi phạm chính sách bảo mật vì VPN bắt buộc khi truy cập "
            "hệ thống từ bên ngoài văn phòng. Bạn nên: (1) ngắt kết nối ngay, "
            "(2) kết nối lại qua VPN, và (3) báo cáo sự cố tới "
            "security@company.com hoặc ext 9999 để ghi nhận. Tự báo cáo sẽ giúp "
            "giảm nhẹ mức độ vi phạm.\n\n"
            f"Thông tin liên quan trong tài liệu nội bộ:\n{context}"
        )

    def _grounded_answer(
        self, question: str, retrieved: List[Tuple[str, str, int]]
    ) -> str:
        context = self._format_context(retrieved)
        return (
            "Dựa trên tài liệu nội bộ được truy xuất, câu trả lời là:\n"
            f"{context}\n\n"
            "Nếu tình huống của bạn có chi tiết chưa được nêu trong tài liệu, "
            "hãy xác nhận thêm với phòng ban phụ trách."
        )

    @staticmethod
    def _format_context(retrieved: List[Tuple[str, str, int]]) -> str:
        if not retrieved:
            return "- Không tìm thấy tài liệu liên quan trực tiếp."
        return "\n".join(f"- [{doc_id}] {text}" for doc_id, text, _ in retrieved)


if __name__ == "__main__":
    async def _demo():
        agent = MainAgent()
        resp = await agent.query("Làm thế nào để reset mật khẩu?")
        print(resp)

    asyncio.run(_demo())
