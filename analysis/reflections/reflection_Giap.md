# Báo cáo Cá nhân (Individual Reflection)

- **Họ tên:** Giáp
- **Vai trò trong nhóm:** AI/Backend — **Multi-Judge Consensus Engine**
- **Module phụ trách:** `engine/llm_judge.py` (+ tích hợp vào `main.py`, `requirements.txt`)
- **Commit chính:** `feat: Implement Multi-Judge Consensus Engine (AI/Backend)` (branch `giap`)

---

## 1. Engineering Contribution (Đóng góp kỹ thuật)

Tôi chịu trách nhiệm xây dựng toàn bộ **Hệ thống chấm điểm đồng thuận đa Judge** — phần lõi đánh giá Generation của Evaluation Factory.

### 1.1. Consensus đa model (≥ 2 Judge khác provider)
- Mặc định chạy **cross-provider**: `claude-opus-4-8` (Anthropic SDK) + `gpt-4o` (OpenAI SDK). Việc dùng 2 provider khác nhau (chứ không phải 2 model cùng nhà) giúp triệt tiêu **bias hệ thống** của một họ model duy nhất — đúng tinh thần "đừng chỉ tin một Judge" trong Expert Tips.
- Mỗi Judge chấm theo 3 tiêu chí có rubric rõ ràng: `accuracy`, `professionalism`, `safety` (thang 1–5).
- Các Judge được gọi **song song bằng `asyncio.gather`** để giảm latency (tận dụng điểm Performance/Async).

### 1.2. Calibration — Agreement Rate
- Với mỗi tiêu chí, tính độ lệch `spread = max - min` rồi chuẩn hoá về `[0, 1]`:
  `agreement = 1 - spread / SCORE_RANGE` (SCORE_RANGE = 4).
- `agreement_rate` cuối cùng = trung bình agreement của tất cả tiêu chí → đây chính là **hệ số đồng thuận** mà `check_lab.py` và `summary.json` yêu cầu.

### 1.3. Xử lý xung đột tự động (Conflict Resolution)
- Khi `spread > conflict_threshold` (mặc định 1.0 trên thang 1–5) → đánh dấu `needs_review = True` và lưu chi tiết xung đột (giá trị từng Judge, spread, cách giải quyết).
- **Chiến lược hợp nhất điểm:**
  - Tiêu chí thường (`accuracy`, `professionalism`): lấy **median** → chống outlier khi một Judge chấm lệch.
  - Tiêu chí `safety`: lấy **min** (thận trọng nhất) → một câu trả lời chỉ an toàn khi **mọi** Judge đều thấy an toàn. Đây là lựa chọn thiết kế có chủ đích: với safety, *false negative* nguy hiểm hơn nhiều *false positive*.

### 1.4. Position Bias Detection
- `check_position_bias()` đảo vị trí A/B giữa 2 lượt chấm. Nếu người thắng đổi theo thứ tự xuất hiện → kết luận Judge **thiên vị vị trí**, không phải thiên vị chất lượng.

### 1.5. Cost & Token Reporting
- Bảng giá `PRICES` (USD / 1M token, tách input/output). Mỗi verdict ghi lại `input_tokens`, `output_tokens`, `cost_usd`.
- `get_cost_report()` trả về tổng chi phí, số lần eval và **chi phí trung bình mỗi lần eval** → đưa thẳng vào `summary.json` (`avg_eval_cost_usd`, block `cost`).

---

## 2. Technical Depth (Chiều sâu kỹ thuật)

### 2.1. Agreement Rate vs. Cohen's Kappa
- **Agreement Rate** (cách tôi triển khai) đo tỉ lệ đồng thuận **thô** trên thang điểm liên tục — đơn giản, dễ giải thích.
- **Cohen's Kappa (κ)** chặt chẽ hơn: nó **trừ đi phần đồng thuận do may rủi** (chance agreement): `κ = (p_o - p_e) / (1 - p_e)`, với `p_o` là đồng thuận quan sát được, `p_e` là đồng thuận kỳ vọng ngẫu nhiên. Hai Judge có thể "đồng thuận 80%" nhưng κ thấp nếu phân phối điểm lệch. Hướng nâng cấp tiếp theo của module là bổ sung κ cho dữ liệu dạng phân loại (ví dụ pass/fail).

### 2.2. Position Bias
- LLM-as-a-Judge có xu hướng ưu ái câu trả lời ở **vị trí đầu** (hoặc cuối) bất kể chất lượng. Đây là lỗi calibration nguy hiểm trong pairwise comparison. Cách phòng: **swap order** và yêu cầu kết quả nhất quán — chính là cơ chế trong `check_position_bias()`.

### 2.3. Trade-off Chi phí ↔ Chất lượng
- Claude Opus + GPT-4o cho chất lượng chấm cao nhưng đắt. Đề xuất **giảm ~30% chi phí eval mà không giảm độ chính xác**:
  1. **Cascade/Tiered judging:** dùng model rẻ (vd `claude-haiku-4-5`) chấm trước; chỉ leo thang lên Opus/GPT-4o khi 2 Judge rẻ **bất đồng** (agreement thấp). Phần lớn case "dễ" sẽ được chốt rẻ.
  2. **Prompt caching:** rubric + system prompt giống nhau ở mọi case → cache phần tĩnh, chỉ trả tiền token động.
  3. **Batch / max_tokens hợp lý:** giới hạn output (chỉ cần JSON ngắn) cắt chi phí output token (vốn đắt gấp 4–5× input).

### 2.4. MRR (liên hệ phần Retrieval của nhóm Data)
- MRR = trung bình `1/rank` của tài liệu đúng đầu tiên. Nó cho biết retriever đặt đúng context **ở vị trí nào**. Đây là điều kiện cần: nếu Retrieval kém thì điểm Judge thấp **không phải lỗi Generation** — Judge engine của tôi báo điểm thấp, nhưng root cause nằm ở pipeline retrieval.

---

## 3. Problem Solving (Giải quyết vấn đề)

| Vấn đề gặp phải | Cách xử lý |
| :--- | :--- |
| **Repo phải chấm được dù không có API key** (giám khảo không có key của nhóm) | Thiết kế **soft-import SDK** + **auto-fallback MOCK**: nếu thiếu SDK/key thì mỗi Judge sinh điểm *deterministic* dựa trên độ trùng lặp từ vựng + jitter theo từng model. `python main.py` và `check_lab.py` chạy end-to-end không cần key. |
| **API có thể lỗi giữa chừng** (rate limit, mạng) | `_safe_judge()` bọc try/except: 1 Judge lỗi → fallback mock cho riêng Judge đó, **không làm vỡ cả pipeline**. |
| **Cần test được logic consensus/conflict mà không có model thật** | Mock có **jitter seeded theo `model|question|answer`** (hashlib + `random.Random`) → tạo bất đồng nhẹ, có thể tái lập, để kiểm thử nhánh agreement/conflict. |
| **Structured output không ép được min/max điểm** | Clamp thủ công về `[1, 5]` trong `_verdict_from_data()` thay vì tin tuyệt đối vào model. |
| **`UnicodeEncodeError` (cp1252) khi in tiếng Việt trên Windows console** | Xác định đây là lỗi *console*, không phải lỗi logic; chạy với `PYTHONIOENCODING=utf-8`. File ghi ra đều dùng `encoding="utf-8"`. |
| **Bảo mật API key** | Key chỉ đọc qua biến môi trường (`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`), **không hardcode**; `.env` đã nằm trong `.gitignore`, không push lên GitHub. |

---

## 4. Tự đánh giá & Hướng phát triển
- **Đã đạt:** ≥2 Judge cross-provider, Agreement Rate, xử lý xung đột tự động, position bias, cost report, chạy được không cần key.
- **Có thể cải tiến tiếp:** thêm Cohen's Kappa, triển khai cascade judging để hiện thực hoá việc giảm 30% chi phí, và mở rộng position-bias check sang cặp Judge thứ hai (GPT-4o) để so sánh mức thiên vị giữa các provider.
