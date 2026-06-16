# Báo cáo Cá nhân (Individual Reflection)

- **Họ tên:** Phan Văn Hiếu
- **Lab:** Day 14 — AI Evaluation Factory (Team Edition)
- **Nhánh làm việc:** `HieuTrung`
- **Ngày:** 2026-06-16

> *Ghi chú: nếu vai trò chính của mình khác với mô tả bên dưới, hãy chỉnh lại mục 1 cho khớp thực tế nhóm.*

---

## 1. Vai trò & đóng góp của tôi

Trong dự án nhóm, tôi tập trung vào **Giai đoạn 3 — Chạy Benchmark, Phân cụm lỗi (Failure Clustering) và Phân tích "5 Whys"**, đầu ra chính là báo cáo [`analysis/failure_analysis.md`](../failure_analysis.md). Cụ thể tôi đã:

- Chạy đầy đủ pipeline đánh giá: `python data/synthetic_gen.py` → `python main.py`, sinh ra `reports/summary.json`, `reports/benchmark_results.json`, `reports/retrieval_eval_results.json`.
- Trích xuất số liệu **thật** từ các report (không dùng số ước lượng) để điền báo cáo phân tích thất bại.
- Phân cụm lỗi theo **tầng pipeline** (Retrieval / Generation / Prompting) thay vì chỉ liệt kê triệu chứng.
- Thực hiện **5 Whys** cho 3 case tệ nhất và đề xuất Action Plan có ưu tiên (P0–P3).

## 2. Những gì tôi học được

- **"Không đo được thì không cải thiện được":** một con số tổng (pass-rate 96.4%) có thể *che giấu* vấn đề. Khi tách theo độ khó, nhóm `hard` chỉ đạt Judge **3.44/5** dù pass — cho thấy phải nhìn metric *phân tầng*, không chỉ trung bình.
- **Tách bạch Retrieval và Generation:** trước đây tôi nghĩ trả lời sai = retriever sai. Thực tế cho thấy có thể **retrieve đúng nhưng sinh sai** (nhóm `hard` Hit@3 = 1.0 nhưng Generation kém nhất), và ngược lại **retrieve sai nhưng Judge vẫn cao** (tc_047/tc_053 bị intent-rule cứng che lấp).
- **Multi-Judge & độ tin cậy:** hiểu vì sao chỉ tin 1 Judge là rủi ro — Agreement Rate **0.86** nhưng Conflict Rate vẫn **25%**, nghĩa là cần cơ chế xử lý xung đột (median/min cho safety) và có thể cần Judge thứ 3.
- **Regression Gate:** ý nghĩa của việc tự động hoá quyết định Release/Rollback dựa trên ngưỡng chất lượng/chi phí/hiệu năng (V2 vượt V1: avg_score +1.59, latency −67%, cost −28% → RELEASE).

## 3. Khó khăn gặp phải & cách xử lý

| Khó khăn | Cách xử lý |
|----------|------------|
| Judge chạy ở chế độ **mock** (chưa có API key) → lo điểm không "thật" | Ghi chú rõ trong báo cáo rằng con số tuyệt đối là minh hoạ, nhưng *kết luận về nhóm lỗi* vẫn vững; đề xuất chạy lại ở mode `live` (P3). |
| Lỗi **UnicodeEncodeError** khi in tiếng Việt ra console Windows | Đặt `PYTHONIOENCODING=utf-8` / `reconfigure` stdout khi phân tích dữ liệu. |
| Phát hiện RAGAS faithfulness/relevancy là **hằng số placeholder** (0.9/0.85) trong `ExpertEvaluator.score()` | Không báo cáo như số thật; đánh dấu là P0 cần nối RAGAS thật. |
| Phân biệt "fail thật" và "fail bị che giấu" | Đọc kỹ `agent_metadata.sources` + per-criterion để thấy tc_047/053 fail Retrieval nhưng được Generation bù. |

## 4. Insight chính tôi rút ra từ dữ liệu

1. **Nút thắt thật sự nằm ở tầng Generation/Prompting, không phải Retrieval.** Hàm `_grounded_answer()` chỉ **dán nguyên văn chunk** thay vì tổng hợp → mọi câu `multi-hop`/`calculation` xuống cấp thành tra cứu 1-hop.
2. **Điểm yếu Retrieval của câu adversarial bị intent-rule cứng che lấp** — đây là *rủi ro ẩn* sẽ lộ ra nếu sản phẩm thật phải dựa hoàn toàn vào retrieval.
3. **Pass-rate cao ≠ chất lượng cao:** 14/55 case nằm vùng biên 3.0–3.5.

## 5. Bài học & hướng cải tiến cho bản thân

- Khi đánh giá AI, luôn **báo cáo metric phân tầng + nêu rõ giả định** (mode mock, placeholder) để không tạo cảm giác an toàn giả.
- Lần sau tôi muốn tự tay nối **RAGAS thật** và thử **embedding-based retrieval** để vá vocabulary gap (P2) — biến phần phân tích thành cải tiến đo lường được.
- Rèn thói quen truy lỗi tới **đúng tầng pipeline** (Ingestion → Chunking → Retrieval → Prompting) trước khi đề xuất giải pháp, thay vì sửa cảm tính.

---

## 6. Đóng góp cho nhóm (tự đánh giá)

- [x] Chạy benchmark & sinh toàn bộ reports.
- [x] Hoàn thiện `analysis/failure_analysis.md` với dữ liệu thật + 5 Whys.
- [ ] *(điền thêm phần mình phụ trách chung với nhóm: SDG / Multi-Judge / Regression Gate…)*
