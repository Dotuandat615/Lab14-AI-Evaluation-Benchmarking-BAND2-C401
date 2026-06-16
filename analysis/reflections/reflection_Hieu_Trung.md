# Báo cáo Cá nhân (Individual Reflection)

- **Họ tên:** Hieu Trung
- **Lab:** Day 14 - AI Evaluation Factory (Team Edition)
- **Nhánh làm việc:** `HieuTrung`
- **Ngày:** 2026-06-16

---

## 1. Vai trò và đóng góp của tôi

Trong dự án nhóm, tôi phụ trách chính phần **Regression Release Gate**, thành phần tự động so sánh Agent V1 và Agent V2 để quyết định nên **Release** hay **Rollback**. Đóng góp của tôi tập trung vào việc biến kết quả benchmark thành một cơ chế ra quyết định có ngưỡng rõ ràng, thay vì chỉ nhìn vào điểm trung bình.

Cụ thể, tôi đã:

- Xây dựng `engine/regression_gate.py` để so sánh summary của `Agent_V1_Legacy` và `Agent_V2_Optimized`.
- Định nghĩa các ngưỡng chất lượng và vận hành: `avg_score`, `hit_rate`, `mrr`, `agreement_rate`, chi phí, latency trung bình, p95 latency và tổng runtime.
- Tích hợp gate vào `main.py` để benchmark chạy theo luồng V1 vs V2, sinh `baseline_summary.json`, `summary.json` và `regression_comparison.json`.
- Thêm `LegacyAgentV1` làm baseline để có đối chiếu rõ với agent V2.
- Viết unit test trong `tests/test_regression_gate.py` cho các trường hợp Release, Rollback vì giảm chất lượng, tăng chi phí, tăng latency và baseline cost bằng 0.

## 2. Những gì tôi học được

- **Regression testing giúp tránh cải tiến "cảm tính":** V2 chỉ được release khi vượt qua ngưỡng chất lượng và không làm tệ hơn chi phí/hiệu năng. Trong run hiện tại, V2 đạt avg score **4.11 / 5.0**, hit rate **94.5%**, agreement rate **86.4%** và gate quyết định **Release**.
- **Cần đo cả chất lượng lẫn vận hành:** một agent có câu trả lời tốt nhưng quá chậm hoặc quá đắt vẫn có thể không phù hợp để release. Vì vậy gate kiểm tra thêm avg latency **0.26s**, p95 **0.27s** và avg eval cost **3.448e-05 USD**.
- **So sánh delta quan trọng hơn nhìn một snapshot:** V2 tăng avg score khoảng **+1.59**, giảm latency khoảng **67%** và giảm chi phí khoảng **28%** so với V1. Những con số này làm quyết định Release có cơ sở hơn.
- **Multi-Judge cần được đưa vào gate:** nếu agreement rate thấp, điểm cao có thể không đáng tin. Vì vậy tôi đưa `agreement_rate_min` vào ngưỡng chặn để gate không release một kết quả đánh giá thiếu ổn định.

## 3. Khó khăn gặp phải và cách xử lý

| Khó khăn | Cách xử lý |
|----------|------------|
| Chọn ngưỡng Release/Rollback sao cho vừa nghiêm túc vừa không quá gắt | Đặt ngưỡng tối thiểu cho chất lượng (`avg_score >= 4.0`, `hit_rate >= 0.80`, `agreement_rate >= 0.70`) và ngưỡng regression riêng cho delta so với V1. |
| Chi phí và latency có hướng "càng thấp càng tốt", khác với score/hit rate | Tách logic thành drop check cho quality metrics và increase check cho cost/performance metrics. |
| Baseline cost có thể bằng 0 làm phép tính phần trăm bị lỗi | Thêm trường hợp skip an toàn, trả warning thay vì crash, và viết test riêng cho case này. |
| Kết quả benchmark có nhiều metric nằm ở các file summary khác nhau | Chuẩn hóa output của `main.py` để summary V2 chứa cả `regression`, đồng thời lưu file comparison riêng để dễ trace lại V1 vs V2. |

## 4. Insight từ benchmark và failure analysis

Kết quả group report cho thấy Retrieval tổng thể khá tốt, nhưng nút thắt chính nằm ở **Generation/Prompting**. Nhóm `hard` có Retrieval gần như tốt nhưng điểm generation thấp hơn, vì agent còn xu hướng dán nguyên văn chunk thay vì tổng hợp câu trả lời.

Điều này giúp tôi hiểu rằng Regression Gate không chỉ là "máy chấm điểm cuối cùng". Gate cần bảo vệ cả những rủi ro hệ thống:

1. Nếu chỉ nhìn pass-rate, ta có thể bỏ sót các case biên 3.0-3.5.
2. Nếu chỉ nhìn generation score, ta có thể không thấy retrieval adversarial bị che bởi intent-rule cứng.
3. Nếu không ghi rõ judge đang ở mode **mock**, các con số tuyệt đối có thể tạo cảm giác an toàn giả.
4. RAGAS faithfulness/relevancy hiện là placeholder 0.9/0.85, nên regression gate cần được cập nhật lại khi nối RAGAS thật.

## 5. Bài học và hướng cải tiến cá nhân

- Lần sau tôi muốn bổ sung gate theo từng nhóm độ khó (`easy`, `medium`, `hard`, `adversarial`) để tránh việc điểm trung bình che lấp một nhóm đang tệ.
- Tôi muốn thêm cơ chế fail nếu conflict rate của Multi-Judge vượt ngưỡng, vì conflict rate hiện tại còn khoảng 25%.
- Khi có API key thật, cần chạy lại benchmark ở live mode để xác nhận gate vẫn Release với điểm Judge thật, không chỉ với mock deterministic judge.
- Tôi học được cách truy vấn để ra quyết định engineering: không chỉ "agent tốt hơn không", mà là "tốt hơn bao nhiêu, có đổi chi phí/latency không, và kết quả đánh giá có đáng tin không".

---

## 6. Đóng góp cho nhóm (tự đánh giá)

- [x] Xây dựng Regression Release Gate cho V1 vs V2.
- [x] Tích hợp gate vào benchmark pipeline và output reports.
- [x] Thêm baseline `LegacyAgentV1` để có đối chiếu delta.
- [x] Viết unit tests cho các quyết định Release/Rollback quan trọng.
- [x] Đọc và liên kết kết quả gate với group failure analysis.
- [ ] Đề xuất cải tiến tiếp theo: gate theo difficulty, conflict-rate threshold và live judge verification.
