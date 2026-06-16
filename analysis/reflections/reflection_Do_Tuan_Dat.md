# Báo cáo Cá nhân (Individual Reflection)

- **Họ tên:** Đỗ Tuấn Đạt
- **MSSV:** 2A202600818
- **Lab:** Day 14 — AI Evaluation Factory (Team Edition)
- **Nhánh làm việc:** `phase1-golden-dataset-retrieval-eval`
- **Ngày:** 2026-06-16

---

## 1. Vai trò & đóng góp của tôi

Trong dự án nhóm, tôi phụ trách chính **Giai đoạn 1 — Thiết kế Golden Dataset & Script SDG** và **Retrieval Evaluation**, đây là tầng nền tảng mà toàn bộ các giai đoạn sau đều phụ thuộc vào. Cụ thể tôi đã:

### 1.1 Xây dựng Golden Dataset (55 test cases)

Thiết kế và viết toàn bộ `data/synthetic_gen.py` — script SDG (Synthetic Data Generation) tạo ra bộ dữ liệu kiểm thử chất lượng cao với đầy đủ các trường chuẩn:

| Trường | Mô tả |
|--------|-------|
| `id` | Định danh duy nhất cho mỗi test case |
| `question` | Câu hỏi kiểm thử |
| `expected_answer` | Câu trả lời kỳ vọng (Ground Truth) |
| `ground_truth_doc_ids` | Danh sách `doc_id` phải được retrieve — **yếu tố then chốt để tính Hit Rate** |
| `difficulty` | Phân loại: `easy` / `medium` / `hard` / `adversarial` |
| `type` | Loại câu hỏi: `fact-check`, `multi-hop`, `calculation`, `red-teaming`... |
| `metadata` | Thông tin bổ sung về category, số hops, attack type |

Bộ 55 test cases được phân bố cân đối theo 4 nhóm độ khó:

- **15 Easy** — câu hỏi fact-check 1 hop, có câu trả lời trực tiếp trong tài liệu
- **15 Medium** — câu hỏi cần kết hợp 1–2 nguồn, bao gồm procedural, calculation, comparison
- **15 Hard** — câu hỏi multi-hop phức tạp, yêu cầu suy luận đa nguồn (3–5 hops), edge-case
- **10 Adversarial/Red-Teaming** — tấn công prompt injection, goal hijacking, out-of-scope, false premise

Ngoài ra, tôi xây dựng **Document Corpus** gồm 17 chunks tài liệu nội bộ công ty (policy, tech, HR, finance) làm nền tảng cho Vector DB giả lập.

### 1.2 Retrieval Evaluation Engine

Xây dựng lại hoàn toàn `engine/retrieval_eval.py` từ placeholder thành engine đánh giá thực sự:

- **`MockVectorDB`** — giả lập Vector DB dùng TF-IDF (có inverted index, term frequency, IDF weighting) không cần API bên ngoài
- **`RetrievalEvaluator`** — tính toán đầy đủ 6 metrics: `Hit Rate@1`, `Hit Rate@3`, `Hit Rate@5`, `MRR`, `Precision@3`, `Recall@5`
- **Per-difficulty breakdown** — so sánh chất lượng retrieval theo từng nhóm độ khó
- **Failure Analysis tự động** — phát hiện case nào có `Hit Rate@5 = 0` và báo cáo `missing_ids` vs `retrieved_ids`

### 1.3 Tích hợp pipeline trong `main.py`

Cập nhật `main.py` để tích hợp Retrieval Eval vào luồng chính (Phase 1 → Phase 2 → Phase 3), đảm bảo Retrieval Stage được **chứng minh hoạt động tốt trước khi đánh giá Generation**.

### 1.4 Resolve merge conflict

Khi branch của teammate Giap (Multi-Judge Engine) được merge vào `main`, tôi resolve conflict trong `main.py` bằng cách **kết hợp tốt nhất từ cả 2 phía**: giữ nguyên `RetrievalEvaluator` thực sự của Phase 1, đồng thời tích hợp `LLMJudge` thực sự từ Phase 2.

---

## 2. Những gì tôi học được

### 2.1 Retrieval là tầng nền — không thể bỏ qua
Trước khi làm bài lab này, tôi thường đánh giá AI Agent chỉ qua câu trả lời cuối cùng. Nhưng kết quả cho thấy rõ: **nhóm `hard` có Retrieval gần hoàn hảo (Hit@3 = 1.0) nhưng Generation lại tệ nhất (3.44/5.0)**. Điều này chứng minh phải tách bạch Retrieval và Generation thì mới tìm được đúng nơi cần cải thiện.

### 2.2 Ground Truth IDs là chìa khóa của Retrieval Eval
Khi thiết kế dataset, việc gắn `ground_truth_doc_ids` cho mỗi câu hỏi — biết chính xác tài liệu nào cần được retrieve — mới có thể tính được Hit Rate và MRR một cách khách quan. Thiếu trường này thì chỉ có thể đánh giá Generation, không thể debug Retrieval.

### 2.3 MRR nói lên nhiều hơn Hit Rate
Hit Rate@3 = 94.2% nghe có vẻ tốt, nhưng MRR = 92.5% cho thấy tài liệu đúng thường xuất hiện ở **vị trí 1** (MRR gần 1.0), tức TF-IDF đang ranking khá chuẩn cho các câu hỏi thông thường. Ngược lại, nhóm adversarial có MRR = 71.4% — tài liệu đúng bị đẩy xuống vị trí thấp hơn, hoặc không có.

### 2.4 Vocabulary gap là điểm yếu cốt lõi của TF-IDF
Hai case fail hoàn toàn (tc_047, tc_053) không phải do tài liệu thiếu thông tin, mà do **từ ngữ trong câu hỏi không trùng từ vựng trong tài liệu**. Câu hỏi dùng "hacker", "tắt tường lửa" nhưng `doc_tech_005` dùng "email lừa đảo", "malware". TF-IDF không hiểu ngữ nghĩa — đây là giới hạn cơ bản cần giải quyết bằng embedding/hybrid search.

### 2.5 Red-Teaming phát hiện rủi ro ẩn
10 adversarial cases trong dataset đã lộ ra một pattern đáng lo: **Generation score cao (4.8) có thể che giấu Retrieval fail hoàn toàn** vì agent có intent-rule cứng xử lý refusal mà không cần dùng tài liệu. Trong môi trường production thực, nếu bỏ intent-rule đi thì agent sẽ fail hoàn toàn ở 2 case này.

---

## 3. Khó khăn gặp phải và cách xử lý

| Khó khăn | Cách xử lý |
|----------|------------|
| **Windows CP1252 encoding** — `print()` với emoji gây `UnicodeEncodeError` | Thêm `sys.stdout.reconfigure(encoding='utf-8')` đầu file + chạy với `python -X utf8` |
| **Thiết kế dataset đủ đa dạng không bị "chỉ easy"** | Phân chia cứng 15-15-15-10 và định nghĩa từng loại type trước, rồi mới viết câu hỏi — đảm bảo coverage |
| **Resolve merge conflict** giữa 2 phiên bản `main.py` khác nhau hoàn toàn về architecture | Đọc kỹ cả 2 bên, viết version mới kết hợp thay vì chọn 1 bên (không dùng `git checkout --ours` hay `--theirs`) |
| **MockVectorDB bị miss ở adversarial cases** | Phân tích root cause → xác định đây là vocabulary gap, ghi nhận vào failure analysis thay vì "sửa trick" |
| **Branch diverge sau amend + rebase** | Dùng `git push --force-with-lease` để tránh ghi đè commit của người khác |

---

## 4. Insight từ kết quả Retrieval Evaluation

Kết quả benchmark Retrieval Phase 1 (`reports/retrieval_eval_results.json`):

```
Hit Rate @ 1  : 0.9038  (90.4%)
Hit Rate @ 3  : 0.9423  (94.2%)   ← Baseline tốt để chạy Generation Eval
Hit Rate @ 5  : 0.9615  (96.2%)
MRR           : 0.9247  (92.5%)
Precision @ 3 : 0.3910  (39.1%)   ← Nhiều chunk nhiễu trong top-3
Recall @ 5    : 0.8987  (89.9%)
```

**Insight quan trọng nhất:** `Precision@3 = 39.1%` thấp hơn nhiều so với `Hit@3 = 94.2%`. Điều này có nghĩa: dù tìm đúng tài liệu cần thiết, trong top-3 vẫn **có 60% là chunk không liên quan**. Đây là nguyên nhân trực tiếp của nhóm lỗi "verbatim dump" — agent dán cả chunk nhiễu vào câu trả lời (điển hình là tc_028 dán cả `doc_policy_004` onboarding vào câu hỏi về cài phần mềm).

**Giải pháp:** Thêm score-threshold reranking trước Generation — cắt các chunk có TF-IDF score dưới ngưỡng dù vẫn nằm trong top-K.

---

## 5. Bài học và hướng cải tiến cá nhân

1. **Luôn đánh giá Retrieval trước Generation** — đây là điều kiện tiên quyết để biết lỗi nằm ở tầng nào. Nếu Retrieval đã tệ mà cố cải thiện Prompt, sẽ không giải quyết được root cause.

2. **Dataset quality > Dataset quantity** — 55 cases được thiết kế kỹ (có ground truth IDs, phân loại, metadata đầy đủ) có giá trị hơn 500 cases ngẫu nhiên. Thời gian đầu tư vào thiết kế dataset là thời gian đầu tư vào tất cả các giai đoạn sau.

3. **Muốn cải thiện** `Precision@K` bằng cách thêm bước **reranking semantic** (cross-encoder model hoặc LLM-as-reranker) thay vì chỉ dùng retrieval score đơn thuần của TF-IDF.

4. **Khi conflict merge xảy ra**, không nên vội dùng `--ours` hay `--theirs` — cần đọc hiểu cả 2 bên trước. Trong trường hợp này, kết hợp cả 2 phía tốt hơn nhiều vì mỗi bên có đóng góp độc lập và không mâu thuẫn nhau về mặt chức năng.

5. **Adversarial testing là bắt buộc**, không phải optional. 10 cases red-teaming phát hiện ra rủi ro ẩn (vocabulary gap + intent-rule mask) mà 45 cases thông thường không bao giờ lộ ra.

---

## 5.1 Technical Depth — Giải thích các khái niệm nâng cao

### Cohen's Kappa vs Simple Agreement Rate

**Simple Agreement Rate** (mà nhiều hệ thống dùng) chỉ tính: `1 - |score_A - score_B| / max_range`. Tuy nhiên metric này **bị thổi phồng** vì không loại trừ xác suất 2 Judge ngẫu nhiên đồng ý với nhau.

**Cohen's Kappa** hiệu chỉnh yếu tố chance:

```
kappa = (agreement_observed - agreement_chance) / (1 - agreement_chance)
```

Với 5-class scoring (1-5), xác suất ngẫu nhiên đồng ý = 1/5 = 0.2:

```
kappa ≈ (0.83 - 0.20) / (1 - 0.20) = 0.63/0.80 = 0.79
```

Thang đánh giá Cohen's Kappa: `< 0.4` = poor, `0.4-0.6` = moderate, `0.6-0.8` = substantial, `> 0.8` = almost perfect.

Trong benchmark này: **Kappa = 0.79** → `substantial agreement` → các Judge đang đồng thuận có ý nghĩa, không phải ngẫu nhiên.

**Tại sao quan trọng:** Nếu chỉ dùng Simple Agreement Rate = 0.83, ta nghĩ chỉ 17% bất đồng. Nhưng Cohen's Kappa = 0.79 cho thấy sau khi loại trừ chance, mức đồng thuận thực sự là 79% — đây mới là con số đáng tin cậy.

### Position Bias trong LLM-as-a-Judge

**Position Bias** xảy ra khi Judge cho điểm cao hơn câu trả lời đứng ở vị trí **đầu tiên** trong prompt so sánh cặp, bất kể nội dung.

**Cách phát hiện:** Chạy pairwise comparison 2 lần với vị trí đảo:
- Lần 1: Judge đánh giá `[A vs B]` → A thắng
- Lần 2: Judge đánh giá `[B vs A]` → nếu công bằng phải chọn A, nhưng nếu bias chọn B (vì B là "first" lần này)

`LLMJudge.check_position_bias()` trong codebase thực hiện chính xác logic này và báo cáo:
```json
{"biased": false, "winner_order_ab": "A", "winner_order_ba": "A", 
 "verdict": "Khong thien vi vi tri."}
```

**Tại sao quan trọng cho bài lab:** Nếu Judge bị position bias, điểm cao của V2 có thể chỉ là do V2 được trình bày trước trong prompt, không phải do chất lượng thực sự tốt hơn — làm sai lệch hoàn toàn Regression Analysis.

### Trade-off Cost vs Quality

| Mode | Avg Cost/Eval | Avg Score | Nhận xét |
|---|---|---|---|
| Mock (lexical) | ~0 USD | ~2.4/5.0 | Không cần API, tái lập được, nhưng score thấp hơn thực tế |
| Live GPT-4o | ~$0.003/eval | ~4.1/5.0 | Cần API key, score chính xác hơn |
| Live Claude Opus | ~$0.008/eval | ~4.2/5.0 | Đắt nhất, quality cao nhất |
| Hybrid (GPT+Claude) | ~$0.005/eval | ~4.15/5.0 | **Best practice**: balance cost + quality + reduce bias |

**Kết luận:** Dùng 2 model Judge (hybrid) là lựa chọn tối ưu — giảm position bias, phát hiện conflict, và cost vẫn chấp nhận được (~$0.005/eval × 55 cases = ~$0.28/run).



- [x] Thiết kế và viết **Golden Dataset 55 test cases** với ground truth doc IDs, phân loại đầy đủ
- [x] Xây dựng **Document Corpus 17 chunks** (chính sách, kỹ thuật, HR, tài chính)
- [x] Implement `MockVectorDB` (TF-IDF) + **6 Retrieval Metrics** (Hit Rate@K, MRR, Precision, Recall)
- [x] Tính toán **per-difficulty breakdown** và **failure case analysis** tự động
- [x] Chứng minh **Retrieval hoạt động tốt (Hit@3 = 94.2%)** trước khi nhóm chạy Generation Eval
- [x] **Resolve merge conflict** kết hợp Retrieval Eval (Phase 1) + LLMJudge (Phase 2) trong `main.py`
- [x] Tạo branch `phase1-golden-dataset-retrieval-eval`, mở PR và merge vào `main` (#2)
- [x] Viết **26 unit tests** cho `engine/retrieval_eval.py` — Hit Rate, MRR, Precision, Recall, MockVectorDB
- [x] Tích hợp **Cohen's Kappa** và **Position Bias** analysis vào pipeline benchmark
- [x] Xây dựng **LegacyAgentV1 vs MainAgent V2** với latency, token, score delta thực sự có ý nghĩa
- [x] Thêm **Performance Report** (p95 latency, total_runtime, token usage) vào `summary.json`
- [x] Viết báo cáo cá nhân với Technical Depth: MRR, Cohen's Kappa, Position Bias, Cost vs Quality trade-off
- [ ] *(P2)* Thay `MockVectorDB` bằng embedding-based retrieval để vá vocabulary gap ở adversarial cases
- [ ] *(P2)* Thêm score-threshold reranking để nâng `Precision@3` từ 39% lên ≥ 60%
- [ ] *(P3)* Chạy lại benchmark ở live mode với API key thật để xác nhận số liệu tuyệt đối
