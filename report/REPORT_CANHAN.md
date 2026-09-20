# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Đức Long
**Nhóm:** K4-L3B (G-66) 
**Ngày:** 20/09

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (gần bằng 1) nghĩa là góc giữa hai vector embedding trong không gian đa chiều rất nhỏ, thể hiện rằng hai đoạn văn bản có sự tương đồng sâu sắc về mặt ngữ nghĩa (semantic meaning), bất kể độ dài hay từ ngữ bề mặt khác nhau.

**Ví dụ có độ tương tự CAO:**
- **Câu A:** "Người mua có 7 ngày để gửi khiếu nại trả hàng."
- **Câu B:** "Khách hàng được quyền yêu cầu hoàn tiền trong vòng 1 tuần kể từ khi nhận hàng."
- **Tại sao tương đồng:** Dù hai câu sử dụng từ vựng hoàn toàn khác nhau ("người mua" vs "khách hàng", "7 ngày" vs "1 tuần", "khiếu nại trả hàng" vs "yêu cầu hoàn tiền"), mô hình embedding hiểu được chúng có cùng ngữ nghĩa quy định thời hạn đổi trả.

**Ví dụ có độ tương tự THẤP:**
- **Câu A:** "Người mua có 7 ngày để gửi khiếu nại trả hàng."
- **Câu B:** "Người bán phải chịu phí phạt 200.000 VNĐ nếu từ chối nghĩa vụ bảo hành."
- **Tại sao khác:** Cả hai câu đều liên quan đến quy định sàn e-commerce nhưng nói về hai đối tượng hoàn toàn khác nhau (người mua vs người bán) và hai mảng quy định độc lập (trả hàng vs phạt bảo hành).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine similarity chỉ đo **hướng** (direction) của vector trong không gian ngữ nghĩa mà không bị ảnh hưởng bởi độ dài (magnitude) của vector hay độ dài câu văn. Ngược lại, khoảng cách Euclid bị biến dạng khi độ dài vector/văn bản thay đổi, dễ dẫn đến việc so sánh sai lệch.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* 
> Công thức: ceil((độ_dài − overlap) / (chunk_size − overlap)) = ceil((10000 − 50) / (500 − 50)) = ceil(9950 / 450) = ceil(22.111...) = 23.
> *Đáp án:* 23 chunks. (Đã kiểm chứng thành công bằng `FixedSizeChunker(chunk_size=500, overlap=50).chunk('a'*10000)` trả về đúng 23).

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Khi overlap tăng lên 100, số lượng chunk tăng từ 23 thành 25 chunks. Ta muốn overlap lớn hơn để tránh bị đứt gãy ngữ cảnh nằm ở ranh giới giữa 2 chunk (boundary cases), giúp đảm bảo thông tin câu từ liên tục không bị cắt đôi chéo giữa các chunk kề nhau.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Sử dụng regex lookbehind `r"(?<=[.!?])\s+"` để tách câu tại các ranh giới sau dấu chấm, chấm hỏi, chấm cảm mà vẫn giữ nguyên dấu câu đính kèm vào cuối câu. Sau đó, các câu được gom nhóm theo số lượng `max_sentences_per_chunk` và loại bỏ khoảng trắng thừa. Xử lý trường hợp text rỗng/chỉ chứa khoảng trắng bằng cách trả về `[]`.
> (Edge case chưa xử lý hoàn hảo: các từ viết tắt như "TS.", "v.v." hay số thập phân như "3.14" sẽ bị ngắt nhầm thành ranh giới câu).

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Áp dụng danh sách separator theo thứ tự ưu tiên `["\n\n", "\n", ". ", " ", ""]` để tách theo khối ngữ nghĩa lớn trước. Thuật toán có 2 chiều: (1) **Đệ quy xuống:** phân tách văn bản bằng separator hiện tại, mảnh nào vượt quá `chunk_size` sẽ tiếp tục đệ quy xuống các separator nhỏ hơn; (2) **Gom lên (Merge):** các mảnh nhỏ liền kề được nối lại với nhau đến sát ngưỡng `chunk_size` để tránh sinh ra các chunk vụn. Base case dừng khi văn bản `chunk_size` hoặc danh sách separator rỗng / `""` (khi đó fallback chia theo ký tự).

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Lưu trữ các bản ghi (records) dạng dictionary trong bộ nhớ `self._store` (in-memory). Hàm `_make_record` chuẩn hóa Document, nhúng vector bằng `self._embedding_fn`, sao chép metadata và đảm bảo có khóa `doc_id`. Khi `search`, câu hỏi được embed thành vector rồi tính tích vô hướng (dot product) với toàn bộ vector lưu trữ, sắp xếp giảm dần theo điểm tương đồng và trả về top-k.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> `search_with_filter` thực hiện **Lọc trước (Pre-filtering)** trên `self._store` theo các tiêu chí trong `metadata_filter`, sau đó mới chạy `_search_records` trên tập ứng viên đã lọc để đảm bảo không bị mất các tài liệu phù hợp. `delete_document` duyệt xóa mọi bản ghi có `metadata['doc_id']` hoặc `id` khớp với `doc_id` truyền vào, trả về `True` nếu có bản ghi bị xóa.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Thực hiện quy trình RAG 3 bước: (1) Kiểm tra store rỗng thì trả lời ngay câu thông báo; nếu có dữ liệu thì gọi `store.search` lấy top-k chunk liên quan nhất; (2) Định dạng từng chunk có gắn nhãn nguồn `[1] (Nguồn: doc_id)` đưa vào ngữ cảnh Prompt, bổ sung ràng buộc nghiêm ngặt chống hallucinations; (3) Truyền prompt cho `llm_fn` để sinh câu trả lời có khả năng trích dẫn nguồn.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

======================================= test session starts ========================================
platform win32 -- Python 3.11.0, pytest-9.1.1, pluggy-1.6.0 -- D:\AI_thuc_chien\day7\K4-L3B-Data-Foundations\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\AI_thuc_chien\day7\K4-L3B-Data-Foundations
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED         [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                  [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED           [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED            [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                 [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED       [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED        [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED      [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                        [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED        [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                   [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED               [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                         [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED    [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED    [ 42%] 
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                        [ 45%] 
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED          [ 47%] 
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED            [ 50%] 
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                  [ 52%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED       [ 54%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED         [ 57%] 
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED          [ 61%] 
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                   [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                  [ 66%] 
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED             [ 69%] 
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED         [ 71%] 
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED    [ 73%] 
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED        [ 76%] 
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED              [ 78%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED        [ 80%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED   [ 85%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED  [ 88%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

======================================== 42 passed in 0.08s ======================================== 

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Người mua có 7 ngày để gửi khiếu nại trả hàng. | Khách hàng được quyền yêu cầu hoàn tiền trong vòng 1 tuần kể từ khi nhận hàng. | cao | 0.8120 | Đúng |
| 2 | Người mua có 7 ngày để gửi khiếu nại trả hàng. | Người bán phải chịu phí phạt 200.000 VNĐ nếu từ chối nghĩa vụ bảo hành. | thấp | 0.6950 | Đúng |
| 3 | Không áp dụng chính sách đổi ý cho sản phẩm mua tại Shopee Mart. | Sản phẩm mua trên Shopee Mart không được trả lại vì lý do đổi ý. | cao | 0.9610 | Đúng |
| 4 | Thời hạn xử lý khiếu nại của người bán là 02 ngày làm việc. | Mở ứng dụng Shopee, chọn mục Tôi để tạo yêu cầu trả hàng. | thấp | 0.6278 | Đúng |
| 5 | Quy định bảo hành áp dụng cho mọi thiết bị điện tử. | Hàng vỡ do vận chuyển được khiếu nại trong 24 giờ. | thấp | 0.6831 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Cặp số 3 đạt điểm tương đồng gần như tuyệt đối (0.9610) và Cặp số 1 đạt 0.8120 với mô hình `gemini-embedding-001` mặc dù sử dụng các từ vựng khác biệt ("người mua" vs "khách hàng", "7 ngày" vs "1 tuần"). Điều này chứng minh rằng Vector Embeddings của Gemini bắt được bản chất ngữ nghĩa sâu sắc của câu trong không gian đa chiều (semantic vector space) thay vì chỉ so sánh trùng khớp từ khóa bề mặt (lexical matching).

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Tôi cần thực hiện những bước nào để gửi yêu cầu Trả hàng/Hoàn tiền trực tiếp trên ứng dụng Shopee? | `shopee-buyer-return-request-guide`: Cách 1 - Bước 1: Mở ứng dụng Shopee, vào mục Tôi > Đã giao... | 0.336 | Có | Thực hiện theo 8 bước: Mở ứng dụng Shopee, chọn mục Tôi, chọn đơn hàng cần trả, bấm Trả hàng/Hoàn tiền... [1] |
| 2 | Thời hạn tối đa để xử lý và phản hồi khiếu nại Trả hàng/Hoàn tiền là bao nhiêu ngày? *(dùng `filter={"audience": "seller"}`)* | `shopee-seller-return-processing-policy`: Người bán có tối đa 02 ngày làm việc để xử lý... | 0.269 | Có | Người bán có tối đa 02 ngày làm việc kể từ khi nhận thông báo khiếu nại để chọn đồng ý hoặc phản hồi. [1] |
| 3 | Nếu Người bán Shopee Mall vi phạm quy định bán hàng giả, hàng nhái thì bị phạt bao nhiêu tiền và xử lý như thế nào? | `shopee-mall-seller-terms-of-service`: Phí phạt 9.818.180 VNĐ hoặc 100% giá trị sản phẩm... | 0.280 | Có | Người bán bị phạt 9.818.180 VNĐ hoặc 100% giá trị sản phẩm và bị loại khỏi Shopee Mall nếu vi phạm 2 lần. [1] |
| 4 | Trường hợp nào Người mua được miễn 100% cước phí vận chuyển hoàn trả hàng? | `shopee-return-shipping-fee-policy`: Miễn 100% cước khi chọn Lấy hàng tại nhà hoặc Gửi tại bưu cục SPX... | 0.282 | Có | Người mua được miễn 100% cước trả hàng khi sử dụng hình thức Lấy hàng tại nhà hoặc Gửi tại bưu cục SPX. [1] |
| 5 | Những trường hợp/lý do nào không được áp dụng chính sách trả hàng với lý do 'Đổi ý'? | `shopee-general-return-policy`: Lý do đổi ý không áp dụng cho sản phẩm hạn chế, Shopee Mart... | 0.452 | Có | Không áp dụng lý do đổi ý cho sản phẩm thuộc danh sách hạn chế trả hàng hoặc mua tại Shopee Mart. [1] |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Qua phần demo và phân tích Failure Analysis của nhóm trong `REPORT_NHOM.md`, tôi nhận ra bài học quan trọng: **Document Hit@3 cao không đồng nghĩa với Complete Evidence**. Mặc dù các chiến lược như `SentenceChunker` hay `FixedSizeChunker` đạt tỉ lệ tìm đúng tài liệu 5/5, nhưng do chunk quá vụn nên thiếu thông tin đầy đủ để Agent trả lời (Complete Evidence@3 thấp). Việc nhóm chuyển sang thử nghiệm và lựa chọn **`HeadingChunker` (ngưỡng 1400 / Header Prefixing)** giúp bảo toàn cấu trúc cây điều khoản, giữ trọn vẹn quy trình nhiều bước và đạt tỉ lệ Complete Evidence@3 tuyệt đối (5/5). Thêm vào đó, việc kết hợp `Metadata Pre-filtering` triệt tiêu hoàn toàn nhiễu giữa tài liệu Người mua và Người bán.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
