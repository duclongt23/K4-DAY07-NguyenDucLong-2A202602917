# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm/lớp:** K4-L3B (G-66)  
**Thành viên nhóm:**  
- Thái Phúc Tiến
- Nguyễn Thành Luân 
- Trần Đình Duy
- Nguyễn Đức Long  
**Chủ đề:** Chính sách thương mại điện tử — Trả hàng/Hoàn tiền Shopee  
**Ngày:** 20/09/2026


> Báo cáo này tổng hợp corpus, benchmark và các cấu hình chunking đã chạy trong repo. Các kết quả chất lượng bên dưới dùng TF-IDF lexical baseline; không được diễn giải thành semantic/LLM accuracy.

---

## 1. Lựa chọn tài liệu (Document Set Quality) — Nhóm (10 điểm)

### Chủ đề & lý do chọn

Nhóm chọn các chính sách Trả hàng/Hoàn tiền Shopee vì corpus có cấu trúc Markdown rõ, nhiều điều kiện và phân biệt rõ đối tượng `buyer`/`seller`. Đây là domain phù hợp để quan sát tác động của chunking theo heading, metadata pre-filtering và khả năng giữ trọn quy trình nhiều bước.

### Danh sách tài liệu (Data Inventory)

| # | Tên tài liệu | Nguồn | Ngày lấy / phiên bản | Ký tự | Metadata |
|---:|---|---|---|---:|---|
| 1 | Buyer return request guide | [Shopee 79233](https://help.shopee.vn/portal/4/article/79233) | 20/09/2026 · 2024-v2 | 2.321 | buyer · returns-guide · vi |
| 2 | Buyer return timeline policy | [Shopee 77244](https://help.shopee.vn/portal/4/article/77244) | 20/09/2026 · 2024-v1 | 1.818 | buyer · returns-policy · vi |
| 3 | General return policy | [Shopee 188931](https://help.shopee.vn/portal/4/article/188931) | 20/09/2026 · not stated | 5.857 | buyer · returns · vi |
| 4 | Shopee Mall seller terms | [Shopee 77262](https://help.shopee.vn/portal/4/article/77262) | 20/09/2026 · not stated | 12.387 | seller · terms · vi |
| 5 | Return shipping fee policy | [Shopee 77247](https://help.shopee.vn/portal/4/article/77247) | 20/09/2026 · not stated | 1.464 | buyer · shipping-policy · vi |
| 6 | Seller return processing policy | [Shopee 77245](https://help.shopee.vn/portal/4/article/77245) | 20/09/2026 · 2024-v1 | 1.953 | seller · seller-policy · vi |
| 7 | Seller warranty responsibility | [Shopee 77246](https://help.shopee.vn/portal/4/article/77246) | 20/09/2026 · 2024-v1 | 1.761 | seller · warranty-policy · vi |

**Data governance:** 7/7 tài liệu dùng nguồn công khai, không chứa API key, thông tin cá nhân hay dữ liệu nội bộ. `data/exchange_policy/sources.csv` đối soát 1-1 với 7 file Markdown và có `source_url`, `retrieved_at`, `document_version`, `license_or_permission`.

### Cấu trúc Metadata

| Trường | Kiểu | Ví dụ | Giá trị cho retrieval |
|---|---|---|---|
| `doc_id` | string | `shopee-seller-return-processing-policy` | Nối chunk về tài liệu gốc và truy vết nguồn |
| `source_url` | string | URL Shopee Help | Kiểm tra provenance |
| `retrieved_at` | date | `2026-09-20` | Biết thời điểm snapshot |
| `document_version` | string | `2024-v1`, `not-stated` | Phân biệt phiên bản/hiệu lực |
| `audience` | enum | `buyer` / `seller` | Pre-filter để tránh lẫn chủ thể |
| `category` | string | `returns`, `terms`, `warranty-policy` | Lọc theo loại chính sách |
| `language` | string | `vi` | Hỗ trợ corpus đa ngôn ngữ về sau |
| `section_path` | string | `Policy > Mục 1 > ...` | Hiển thị heading context cho chunk |

---

## 2. Thiết kế chiến lược (Strategy Design) — Nhóm (15 điểm)

### Phân tích đường cơ sở

`ChunkingStrategyComparator().compare()` với `chunk_size=1000` trên ba tài liệu đầu:

| Tài liệu | Fixed size | Sentence (3 câu) | Recursive |
|---|---:|---:|---:|
| buyer-return-request-guide | 3 · 685,7 chars | 6 · 341,2 | 4 · 512,8 |
| buyer-return-timeline-policy | 2 · 766,5 | 6 · 254,2 | 2 · 765,5 |
| general-return-policy | 6 · 901,8 | 9 · 599,9 | 6 · 900,3 |

### Các cấu hình đã thử

| Vai trò thí nghiệm | Chiến lược | Lý do |
|---|---|---|
| Baseline A | Fixed 1000, overlap 100 | Đơn giản, ít chunk; overlap giảm mất chữ ở biên nhưng không hiểu heading |
| Baseline B | Sentence 3 câu | Chunk dễ đọc nhưng có nguy cơ tách các bước của một quy trình |
| Baseline C | Recursive 1000 | Ưu tiên paragraph/line/sentence/word, giữ cấu trúc tự nhiên |
| Custom | Heading 600 | Giữ heading nhưng ngưỡng nhỏ, dùng để kiểm tra trade-off |
| Custom | Heading 1000 | Giữ heading với ngưỡng cân bằng |
| Custom được chọn | **Heading 1400** | Giữ section chính sách dài, fallback recursive khi section quá lớn |

### So sánh định lượng

| Chiến lược | Chunks | Avg chars | Document Hit@3 | Complete Evidence@3 | Complete Top-1 | Avg coverage |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 1000 + overlap 100 | 31 | 897,6 | 5/5 | 5/5 | 4/5 | 100% |
| Sentence 3 câu | 68 | 372,2 | 5/5 | 1/5 | 1/5 | 65% |
| Recursive 1000 | 33 | 768,9 | 5/5 | 5/5 | 5/5 | 100% |
| Heading 600 | 74 | 429,1 | 5/5 | 5/5 | 1/5 | 100% |
| Heading 1000 | 50 | 575,5 | 5/5 | 5/5 | 5/5 | 100% |
| **Heading 1400** | **48** | **593,7** | **5/5** | **5/5** | **5/5** | **100%** |

Heading 1400 được chọn vì đạt chất lượng tương đương Recursive/Heading 1000 nhưng giữ được đường dẫn heading và dùng ít chunk hơn Heading 1000. Recursive có 33 chunk, ít hơn nữa, nên đây là trade-off chứ không phải bằng chứng Heading luôn tốt nhất.

---

## 3. Câu hỏi đánh giá & Chất lượng truy xuất (Retrieval Quality) — Nhóm (10 điểm)

### Câu hỏi và câu trả lời chuẩn

| # | Query | Gold answer rút gọn | Tài liệu/section |
|---:|---|---|---|
| 1 | Các bước gửi yêu cầu trên ứng dụng? | Tôi → chọn đơn → Trả hàng/Hoàn tiền → tình huống/lý do → bằng chứng → Gửi yêu cầu. | buyer request guide · Mục 1, Cách 1 |
| 2 | Seller có bao nhiêu thời gian phản hồi? | 02 ngày làm việc (48 giờ); quá hạn tự động chấp nhận. | seller processing · Mục 1.1; filter `audience=seller` |
| 3 | Phạt hàng giả tại Shopee Mall? | 9.818.180 VND hoặc 100% giá trị; 07 ngày; 02 lần bị loại Mall. | Mall terms · Mục 3.3 |
| 4 | Khi nào miễn 100% phí trả hàng? | Pickup hoặc Drop-off tại đối tác với mã vận đơn trả hàng. | shipping fee · Mục 1 |
| 5 | Ngoại lệ lý do “Đổi ý”? | Danh sách hạn chế, Shopee Mart, sản phẩm quy định riêng theo từng thời điểm. | general return · Mục 1.3 |

### Tổng hợp chất lượng với Heading 1400

| # | Top-1 chunk | Score | Top-3 có tài liệu gold? | Đủ evidence? |
|---:|---|---:|---|---|
| 1 | `buyer-return-request-guide#1` | 0,3317 | Có | Có |
| 2 | `seller-return-processing-policy#0` | 0,2983 | Có | Có |
| 3 | `mall-seller-terms-of-service#12` | 0,2756 | Có | Có |
| 4 | `return-shipping-fee-policy#0` | 0,3744 | Có | Có |
| 5 | `general-return-policy#5` | 0,3429 | Có | Có |

**Kết quả:** Document Hit@3 = 5/5, Complete Evidence@3 = 5/5, Complete Top-1 = 5/5. Theo proxy nghiêm ngặt 2 điểm cho mỗi top-1 complete, retrieval đạt 10/10; đây vẫn là phép đo phát triển trên 5 query, chưa phải hold-out accuracy.

### Metadata A/B

Ở Q2, không filter tìm trong 48 chunks; filter `audience=seller` thu hẹp còn 29 chunks. Top-1 vẫn là seller policy ở cả hai lượt, nhưng các chunk buyer bị loại khỏi top-3. Kết luận: filter giúp giảm nhiễu/chủ thể sai; trên query này chưa chứng minh tăng top-1 accuracy.

### Failure analysis

`SentenceChunker(3)` có Document Hit@3 = 5/5 nhưng Complete Evidence@3 = 1/5. Lý do là các bước của quy trình bị tách thành nhiều chunk, nên retrieval thấy đúng tài liệu nhưng không gom đủ bằng chứng. Heading 600 cũng đạt evidence top-3 nhưng chỉ Top-1 = 1/5; tăng ngưỡng 1000/1400 giúp một chunk chứa nhiều điều kiện hơn.

---

## 4. Thuyết trình (Demo) & Bài học nhóm — Nhóm (5 điểm)

### Ba insight trình bày

1. **Đúng tài liệu chưa đủ:** Sentence đạt document hit 5/5 nhưng thiếu evidence vì chunk quá nhỏ.
2. **Cấu trúc domain có giá trị:** Heading 1400 giữ path cha–con và đạt 5/5 top-1 complete trên benchmark.
3. **Metadata là pre-filter, không phải phép màu:** Q2 giảm candidate space 48 → 29 và loại nhiễu buyer, nhưng top-1 vốn đã đúng.

### Bài học

Chunking phải được chọn cùng cách đo. Nếu chỉ nhìn số document đúng, Sentence có vẻ tốt; khi đo độ đầy đủ evidence, Recursive và Heading 1000/1400 tốt hơn. Với chính sách nhiều bước, section/heading là tín hiệu hữu ích hơn cắt câu đơn thuần.

### Nếu làm lại

Nhóm sẽ bổ sung một tập query hold-out chưa dùng khi tuning, xác minh lại nội dung/gold answer trực tiếp từ nguồn công khai tại thời điểm nộp, và chạy thêm semantic embedding đa ngữ để so sánh với TF-IDF. Khi đó cần báo cáo riêng latency/cost của API và không trộn số liệu backend với số liệu chunking.

### Demo

- Mở `http://127.0.0.1:8766/src/demo.html` hoặc chạy `run_demo.cmd`.
- Tab **Compare Chunking** hiển thị metric definitions, bars Evidence@3/Top-1, bảng sort và drill-down Q1–Q5.
- Tab **Metadata Filter A/B** minh họa candidate space trước/sau filter.
- Dashboard báo backend đang đo (`lexical TF-IDF`) và chỉ hiển thị trạng thái model/API, không hiển thị secret.

---

## Tự Đánh Giá (Phần Nhóm)

| Tiêu chí | Điểm tự đánh giá |
|---|---:|
| Lựa chọn tài liệu | 10 / 10 |
| Thiết kế chiến lược | 15 / 15 |
| Chất lượng truy xuất | 10 / 10 |
| Thuyết trình/Demo | 5 / 5 |
| **Tổng phần nhóm** | **40 / 40** |
