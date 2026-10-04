# Failure Analysis — Lab 18: Production RAG

**Học viên:** Nguyễn Ngọc Vinh
**Khóa:** K4 — Track 3A
**Nguồn kết quả:** `reports/ragas_report.json`, chạy 20 câu trong `test_set.json`.

## Kết quả đánh giá

| Metric | Naive baseline | Production (offline) | Nhận xét |
|---|---:|---:|---|
| Faithfulness | 1.0000 | 1.0000 | Answer fallback lấy trực tiếp từ retrieved context. |
| Answer relevancy | 0.6881 | 0.6997 | Production tăng 0.0116; context vẫn dài, chưa được LLM tổng hợp thành câu trả lời ngắn. |
| Context precision | 1.0000 | 1.0000 | Ba context đầu đều có ít nhất một phần giao với đáp án tham chiếu. |
| Context recall | 0.7733 | 0.7655 | Production giảm 0.0079; câu hỏi đa nguồn và có version vẫn thiếu một phần evidence. |

Lần chạy này đặt `RAG_OFFLINE=1`, vì môi trường không dùng API/model đã tải sẵn. Các metric production là lexical diagnostics dự phòng của M4, không phải điểm judge-based RAGAS. Khi có `OPENAI_API_KEY` hợp lệ, bỏ biến này để chạy bốn metric RAGAS thật.

## Bottom-5 failures

### 1. Laptop 30 triệu: ai phê duyệt và cần gì từ CNTT?

- **Expected:** Director phê duyệt; có xác nhận cấu hình từ CNTT và tối thiểu 3 báo giá.
- **Got:** Chunk từ chính sách hoàn chi đào tạo vì cùng có cụm “30 triệu”.
- **Worst metric:** Answer relevancy (0.2857), score trung bình 0.5521.
- **Error tree:** Output sai → context sai → lexical/hashing retrieval ưu tiên con số hơn loại nghiệp vụ.
- **Root cause:** Chưa có metadata filter theo category/source và query chưa nhấn mạnh `mua_sam` / CNTT.
- **Suggested fix:** Thêm metadata category, query rewrite có entity “laptop/mua sắm”, và rerank bằng BGE cross-encoder thật.

### 2. Mentor và buddy có thể là một người không?

- **Expected:** Không; quản lý trực tiếp cũng không được làm mentor hoặc buddy.
- **Got:** Context không xếp section quy định quan hệ mentor–buddy lên đầu.
- **Worst metric:** Answer relevancy, score trung bình 0.6866.
- **Error tree:** Output thiếu → context chỉ khớp một phần → query có hai điều kiện phủ định.
- **Root cause:** Chunking cắt quy tắc và ngoại lệ thành các child chunk riêng; fallback answer không tổng hợp hai điều kiện.
- **Suggested fix:** Retrieve parent sau child hit, giữ cả section “quy định”, rồi dùng prompt bắt buộc trả lời từng mệnh đề.

### 3. Senior 9 năm: phép năm và khoảng lương

- **Expected:** 18 ngày phép và lương Senior P3–P4 từ 20–35 triệu/tháng.
- **Got:** Chỉ lấy evidence về thâm niên; thiếu bảng lương.
- **Worst metric:** Answer relevancy, score trung bình 0.7554.
- **Error tree:** Output thiếu → một nguồn context đúng → query multi-hop cần hai tài liệu.
- **Root cause:** Top-3 bị chiếm bởi policy nghỉ phép, không bảo đảm coverage theo hai entity “Senior” và “lương”.
- **Suggested fix:** Mở rộng candidate pool, diversify theo source trước rerank, và đánh giá multi-hop theo coverage từng sub-question.

### 4. Mua thiết bị 55 triệu cần ai phê duyệt?

- **Expected:** CEO phê duyệt đơn hàng trên 50 triệu.
- **Got:** Chunk quy trình chung, chưa có hàng approval threshold.
- **Worst metric:** Answer relevancy, score trung bình 0.7596.
- **Error tree:** Output thiếu → context đúng tài liệu nhưng sai section → reranker fallback không phân biệt quan hệ số tiền–thẩm quyền.
- **Root cause:** Child chunk 256 ký tự tách bảng/ngưỡng khỏi câu hỏi và reranker thật chưa được tải.
- **Suggested fix:** Structure-aware chunk bảng và chạy `BAAI/bge-reranker-v2-m3` ở production.

### 5. Lương thử việc Junior cao nhất là bao nhiêu?

- **Expected:** 17 triệu/tháng (`85% × 20 triệu`).
- **Got:** Context chỉ có quy tắc 85%, thiếu mức lương Junior từ bảng lương.
- **Worst metric:** Answer relevancy, score trung bình 0.7775.
- **Error tree:** Output thiếu phép tính → context thiếu dữ liệu thứ hai → câu hỏi numeric multi-hop.
- **Root cause:** Retrieval không nối `thu_viec.md` với `bang_luong_2024.md`; answer fallback không tính toán.
- **Suggested fix:** Query decomposition thành “lương Junior” và “tỷ lệ thử việc”, sau đó tổng hợp có công cụ tính số học.

## Case study

**Câu hỏi:** “Nhân viên được nghỉ bao nhiêu ngày phép năm?”

1. Output chưa đúng vì trả về policy v2023: 12 ngày.
2. Context chứa cả v2023 và v2024, nhưng v2023 đứng đầu.
3. Query không mang version nên BM25/hashing không biết ưu tiên tài liệu đang hiệu lực.
4. Cần đưa `effective_date`, `version`, `status=active/superseded` vào metadata và boost version mới trong reranking.

**Nếu có thêm một giờ:** chạy model embeddings và cross-encoder thật, thêm metadata/version filter, rồi đối chiếu lại bằng RAGAS judge-based thay cho lexical fallback.
