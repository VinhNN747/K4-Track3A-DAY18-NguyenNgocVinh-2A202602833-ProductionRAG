# Individual Reflection — Lab 18: Production RAG

**Học viên:** Nguyễn Ngọc Vinh  
**Khóa:** K4 — Track 3A  
**Ngày hoàn thành:** 04/10/2026

## Phần 1: Mapping bài giảng vào code

| Lecture concept | Module | Hàm cụ thể | Quan sát |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Nhóm các câu liên tiếp bằng cosine similarity; khi model chưa sẵn có, lexical fallback giữ pipeline chạy nhưng chất lượng thấp hơn semantic embedding. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | 100 child chunks được tạo từ 26 tài liệu. Child phục vụ retrieval, còn `parent_id` bảo toàn đường nối để mở rộng context ở bước answer. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | BM25 xử lý tốt từ khóa/chữ số chính xác; dense bù cho khác biệt diễn đạt. RRF không trộn trực tiếp các thang điểm khác nhau. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Rerank top candidate xuống top-3. Bản offline dùng lexical fallback, cho thấy lý do cần BGE reranker thật đối với câu multi-hop và ngưỡng số tiền. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Pipeline xuất đủ Faithfulness, Answer Relevancy, Context Precision và Context Recall. Offline diagnostics ghi nhận relevancy 0.6997 và recall 0.7655 là hai hướng cần ưu tiên. |
| Contextual embeddings | M5 | `_enrich_single_call()` | Một call trả summary, hypothetical questions, context và metadata; fallback tạo các trường tương tự để indexing không bị gián đoạn khi không có API key. |

## Phần 2: Khó khăn và cách giải quyết

- **Lỗi gặp phải:** `qdrant_client.http.exceptions.ResponseHandlingException: timed out` khi Qdrant local không phản hồi trong lúc tạo collection.
  - **Nguyên nhân:** Client có thể khởi tạo nhưng service Docker không đủ sẵn sàng cho thao tác ghi.
  - **Cách xử lý:** `DenseSearch` hỗ trợ in-process cosine search khi bật `RAG_OFFLINE=1`; chế độ mặc định vẫn dùng Qdrant khi service khả dụng.

- **Lỗi gặp phải:** `Connection error.` do key mẫu trong `.env` không dùng được để gọi LLM.
  - **Cách xử lý:** M5, M4 và pipeline có offline switch. Enrichment dùng extractive/contextual fallback; evaluation dùng lexical diagnostic có ghi rõ giới hạn trong report.

- **Kiến thức cần bổ sung:** Version-aware retrieval và multi-hop QA. Câu phép năm có cả v2023/v2024, còn câu lương thử việc cần ghép hai tài liệu. Retrieval theo similarity đơn thuần chưa biểu diễn được hiệu lực chính sách hoặc coverage của các sub-question.

## Phần 3: Action plan cho project cá nhân

### Project: Trợ lý hỏi đáp chính sách nội bộ

#### Hiện trạng

- **Pipeline hiện tại:** Nạp Markdown/PDF text layer, chunk theo parent/child, hybrid search, rerank và sinh câu trả lời từ context.
- **Known issues:** PDF scan chưa OCR; version cũ có thể cạnh tranh với version đang hiệu lực; câu đa nguồn cần coverage nhiều section.

#### Kế hoạch áp dụng

1. [ ] **Chunking:** Dùng hierarchical chunking cho retrieval và structure-aware chunking cho bảng/quy trình; lưu `parent_id`, section và source.
2. [ ] **Search:** Dùng BM25 + BGE-M3 + RRF; thêm filter/boost theo `status=active`, version và effective date.
3. [ ] **Reranking:** Chạy `BAAI/bge-reranker-v2-m3` trên top-20 để chọn top-3, sau đó diversify theo source cho câu multi-hop.
4. [ ] **Evaluation:** Dùng RAGAS 4 metrics với test set có lookup, negation, version, numeric và multi-hop; đọc bottom-N sau mỗi thay đổi.
5. [ ] **Enrichment:** Dùng contextual prepend + metadata extraction; HyQA cho dữ liệu có vocabulary khác với cách người dùng đặt câu.

#### Timeline

- **Tuần 1:** Chuẩn hóa metadata, OCR các PDF scan và benchmark BM25/dense/hybrid trên test set.
- **Tuần 2:** Thêm reranking, version filter, prompt trả lời có citation; chạy RAGAS judge-based và xử lý năm failure tệ nhất.
