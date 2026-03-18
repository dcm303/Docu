# Kiến trúc Encoder-Decoder và Cơ chế Vận hành trong Transformer

Tài liệu này hệ thống hóa cấu trúc của Transformer, quy trình tạo câu tự hồi quy và ứng dụng thực tế trong hệ thống RAG (Retrieval-Augmented Generation).

---

## 🏗️ 1. Khối Encoder: Chuyên gia Đọc hiểu
Nhiệm vụ của Encoder là chuyển hóa câu văn thô thành một ma trận ngữ cảnh "thông minh" (**Contextual Embeddings**).

* **Lớp cửa ngõ:** Token được Embed thành vector $d$ chiều, sau đó cộng với **Positional Encoding** (sử dụng hàm Sin/Cos) để giữ thông tin về thứ tự từ mà không làm tăng kích thước dữ liệu.
* **Multi-Head Attention:** Mỗi từ tự nhìn vào tất cả các từ khác trong câu để hiểu mối quan hệ (ví dụ: từ "đắt" đang bổ nghĩa cho "váy").
* **Add & Norm:** * **Add (Residual):** Cộng kết quả Attention với đầu vào để tránh mất thông tin gốc và lỗi triệt tiêu đạo hàm.
    * **Norm (Layer Norm):** Nắn các con số về vùng an toàn để mạng học ổn định.
* **Feed Forward Layer (FFL):** Xử lý sâu cho từng token (phóng đại chiều $\rightarrow$ hàm kích hoạt $\rightarrow$ nén lại) để chốt lại đặc trưng cuối cùng.



---

## ✍️ 2. Khối Decoder: Nhà văn Tự hồi quy
Nhiệm vụ của Decoder là tạo ra câu trả lời từng chữ một dựa trên gợi ý từ Encoder.

* **Masked Self-Attention:** Đây là bước "khóa miệng" Decoder.
    * **Tương lai:** Là những từ trong câu trả lời mà máy chưa viết tới.
    * **Cơ chế:** Dùng ma trận mặt nạ (Mask) để gán giá trị $-\infty$ vào các từ phía sau trong ma trận $QK^T$, khiến xác suất chú ý vào chúng bằng $0$. Điều này buộc máy phải học cách suy luận từ những từ đã có.
* **Encoder-Decoder Attention (Cây cầu):** * **Query ($Q$):** Lấy từ những gì Decoder vừa viết.
    * **Key ($K$) & Value ($V$):** Lấy từ ma trận ngữ cảnh của Encoder.
    * **Mục đích:** Giúp câu trả lời bám sát nội dung câu hỏi gốc (ví dụ: dịch đúng nghĩa từ "nó" là "váy").
* **FFL & Đầu ra:** Sau khi "suy nghĩ" qua FFL, dữ liệu đi qua lớp Linear và Softmax để chọn ra một từ thực tế từ hàng nghìn từ trong từ điển.



---

## 🔄 3. Quy trình tạo câu (Inference Loop)
Decoder không hoạt động song song như Encoder mà hoạt động theo vòng lặp (**Autoregressive**):

1. **Cú hích đầu tiên:** Bắt đầu bằng token mồi `<SOS>`.
2. **Vòng lặp:** * Lấy `<SOS>` + (các từ đã tạo) làm đầu vào.
    * Soi vào Encoder để lấy ngữ cảnh.
    * Tạo ra từ mới.
    * Nạp từ mới đó ngược lại vào đầu vào cho bước tiếp theo.
3. **Kết thúc:** Dừng lại khi mô hình tự tạo ra token `<EOS>`.

---

## 👗 4. Liên hệ thực tế: Dự án Fashion Chatbot (RAG)
Khi áp dụng toàn bộ đống lý thuyết này vào hệ thống RAG:

* **Encoder:** Sử dụng để "hiểu" câu hỏi khách hàng và tạo ra các Vector sản phẩm chuẩn xác để lưu vào Vector Database.
* **Decoder:** Khi đã tìm được sản phẩm phù hợp (ví dụ: "Váy lụa 500k"), thông tin này được đưa vào Encoder. Decoder sẽ dùng cơ chế Attention để viết thành câu tư vấn tự nhiên: *"Dạ, mẫu váy lụa bạn thích đang có giá là 500k ạ!"*