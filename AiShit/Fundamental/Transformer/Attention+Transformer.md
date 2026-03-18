# Cơ chế Attention trong Kiến trúc Transformer

Tài liệu này chi tiết hóa luồng toán học của Self-Attention và sức mạnh của Multi-Head Attention trong việc nắm bắt ngữ cảnh dữ liệu.

---

## 🧠 1. Luồng toán học của Self-Attention

Dữ liệu đầu vào là ma trận $X$ kích thước $(n, d)$, trong đó $n$ là số token và $d$ là số chiều embedding.

### Bước 1: Trích xuất Q, K, V
Thay vì nhân với các ma trận có kích thước $(n, d)$, chúng ta nhân $X$ với các ma trận trọng số học được $W^Q, W^K, W^V$ có kích thước $(d, d_k)$:

* **Query ($Q$):** $Q = X \cdot W^Q$ (Kích thước $n \times d_k$)
* **Key ($K$):** $K = X \cdot W^K$ (Kích thước $n \times d_k$)
* **Value ($V$):** $V = X \cdot W^V$ (Kích thước $n \times d_v$)

### Bước 2: Phép toán "Vàng"
Để tính sự tương quan giữa mọi cặp từ, Transformer sử dụng tích vô hướng và hệ số tỉ lệ $\sqrt{d_k}$ để tránh bão hòa hàm Softmax:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Phân tích thành phần:**
1.  **$QK^T$ ($n \times n$):** Ma trận điểm số (Scores). Ô $(i, j)$ cho biết mức độ "chú ý" của từ thứ $i$ vào từ thứ $j$.
2.  **Softmax:** Chuyển đổi điểm số thành xác suất (trọng số), đảm bảo tổng các trọng số bằng 1.
3.  **Nhân với $V$ ($n \times d$):** Tổng hợp thông tin từ tất cả các từ khác trong chuỗi dựa trên trọng số đã tính để tạo ra vector đại diện mới cho mỗi từ.

---

## 🚀 2. Multi-Head Attention: Nhiều "góc nhìn" cùng lúc

Thay vì thực hiện một phép Attention duy nhất, chúng ta chia vector $d$ chiều thành $h$ phần nhỏ (Heads) để học các mối quan hệ khác nhau đồng thời.

### Tại sao lại cần Multi-Head?
Nếu chỉ có 1 Head, mô hình chỉ có thể tập trung vào một kiểu quan hệ duy nhất. Với Multi-Head (ví dụ 8 heads), mỗi đầu sẽ phụ trách một khía cạnh:
* **Head 1:** Chú ý vào quan hệ ngữ pháp (chủ ngữ - động từ).
* **Head 2:** Chú ý vào quan hệ thực thể (tên riêng - vị trí).
* **Head 3:** Chú ý vào đặc điểm sản phẩm (màu sắc, chất liệu - rất quan trọng trong dự án Fashion Chatbot).

### Quy trình thực hiện:
1.  **Chia nhỏ:** Chia vector embedding thành $h$ phần.
2.  **Tính toán song song:** Thực hiện Self-Attention độc lập trên mỗi phần.
3.  **Ghép nối (Concatenate):** Nối kết quả từ tất cả các Heads lại thành một ma trận duy nhất kích thước $(n, d)$.
4.  **Phóng chiếu tuyến tính (Linear Projection):** Nhân với ma trận trọng số cuối cùng ($W^O$) để hòa trộn thông tin từ các "góc nhìn" khác nhau thành một kết quả thống nhất.




# Chi tiết cơ chế Self-Attention và Multi-Head Attention

Tài liệu này hệ thống hóa quy trình tính toán bên trong lớp Attention - thành phần cốt lõi của kiến trúc Transformer, giúp mô hình nắm bắt ngữ cảnh phức tạp trong các dự án như Fashion Chatbot.

---

## 🏗️ 1. Khởi tạo: Ma trận đầu vào $X$

Quá trình bắt đầu với một câu văn gồm $n$ tokens. Mỗi token được chuyển thành một vector ngữ nghĩa (Semantic Embedding) có số chiều là $d$.

* **Ma trận $X$ ($n \times d$):** Chứa toàn bộ thông tin của câu. 
    * Mỗi hàng đại diện cho một token.
    * Mỗi cột đại diện cho một đặc trưng ngữ nghĩa.

---

## 🗝️ 2. Trích xuất Query, Key, Value ($Q, K, V$)

Thay vì dùng trực tiếp $X$, mô hình học 3 ma trận trọng số $W^Q, W^K, W^V$ (thường có kích thước $d \times d_k$) để chiếu $X$ vào các không gian khác nhau nhằm phục vụ các mục đích riêng biệt:

1.  **Query ($Q = X \cdot W^Q$):** Đóng vai trò là **"câu hỏi"**. Mỗi token tự hỏi: *"Tôi nên tìm kiếm thông tin gì từ các từ khác?"*
2.  **Key ($K = X \cdot W^K$):** Đóng vai trò là **"nhãn dán"**. Mỗi token tự giới thiệu: *"Tôi có thông tin loại này đây."*
3.  **Value ($V = X \cdot W^V$):** Đóng vai trò là **"nội dung"**. Chứa thông tin thực sự của token đó nếu nó được lựa chọn để "chú ý" tới.

---

## 📈 3. Cơ chế tính toán sự tương quan (Attention Score)

Đây là nơi "ma thuật" xảy ra giúp mô hình hiểu được ngữ cảnh thông qua phép toán:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Quy trình diễn ra như sau:**

* **Phép nhân $QK^T$ ($n \times n$):** Tạo ra ma trận điểm số tương quan. Ô tại hàng $i$ cột $j$ cho biết mức độ liên quan/quan trọng giữa từ thứ $i$ và từ thứ $j$.
* **Hệ số tỉ lệ $\sqrt{d_k}$:** Chia cho căn bậc hai số chiều để tránh việc giá trị quá lớn sau khi nhân, giúp hàm Softmax không bị bão hòa (tránh triệt tiêu đạo hàm).
* **Hàm Softmax:** Chuyển điểm số thành xác suất (trọng số) trong khoảng $[0, 1]$. Nó giúp mô hình tập trung tối đa vào các mối quan hệ quan trọng nhất (ví dụ: từ "nó" sẽ tập trung vào "chiếc váy").
* **Nhân với $V$ ($n \times d$):** Mỗi token lúc này sẽ được cập nhật thành một vector mới, là tổng hòa thông tin của chính nó và các token liên quan khác trong câu dựa trên trọng số đã tính.

---

## 🚀 4. Multi-Head Attention: Đa góc nhìn

Thay vì chỉ dùng một "đầu" Attention duy nhất, chúng ta chia nhỏ số chiều $d$ thành nhiều phần (ví dụ 8 heads) để học song song.

* **Cơ chế:** Mỗi Head sẽ học các bộ ma trận $W^Q, W^K, W^V$ riêng biệt.
    * **Head 1:** Có thể học về quan hệ ngữ pháp (chủ ngữ - vị từ).
    * **Head 2:** Có thể học về đặc điểm sản phẩm (màu sắc - chất liệu).
* **Ghép nối (Concatenation):** Sau khi tính xong kết quả của 8 heads, ta nối chúng lại và nhân với một ma trận trọng số cuối cùng $W^O$ để đưa về đúng kích thước $(n, d)$ ban đầu.

**Ý nghĩa:** Multi-Head Attention cho phép mô hình hiểu được câu văn ở nhiều tầng ý nghĩa khác nhau cùng một lúc. Điều này cực kỳ quan trọng cho việc hiểu các yêu cầu phức tạp của khách hàng, ví dụ như phân biệt giữa "chất liệu vải" và "kiểu dáng thiết kế" trong dự án **Fashion Chatbot**.