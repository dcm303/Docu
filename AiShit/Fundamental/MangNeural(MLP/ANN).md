Là một mạng bao gồm input layer sẽ là lớp nhận dữ liệu thô -> hidden layer là nơi diễn ra các phép toán biến đổi input để đưa ra output layer là nơi đưa ra dự đoán cuối cùng. Phép toán diễn ra tại mỗi neural sẽ là kết quả của hàm kích hoạt dựa trên đầu ra của Wi.xi + B trong đó W sẽ quyết định độ quan trọng của từng input và Bias sẽ quyết định ngưỡng ưu tiên để xem điểm số Wx phải cao đến đâu thì sẽ kích hoạt. Hàm kích hoạt này đóng vai trò tạo nếp gấp, đảm bảo sự phi tuyến tính giúp mạng hiểu được các khái niệm phức tạp. Đây là quá trình Forward Propagation và sau khi đã có y, chúng ta sẽ tính hàm loss và thực hiện Backpropagation dựa trên việc tính đạo hàm lần lượt từ lớp sát cuối dần lên kết hợp cùng gradient descent là công thức để giúp cập nhật trọng số (cùng learning rate đóng vai trò như việc học nhanh hay chậm để điều chỉnh trọng số w) để điểu chỉnh các trọng số để giảm lỗi. 2 tham số còn lại đáng nhắc đến sẽ là batch size là số dữ liệu mà mạng có thể sử dụng ở mỗi lần cập nhật trọng số -> batch size nhỏ sẽ giúp cập nhật liên tục, thoát khỏi local minima nhưng sẽ nhiều nhiễu, và batch size lớn thì sẽ tính toán nặng hơn nhưng ổn định hơn và epoch là một lần mà mạng đã đi qua toàn bộ tập dữ liệu (theo từng batch size) -> nhiều epoch sẽ giúp mạng học được nhiều quy luật phức tạp nhưng có thể dẫn đến overfitting

# Hệ thống hóa kiến thức Mạng Neural (Neural Networks)

Tài liệu này tổng hợp cấu trúc, cơ chế vận hành và các chiến lược huấn luyện quan trọng trong Deep Learning.

---

## 1. Hệ thống hóa lại "Bộ não" Neural Network
Mạng Neural là một chuỗi các lớp xử lý thông tin nối tiếp nhau:

* **Input Layer:** Nhận dữ liệu thô (như pixel ảnh xe đua hoặc log từ NASA).
* **Hidden Layers:** Nơi "nếp gấp" (phi tuyến tính) xuất hiện nhờ hàm kích hoạt (**Activation**) để mô hình học được các quy luật phức tạp và trừu tượng.
* **Output Layer:** Đưa ra kết quả cuối cùng (như xác suất rẽ trái/phải hoặc dự báo tải lượng server).

---

## 2. Sự phối hợp giữa $W$, $b$ và Activation
Đây là "linh hồn" của quá trình xử lý tín hiệu trong từng neuron:

* **Trọng số ($W$):** Quyết định *"Tín hiệu nào đáng nghe?"*. Nó thể hiện tầm quan trọng của từng đầu vào.
* **Độ chệch ($b$):** Quyết định *"Ngưỡng nào thì bắt đầu phản ứng?"*. Bias giúp dịch chuyển hàm kích hoạt để khớp chính xác hơn với phân phối dữ liệu thực tế.
* **Hàm kích hoạt (Activation Function):** Đưa vào tính **Phi tuyến (Non-linearity)**. Nếu không có nó, mạng Neural dù sâu đến đâu cũng chỉ tương đương với một phép tính tuyến tính đơn giản (Hồi quy tuyến tính).

---

## 3. Chu kỳ Học: Forward & Backward

Quá trình học là một vòng lặp liên tục giữa việc "phán đoán" và "sửa sai":

* **Forward Propagation:** Là quá trình "đưa ra nhận định" bằng cách truyền dữ liệu từ lớp đầu vào đến lớp đầu ra.
* **Loss Function:** Là "thước đo lỗi sai". Ví dụ: Nếu dự đoán tải server là 100 nhưng thực tế là 500, giá trị hàm Loss sẽ rất lớn.
* **Backpropagation (Lan truyền ngược):** Là quá trình "truy cứu trách nhiệm". Thuật toán sử dụng đạo hàm (**Chain Rule**) để tính toán xem mỗi lớp và mỗi tham số đã đóng góp bao nhiêu phần vào lỗi sai tổng thể.

### Gradient Descent & Learning Rate
* **Gradient Descent:** Là thuật toán tối ưu hóa giúp ta tìm ra điểm mà lỗi (Loss) thấp nhất.
* **Learning Rate ($\eta$):** Được ví như "độ dài bước chân" của mô hình:
    * Bước quá dài: Dễ nhảy vọt qua đích (diverge).
    * Bước quá ngắn: Đi mãi không tới đích hoặc tốn quá nhiều thời gian (slow convergence).

---

## 4. Chiến lược huấn luyện: Batch & Epoch

* **Batch Size:** Sự đánh đổi giữa **độ ổn định** và **tính ngẫu nhiên**. Việc sử dụng Batch nhỏ tạo ra "nhiễu" có ích, giúp mô hình thoát khỏi các điểm cực tiểu cục bộ (local minima).
* **Epoch:** Sự đánh đổi giữa **độ hiểu biết** và **tính học vẹt**. 
    * Quá ít Epoch: Mô hình chưa học hết quy luật (Underfitting).
    * Quá nhiều Epoch: Mô hình bắt đầu học thuộc lòng cả nhiễu của dữ liệu (Overfitting).