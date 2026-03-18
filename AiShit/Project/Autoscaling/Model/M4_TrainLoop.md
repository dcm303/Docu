# Quy trình Huấn luyện và Tối ưu hóa Mô hình

Tài liệu này hệ thống lại các thành phần cốt lõi trong quá trình huấn luyện mạng LSTM, từ việc chọn hàm mục tiêu đến cơ chế dừng sớm để ngăn chặn quá khớp.

---

## 1. "Vị giám khảo" - Hàm Loss MSE (`nn.MSELoss`)
Trong mã nguồn, hàm mất mát được khai báo là `criterion = nn.MSELoss()`. 

* **Cơ chế:** MSE (Mean Squared Error) tính bình phương khoảng cách giữa giá trị dự báo ($\hat{y}$) và thực tế ($y$):
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
* **Tại sao dùng cho Autoscaling?** Đây là bài toán hồi quy (Regression). MSE cực kỳ nhạy cảm với các sai số lớn do phép toán bình phương. Điều này buộc mô hình phải cố gắng thu hẹp các khoảng cách lớn, giúp dự báo tải không bị lệch quá xa so với thực tế, đảm bảo an toàn cho hệ thống server.

---

## 2. "Người dẫn đường" - Thuật toán Adam (`torch.optim.Adam`)
Sử dụng `optimizer = torch.optim.Adam(model.parameters(), lr=lr)` mang lại sự ổn định cho quá trình hội tụ.

* **Đặc điểm nổi bật:** Adam có **tốc độ học thích ứng (Adaptive Learning Rate)**. 
* **Cơ chế:** Thuật toán tự động điều chỉnh bước đi: đi nhanh ở những vùng bằng phẳng và đi chậm lại khi tiến gần đến điểm tối ưu (điểm có Loss thấp nhất). Điều này giúp mô hình hội tụ nhanh hơn và ổn định hơn so với thuật toán SGD truyền thống.

---

## 3. Chu kỳ 4 bước trong mỗi Batch
Trong mỗi lượt huấn luyện (Batch), mô hình thực hiện vòng lặp 4 bước "vàng":

1.  **Forward (Lan truyền tiến):** `outputs = model(X_batch)` — Đưa chuỗi dữ liệu quá khứ vào để mô hình đưa ra dự báo.
2.  **Calculate Loss (Tính sai số):** `loss = criterion(outputs.squeeze(), y_batch)` — So sánh dự báo với thực tế để định lượng mức độ "sai".
3.  **Backward (Lan truyền ngược):** `loss.backward()` — Sử dụng đạo hàm để tính toán xem mỗi trọng số trong mạng cần thay đổi bao nhiêu để giảm bớt sai số.
4.  **Optimizer Step (Cập nhật):** `optimizer.step()` — Adam thực hiện điều chỉnh các trọng số dựa trên kết quả tính toán ở bước 3.

---

## 4. Early Stopping - "Chốt chặn" thông minh
Sử dụng Early Stopping kết hợp với tham số `patience` là một kỹ thuật quản trị huấn luyện chuyên nghiệp.

* **Cách hoạt động:** Sau mỗi Epoch, mô hình được đánh giá trên tập **Validation**. 
* **Mục đích:** Nếu trong khoảng `patience` (ví dụ: 10) Epoch liên tiếp mà sai số trên tập Validation không giảm, chương trình sẽ tự động dừng huấn luyện.
* **Ý nghĩa:** Ngăn chặn hiện tượng **Overfitting** (Học vẹt). Nó đảm bảo mô hình không chỉ "học thuộc lòng" dữ liệu cũ mà còn có khả năng dự báo tốt cho dữ liệu mới chưa từng thấy.

---

### 💡 Giải đáp về `optimizer.zero_grad()`
Lý do chúng ta phải "xóa sạch đạo hàm" về 0 trước khi thực hiện bước Backward là vì:
Trong PyTorch, các đạo hàm được **cộng dồn (accumulate)** qua mỗi lần gọi `.backward()`. Nếu không xóa về 0, đạo hàm của batch mới sẽ bị cộng thêm vào đạo hàm của batch cũ. Điều này sẽ khiến hướng cập nhật trọng số bị sai lệch hoàn toàn, làm mô hình không thể hội tụ.



# Chi tiết Tham số và Quy trình Huấn luyện Mô hình Deep Learning

Tài liệu này giải thích ý nghĩa của các tham số then chốt trong quá trình huấn luyện mạng LSTM và cơ chế dừng sớm để tối ưu hóa hiệu suất.

---

## 1. Patience là gì? (Tham số quan trọng nhất cho Early Stopping)

Trong tiếng Anh, **Patience** nghĩa là "sự kiên nhẫn".

* **Định nghĩa:** Trong huấn luyện Deep Learning, patience là số lượng Epoch (vòng lặp) mà bạn cho phép mô hình tiếp tục chạy mặc dù kết quả trên tập kiểm định (**Validation Loss**) không có sự cải thiện.
* **Cơ chế vận hành:**
    * Nếu bạn đặt `patience=10`, hệ thống sẽ liên tục theo dõi sai số trên tập Validation.
    * Giả sử tại Epoch 20 bạn đạt kết quả tốt nhất (Loss thấp nhất). Nếu từ Epoch 21 đến 30 sai số không giảm xuống thấp hơn mức ở Epoch 20, hệ thống sẽ tự động dừng lại.
* **Ý nghĩa:** * **Tiết kiệm thời gian:** Không tốn tài nguyên cho những vòng lặp không mang lại giá trị.
    * **Ngăn chặn Overfitting:** Đây là vai trò quan trọng nhất. Nếu không có patience, mô hình sẽ tiếp tục học và bắt đầu "học thuộc lòng" cả những nhiễu của tập Train, dẫn đến việc dự báo trên dữ liệu thực tế (Test) bị sai lệch hoàn toàn.

---

## 2. Ý nghĩa của các tham số khác trong hàm Train

### Epochs (Số vòng lặp tối đa)
* **Mục đích:** Quy định số lần tối đa mô hình được nhìn thấy toàn bộ dữ liệu huấn luyện.
* **Tại sao cần?** Nếu đặt quá nhỏ, mô hình chưa kịp học xong quy luật (**Underfitting**). Nếu đặt quá lớn, bạn đã có `patience` bảo vệ để dừng đúng lúc. Con số `50` thường là một ngưỡng an toàn để mô hình đủ thời gian hội tụ.

### Learning Rate (lr - Tốc độ học)
* **Mục đích:** Quyết định độ lớn của "bước đi" khi thuật toán tối ưu cập nhật trọng số. Trong dự án này, chúng ta sử dụng `lr=0.001` cho thuật toán Adam.
* **Hệ quả:**
    * **Nếu lr quá lớn:** Mô hình sẽ "nhảy" lung tung và không bao giờ tìm được điểm tối ưu (giống như người bước quá dài sẽ nhảy vọt qua vạch đích).
    * **Nếu lr quá nhỏ:** Mô hình học cực kỳ chậm và dễ bị kẹt ở những vùng kiến thức hạn hẹp (Local Minima).

### Batch Size (Kích thước lô)
* **Mục đích:** Số lượng mẫu dữ liệu (cửa sổ 12 bước thời gian) mà mô hình xem xét trước khi thực hiện cập nhật trọng số một lần.
* **Tại sao dùng 32?** Đây là con số tiêu chuẩn giúp cân bằng giữa tốc độ tính toán của phần cứng (CPU/GPU) và độ ổn định của việc cập nhật trọng số.

### Model Save Path
* **Mục đích:** Đường dẫn để lưu lại bộ trọng số (Weights) tốt nhất mà mô hình tìm thấy được trong suốt quá trình huấn luyện.
* **Tại sao cần?** Nhờ tham số này, ngay cả khi mô hình học đến Epoch 50 nhưng kết quả tốt nhất nằm ở Epoch 25, bạn vẫn có thể lấy lại phiên bản hoàn hảo nhất đó để thực hiện dự báo thực tế.

---

## 💡 Tóm tắt logic phỏng vấn (Interview Cheat Sheet)

> "Trong hàm huấn luyện, em đã tích hợp cơ chế **Early Stopping** với tham số **patience** để đảm bảo tính tổng quát hóa cho mô hình. Em sử dụng **Learning Rate 0.001** kết hợp với **Adam Optimizer** để tối ưu hóa tốc độ hội tụ và tránh hiện tượng triệt tiêu đạo hàm vốn thường gặp ở mạng RNN truyền thống."