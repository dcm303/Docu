# Kiến trúc Mô hình LSTM (PyTorch Implementation)

Tài liệu này chi tiết hóa cấu trúc và quy trình truyền tin bên trong mạng LSTM được thiết kế để dự báo tải lượng hệ thống.

---

## 1. Khởi tạo kiến trúc (`__init__`)
Trong hàm khởi tạo, các thành phần chính được định nghĩa để xây dựng nên "khung xương" của mô hình:

* **`self.lstm`:** Sử dụng lớp `nn.LSTM` của PyTorch.
    * **input_size:** Số lượng đặc trưng đầu vào (trong dự án này là 5: `requests_target`, `error_rate`, `hour_sin`, `hour_cos`, `is_weekend`).
    * **hidden_size:** Số lượng "đơn vị nhớ" trong mỗi lớp LSTM (ví dụ: 64). Số lượng đơn vị nhớ càng lớn, mô hình càng có khả năng học các quy luật phức tạp nhưng cần cẩn trọng với hiện tượng **Overfitting** (quá khớp).
    * **num_layers:** Số lượng lớp LSTM chồng lên nhau (ví dụ: 2 lớp). Lớp đầu tiên học các đặc trưng thô, các lớp sau học các đặc trưng trừu tượng hơn.
    * **dropout:** Kỹ thuật ngắt kết nối ngẫu nhiên (ví dụ: 0.2 hay 20%) trong quá trình huấn luyện để tăng tính tổng quát hóa và ngăn mô hình "học vẹt".
    * **batch_first=True:** Quy định thứ tự dữ liệu truyền vào là $(Batch, Sequence, Feature)$.
* **`self.fc` (Fully Connected):** Một lớp tuyến tính (`nn.Linear`) để chuyển đổi đầu ra từ không gian `hidden_size` về một con số duy nhất chính là giá trị dự báo số lượng request.

---

## 2. Quy trình truyền tin (`forward`)
Hàm này mô tả cách dữ liệu đi từ cửa sổ 12 bước thời gian đến khi ra kết quả dự báo cuối cùng:

1.  **Khởi tạo trạng thái:** $h_0$ (hidden state) và $c_0$ (cell state) được tạo mới với giá trị bằng $0$ cho mỗi lượt dự báo. Đây đóng vai trò là các "cuốn sổ tay" để LSTM ghi chép thông tin quan trọng qua từng bước thời gian.
2.  **Xử lý chuỗi:** Dữ liệu đi qua các lớp LSTM. Tại mỗi bước trong 12 bước thời gian, các cổng nội bộ sẽ thực hiện nhiệm vụ:
    * **Forget Gate:** Quyết định thông tin nào trong quá khứ nên quên đi.
    * **Input Gate:** Quyết định thông tin mới nào nên lưu lại.
    * **Output Gate:** Quyết định thông tin nào sẽ được dùng để tạo đầu ra.
3.  **Lấy kết quả cuối cùng (`out = out[:, -1, :]`):** Đây là bước quan trọng nhất. Chúng ta chỉ lấy đầu ra của bước thời gian cuối cùng (thứ 12). 
    * *Tại sao?* Vì bước thứ 12 là thời điểm LSTM đã "đọc" xong toàn bộ 1 giờ dữ liệu quá khứ và đã tổng hợp đầy đủ ngữ cảnh nhất để đưa ra dự báo cho 5 phút tới.
4.  **Dự báo:** Đầu ra của bước cuối cùng được đưa qua lớp `self.fc` để ánh xạ về giá trị dự báo cuối cùng.