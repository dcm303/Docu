# Quy trình Triển khai và Vận hành API cho Mô hình LSTM

Tài liệu này hướng dẫn cách nạp các thành phần cần thiết và thiết lập logic xử lý để đưa mô hình dự báo chuỗi thời gian vào môi trường thực tế.

---

## 1. Các File cần nạp (The Assets)

Để khởi tạo API, bạn sẽ cần nạp 4 file sau vào bộ nhớ để đảm bảo tính nhất quán giữa huấn luyện và dự báo:

| Tên File | Vai trò |
| :--- | :--- |
| **model_metadata.json** | Chứa thông số cấu hình (`hidden_size`, `num_layers`, `input_size`) để khởi tạo đúng đối tượng mô hình. |
| **lstm_5m_best_model.pth** | Chứa các "tri thức" (trọng số - weights) đã được tối ưu hóa sau quá trình huấn luyện. |
| **scaler_features_5m.pkl** | Dùng để chuẩn hóa dữ liệu đầu vào từ API (ví dụ: 1000 requests) về khoảng $0 \dots 1$ trước khi đưa vào mô hình. |
| **scaler_target_5m.pkl** | Dùng để giải chuẩn hóa kết quả dự báo từ $0 \dots 1$ quay lại con số requests thực tế mà hệ thống Autoscaler hiểu được. |

---

## 2. Các File Code cần sử dụng (The Logic)

Theo cấu trúc module hóa, đây là những đoạn mã sẽ trực tiếp tham gia vào quá trình xử lý request:

* **`src/lstm/models/lstm_model.py`**: Chứa định nghĩa lớp `LSTMModel`. Đây là bản thiết kế bắt buộc phải import để tạo instance mô hình trước khi nạp trọng số.
* **`src/lstm/models/model_utils.py`**: Cung cấp các hàm tiện ích giúp nạp nhanh các file `.pth` và `.pkl` một cách an toàn.
* **`src/lstm/inference/predictor.py`**: **"Trái tim" của API.** Lớp `Predictor` trong này đóng vai trò điều phối:
    1. Nhận dữ liệu thô.
    2. Kiểm tra tính hợp lệ (`validate_input`).
    3. Chuẩn hóa bằng `scaler_features`.
    4. Thực hiện dự báo qua mô hình.
    5. Giải chuẩn hóa kết quả bằng `scaler_target`.
    6. Trả về kết quả cuối cùng cho hệ thống.

---

## 3. Quy trình khởi tạo trong API (Workflow)

Khi server khởi động (ví dụ: sử dụng `uvicorn main:app`), quy trình diễn ra theo các bước sau:

1.  **Load Metadata:** Đọc file JSON để lấy thông số cấu hình mạng.
2.  **Khởi tạo Model:** Tạo đối tượng từ class `LSTMModel` với các thông số vừa đọc.
3.  **Load Trọng số:** Nạp file `.pth` vào đối tượng mô hình thông qua `load_state_dict()` và chuyển sang chế độ `model.eval()`.
4.  **Load Scalers:** Nạp 2 file bộ lọc `.pkl` vào biến để sẵn sàng xử lý dữ liệu.
5.  **Sẵn sàng:** API chuyển sang trạng thái chờ request. Khi có dữ liệu mới, nó chỉ việc gọi hàm `predict()` từ module `predictor`.

---

## 💡 Lưu ý về "Inference Buffer"

Trong module `predictor.py`, việc tích hợp **Buffer** là một giải pháp xử lý cực kỳ thông minh:

* **Thách thức:** LSTM cần một chuỗi gồm 12 bước thời gian liên tiếp để dự báo. Tuy nhiên, trong thực tế, API thường chỉ nhận được **1 điểm dữ liệu** mới nhất tại mỗi thời điểm.
* **Giải pháp:** Buffer đóng vai trò là "bộ nhớ tạm", lưu giữ 11 điểm dữ liệu trước đó. Khi có điểm thứ 12 gửi đến, hệ thống sẽ kết hợp lại thành một **cửa sổ trượt (Sliding Window)** hoàn chỉnh để mô hình thực hiện dự báo ngay lập tức mà không cần truy vấn lại toàn bộ cơ sở dữ liệu.