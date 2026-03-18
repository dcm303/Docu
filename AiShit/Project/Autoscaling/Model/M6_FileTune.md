# Quy trình Dò tìm Siêu tham số (Hyperparameter Tuning)

Tài liệu này mô tả phương pháp thực nghiệm để tìm ra bộ cấu hình tối ưu cho mô hình LSTM, đảm bảo sự cân bằng giữa độ chính xác và hiệu suất tính toán.

---

## 1. Siêu tham số (Hyperparameters) là gì?

Khác với "trọng số" (weights) mà mô hình tự học được trong quá trình huấn luyện, siêu tham số là những con số do chính bạn quyết định trước khi bắt đầu. Chúng quyết định cấu trúc và cách thức học tập của mô hình:

* **Cấu trúc mạng:** `hidden_size` (số đơn vị nhớ), `num_layers` (số lớp LSTM chồng lên nhau), `dropout`.
* **Quá trình huấn luyện:** `learning_rate` (tốc độ học), `batch_size`, `patience`.
* **Dữ liệu:** `sequence_length` (độ dài cửa sổ quá khứ).

---

## 2. Chiến lược "Grid Search" tự động

Thay vì thay đổi thủ công, chúng ta sử dụng hàm `run_hyperparameter_tuning` để duyệt qua danh sách các tổ hợp cấu hình (tuning grid) từ file **YAML**:

1.  **Tự động hóa:** Với mỗi tổ hợp, hệ thống tự động: Chuẩn bị dữ liệu $\rightarrow$ Khởi tạo model $\rightarrow$ Huấn luyện $\rightarrow$ Đánh giá.
2.  **Tính khách quan:** Tất cả các biến thể mô hình đều được đánh giá trên cùng một tập Test để đảm bảo sự công bằng tuyệt đối.

---

## 3. Cách chọn ra "Nhà vô địch"

Sau khi hoàn tất thử nghiệm, việc so sánh được thực hiện dựa trên các tiêu chí chiến lược:

* **Tiêu chí ưu tiên:** Sắp xếp kết quả theo `test_mape` tăng dần. Mô hình nào có MAPE thấp nhất (sai số phần trăm nhỏ nhất) sẽ được ưu tiên.
* **Sự đánh đổi (Trade-off):** Đôi khi mô hình có MAPE thấp nhất lại quá nặng (nhiều lớp, `hidden_size` lớn). Trong thực tế, chúng ta có thể chọn một mô hình "đủ tốt" (ví dụ MAPE chỉ lệch $0.1\%$ so với bản tốt nhất) nhưng nhẹ hơn để tiết kiệm tài nguyên khi chạy thực tế (Production).

---

## 4. Trực quan hóa kết quả

Sử dụng biểu đồ thanh nằm ngang để so sánh trực quan giữa các cấu hình:
* **MAPE Comparison:** Cho thấy mô hình nào dự báo "sát" nhất theo tỷ lệ phần trăm.
* **RMSE Comparison:** Cho thấy mô hình nào ổn định và ít bị ảnh hưởng bởi các sai số cực lớn (outliers) nhất.

---

## 💡 Điểm nhấn phỏng vấn (Interview Tips)

> "Trong dự án này, em không chỉ dừng lại ở một mô hình duy nhất. Em đã triển khai một pipeline **Hyperparameter Tuning** chuyên nghiệp, sử dụng file cấu hình **YAML** để quản lý các thử nghiệm.
>
> Em đã thử nghiệm các biến thể từ mô hình 'Small' (ít tham số) đến 'Large' (nhiều lớp) để tìm ra điểm cân bằng giữa độ chính xác (Accuracy) và hiệu suất tính toán (Efficiency). Kết quả cuối cùng được lựa chọn dựa trên chỉ số **MAPE**, vì nó phản ánh chính xác nhất mức độ đáp ứng tải của hệ thống Autoscaling trong thực tế."

---

### ❓ Câu hỏi mở rộng:
**Tại sao đôi khi mô hình nhiều lớp (num_layers=2) lại không tốt bằng mô hình 1 lớp?**

**Giải đáp:** Đây là hiện tượng **Overfitting** (Quá khớp). Khi mô hình quá phức tạp so với lượng dữ liệu hiện có, nó bắt đầu "học thuộc lòng" cả những nhiễu (noise) và biến động ngẫu nhiên trong tập Train thay vì học quy luật tổng quát. Khi đem sang tập Test (dữ liệu mới), những "kiến thức học vẹt" này không còn đúng nữa, dẫn đến kết quả tệ hơn một mô hình đơn giản nhưng bền bỉ.