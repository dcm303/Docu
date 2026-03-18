# Quy trình Đánh giá Mô hình (Model Evaluation)

Tài liệu này chi tiết hóa các bước diễn ra trong quá trình đánh giá sức mạnh dự báo của mô hình trên tập dữ liệu kiểm thử (Test Set).

---

## 1. Chế độ Đánh giá và Tắt Gradient

* **`model.eval()`:** Câu lệnh này thông báo cho mạng Neural rằng: "Chúng ta đang thi thật, không phải đang học". Nó sẽ tắt các lớp **Dropout** để mô hình sử dụng toàn bộ sức mạnh của các trọng số đã được tối ưu hóa.
* **`torch.no_grad()`:** Trong bước này, chúng ta chỉ cần dự báo (Forward pass) chứ không cần tính toán đạo hàm để cập nhật trọng số (Backward pass). Việc này giúp tiết kiệm bộ nhớ và tăng tốc độ tính toán đáng kể.

---

## 2. Giải mã dữ liệu (Inverse Transform)

Đây là bước cực kỳ quan trọng để kết quả có thể sử dụng được trong thực tế.

* **Vấn đề:** Mô hình LSTM trả về kết quả dự báo nằm trong khoảng $[0, 1]$ (do đã qua bước `MinMaxScaler`). Con số như $0.23$ không mang lại ý nghĩa trực quan cho kỹ sư vận hành.
* **Giải pháp:** Sử dụng `scaler_target.inverse_transform` để đưa những con số thập phân này quay trở lại đơn vị thực tế (ví dụ: $1500$ requests). Việc này đảm bảo các chỉ số sai số sau đó có giá trị thực tiễn cao.

---

## 3. Bộ chỉ số đo lường (Metrics)

Sử dụng đa dạng các chỉ số giúp đánh giá mô hình dưới nhiều góc độ khác nhau:

* **MSE (Mean Squared Error):** Trung bình bình phương sai số.
* **RMSE (Root Mean Squared Error):** * Công thức: $RMSE = \sqrt{MSE}$.
    * Ý nghĩa: Đưa sai số về cùng đơn vị với số lượng request. Ví dụ: $RMSE = 50$ nghĩa là trung bình mô hình lệch khoảng $50$ requests mỗi lần dự báo.
* **MAE (Mean Absolute Error):** Trung bình trị tuyệt đối sai số. Cho biết mức độ lệch trung bình mà không bị khuếch đại bởi các sai số cực lớn.
* **MAPE (Mean Absolute Percentage Error):** Giúp kết luận về độ chính xác theo tỷ lệ phần trăm (ví dụ: "Mô hình đạt độ chính xác $95\%$").

---

## 4. Trực quan hóa kết quả (Visualization)

Việc vẽ biểu đồ giúp phát hiện các vấn đề mà con số (metrics) có thể bỏ sót:

* **Zoom-in (300 điểm đầu tiên):** Phóng to một khoảng thời gian ngắn để nhìn rõ đường dự báo (màu cam) có bám sát đường thực tế (màu xanh) tại các đỉnh (peaks) và đáy (troughs) hay không.
* **Toàn bộ tập Test:** Nhìn thấy bức tranh tổng thể, đảm bảo mô hình ổn định và không bị lệch (**bias**) theo thời gian.

---

## 💡 Cách trả lời phỏng vấn chuyên nghiệp

> "Trong bước đánh giá, em đặc biệt chú trọng vào việc sử dụng **Inverse Transform** để đưa các kết quả dự báo về đơn vị requests thực tế trước khi tính toán metrics.
> 
> Em sử dụng **RMSE** để giám sát các sai số lớn — điều cực kỳ quan trọng trong Autoscaling vì nếu dự báo thiếu một lượng lớn requests, hệ thống có thể bị sập. Đồng thời, em dùng **MAPE** để đưa ra cái nhìn trực quan về hiệu suất tổng thể của mô hình cho các bên liên quan."