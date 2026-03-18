# Phân tích Đặc trưng với Random Forest trong Dự báo Chuỗi thời gian

Tài liệu này giải thích cơ chế tìm kiếm mối quan hệ dữ liệu của thuật toán Random Forest và lý do tại sao nó đóng vai trò quan trọng trong việc tối ưu hóa mô hình LSTM.

---

## 1. Cách Random Forest tìm ra mối quan hệ
Random Forest là một thuật toán **Học có giám sát (Supervised Learning)**. Để hoạt động, nó cần cả đầu vào ($X$) và nhãn đích ($y$ - cụ thể trong dự án này là `requests_target`).

* **Không chỉ là đường thẳng:** Khác với các mô hình tuyến tính (chỉ tìm mối quan hệ đơn giản kiểu "A tăng thì B tăng"), Random Forest sử dụng hàng loạt các câu hỏi **"Nếu - Thì" (If-Else)** trong các cây quyết định để phân mảnh dữ liệu.
* **Mối quan hệ phi tuyến:** Thuật toán có khả năng nhận diện các quy luật phức tạp. 
    * *Ví dụ:* "Nếu là khung giờ từ 8h-11h **VÀ** không phải cuối tuần, thì tải sẽ cực cao". 
    * Đây là những mối quan hệ mà các hàm toán học đơn giản rất khó mô tả chính xác.

---

## 2. Chỉ số quan trọng (Feature Importance) được tính như thế nào?
Khi gọi thuộc tính `rf.feature_importances_` trong mã nguồn, mô hình đang thực hiện một quá trình "chấm điểm" khách quan:

1. **Đo lường sự đóng góp:** Mỗi khi một cây trong "khu rừng" sử dụng một cột (ví dụ: `hour_sin`) để chia dữ liệu, nó sẽ tính toán mức độ giảm sai số (**MSE - Mean Squared Error**) sau khi chia.
   $$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
2. **Cộng dồn điểm số:** Cột nào giúp giảm sai số càng nhiều và được sử dụng thường xuyên ở các tầng trên cùng của cây thì điểm **Importance** càng cao.
3. **Kết quả cuối cùng:** Tổng điểm của mỗi cột trên toàn bộ 50 cây (`n_estimators=50`) sẽ được chuẩn hóa về dạng phần trăm (tổng bằng $1.0$ hoặc $100\%$).

---

## 3. Tại sao bước này lại "cứu nguy" cho LSTM?
Việc thực hiện hàm `select_features_lstm` mang lại 3 lợi ích chiến lược:

* **Lọc nhiễu:** Nếu một cột dữ liệu hoàn toàn ngẫu nhiên và không có giá trị dự báo, Random Forest sẽ chấm điểm gần bằng $0$. Điều này cho phép chúng ta loại bỏ rác dữ liệu ngay lập tức.
* **Giảm độ phức tạp:** LSTM vốn nổi tiếng là "ngốn" tài nguyên và học lâu. Bằng cách chỉ chọn 5 đặc trưng quan trọng nhất (ví dụ: `requests_target`, `error_rate`, `hour_sin`, `hour_cos`, `is_weekend`), bạn giúp mô hình hội tụ nhanh hơn và tránh hiện tượng **Quá khớp (Overfitting)**.
* **Xác nhận trực quan:** Biểu đồ thanh (`kind='barh'`) cung cấp bằng chứng toán học đanh thép cho việc lựa chọn đặc trưng, thay vì dựa vào cảm tính cá nhân.

# Tại sao dùng Random Forest để chọn Feature cho LSTM?

Sự kết hợp giữa Random Forest và LSTM giúp tối ưu hóa quá trình huấn luyện và nâng cao độ chính xác cho các bài toán dự báo chuỗi thời gian.

---

## A. Tìm ra "Tín hiệu" trong "Nhiễu"
Mô hình **LSTM** rất mạnh nhưng lại khá "nhạy cảm" và tốn thời gian huấn luyện. 

* **Thách thức:** Nếu đưa quá nhiều đặc trưng rác (noise) vào, LSTM sẽ mất rất nhiều thời gian để hội tụ hoặc thậm chí học sai lệch. 
* **Giải pháp:** **Random Forest** giúp tính toán mức độ đóng góp của từng đặc trưng vào việc giảm sai số dự báo. Những đặc trưng có điểm số quan trọng (Importance Score) thấp sẽ bị loại bỏ, giúp mô hình trở nên gọn nhẹ và tập trung vào những "tín hiệu" thực sự giá trị.

---

## B. Khả năng xử lý đa dạng loại dữ liệu
Trong các bài toán thực tế (như dự án NASA-HTTP workload), dữ liệu thường rất đa dạng:
* **Số thực:** `error_rate`.
* **Số nhị phân:** `is_weekend`.
* **Giá trị lượng giác (Cyclic Encoding):** `hour_sin`, `hour_cos`.

**Random Forest** có thể xử lý tất cả các loại dữ liệu này cùng một lúc mà không yêu cầu chuẩn hóa dữ liệu (scaling) quá phức tạp. Điều này giúp bạn nhanh chóng có cái nhìn tổng quan về tầm quan trọng của các biến số ngay từ giai đoạn tiền xử lý.

---

## C. Sự kết hợp giữa "Cấu trúc" và "Thời gian"
Đây là sự phân chia nhiệm vụ hoàn hảo trong một hệ thống dự báo:

1. **Random Forest (Xác định mối quan hệ cấu trúc):** Nó nhận diện các quy luật tĩnh và logic. 
    * *Ví dụ:* "Cứ đến cuối tuần (`is_weekend`) là tải hệ thống thay đổi" — quy luật này đúng bất kể đó là tuần nào trong năm.
2. **LSTM (Xử lý mối quan hệ thời gian):** Sau khi đã có các đặc trưng quan trọng nhất, LSTM sẽ tiếp quản để học các phụ thuộc động.
    * *Ví dụ:* "Tải của 5 phút trước ảnh hưởng thế nào đến 5 phút sau" — tập trung vào tính tuần tự và xu hướng ngắn hạn/dài hạn.