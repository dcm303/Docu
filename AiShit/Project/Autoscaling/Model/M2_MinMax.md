# Quy trình Chuẩn bị Dữ liệu cho Mô hình LSTM (NASA-HTTP Workload)

Tài liệu này hệ thống lại các bước tiền xử lý dữ liệu quan trọng, đảm bảo tính khách quan và hiệu quả cho mô hình dự báo chuỗi thời gian.

---

## 1. Chia tập dữ liệu (Split) theo trình tự thời gian
Khác với các bài toán phân loại ảnh (thường trộn ngẫu nhiên), dữ liệu chuỗi thời gian bắt buộc phải chia theo đúng trình tự để tránh việc mô hình "biết trước tương lai". Tỷ lệ phân chia được áp dụng là **60/20/20**:

* **Tập Huấn luyện (Train):** Từ đầu đến hết ngày 16/08. Đây là dữ liệu mô hình dùng để học quy luật.
* **Tập Kiểm định (Validation):** Từ 17/08 đến 22/08. Dùng để điều chỉnh siêu tham số và thực hiện **Early Stopping** (dừng huấn luyện khi mô hình bắt đầu "học vẹt").
* **Tập Kiểm thử (Test):** Từ ngày 23/08 trở đi. Đây là dữ liệu "mới hoàn toàn" để đánh giá sức mạnh thực tế của mô hình trước khi triển khai.

---

## 2. Chuẩn hóa dữ liệu (Scaling)
Sử dụng `MinMaxScaler` để đưa tất cả giá trị về khoảng $[0, 1]$.

* **Lý do:** Các hàm kích hoạt trong LSTM như `tanh` hay `sigmoid` hoạt động tốt nhất và tránh được hiện tượng triệt tiêu đạo hàm khi dữ liệu nằm trong khoảng nhỏ này.
* **Điểm kỹ thuật quan trọng:** Chỉ thực hiện `fit` (tính toán Min, Max) trên tập **Train**, sau đó dùng các thông số đó để `transform` cho tập **Val** và **Test**.
* **Tại sao?** Nếu `fit` trên toàn bộ dữ liệu, mô hình sẽ biết được giá trị cực đại của tương lai nằm ở đâu, dẫn đến lỗi **Data Leakage** (Rò rỉ dữ liệu), khiến kết quả thực tế không chính xác.

---

## 3. Tạo cửa sổ trượt (Sliding Window)
Thông qua lớp `TimeSeriesDataset`, dữ liệu được biến đổi thành các cặp ($Input$, $Label$):

* **Input ($X$):** Một chuỗi gồm 12 bước thời gian liên tiếp (tương đương 60 phút quá khứ).
* **Label ($y$):** Giá trị `requests_target` của bước thời gian thứ 13 (5 phút tiếp theo).

---

## 4. Đóng gói vào DataLoader
Dữ liệu cuối cùng được đưa vào `DataLoader` của PyTorch để tối ưu hiệu suất:

* **batch_size=32:** Nạp 32 chuỗi cùng lúc để tối ưu hóa tốc độ xử lý của CPU/GPU, thay vì nạp từng dòng đơn lẻ.
* **shuffle=True (Chỉ cho tập Train):** Giúp thuật toán tối ưu (Adam) không bị rơi vào các cực trị cục bộ do tính chu kỳ của dữ liệu.

> ### 💡 Câu hỏi phỏng vấn "hóc búa":
> **"Tại sao ở tập Train bạn lại dùng shuffle=True, trong khi bạn vừa nói dữ liệu chuỗi thời gian không được trộn ngẫu nhiên?"**
>
> **Gợi ý trả lời:** Chúng ta tuyệt đối không trộn thứ tự các dòng *trước* khi tạo cửa sổ trượt (Windowing) để đảm bảo mỗi cửa sổ chứa 12 bước thời gian liên tiếp đúng thực tế. Tuy nhiên, sau khi đã có hàng nghìn cửa sổ độc lập, việc trộn thứ tự các cửa sổ này khi đưa vào các Batch huấn luyện sẽ giúp mô hình hội tụ ổn định hơn và học được các đặc điểm tổng quát mà không bị lệ thuộc vào thứ tự xuất hiện của các ngày.


# Quy trình Xử lý Dữ liệu cho Mô hình LSTM

Tài liệu này giải thích chi tiết các bước chuẩn bị dữ liệu, từ việc chia tập dữ liệu theo trình tự thời gian đến các kỹ thuật chuẩn hóa và tạo cửa sổ trượt để tối ưu hóa hiệu suất mô hình.

---

## 1. Tại sao chia tỷ lệ 60/20/20?
Trong dự án này, việc phân chia theo tỷ lệ 60/20/20 đảm bảo sự cân bằng giữa học tập, điều chỉnh và kiểm tra khách quan:

* **Tập Train (60%) - "Học tập":** Đây là lượng dữ liệu lớn nhất để mô hình có đủ "vốn kiến thức" học được các quy luật như: giờ cao điểm ban ngày, tải thấp ban đêm, và các xu hướng ngày thường.
* **Tập Validation (20%) - "Thi thử":** Dùng để điều chỉnh các siêu tham số (như `hidden_size`, `num_layers`). Quan trọng nhất, tập này được dùng để thực hiện **Early Stopping** — dừng huấn luyện ngay khi mô hình bắt đầu "học vẹt" trên tập Train nhưng sai số trên tập Validation bắt đầu tăng.
* **Tập Test (20%) - "Thi thật":** Đây là dữ liệu hoàn toàn mới. Kết quả trên tập này là bằng chứng thực tế nhất cho thấy hệ thống hoạt động tốt thế nào khi triển khai thực tế.

**Lưu ý về tính tuần tự:** Vì là chuỗi thời gian, việc phân chia phải tuân thủ mốc thời gian (Tháng 7 $\rightarrow$ 16/08 $\rightarrow$ 22/08 $\rightarrow$ 31/08). Chúng ta không thể lấy dữ liệu tương lai để huấn luyện cho quá khứ.

---

## 2. Ý nghĩa của MinMaxScaler và sự khác biệt giữa các tập
`MinMaxScaler` đưa toàn bộ các cột dữ liệu (từ số lượng request hàng nghìn đến tỷ lệ lỗi 0.01) về cùng một khoảng giá trị từ 0 đến 1.

* **Tại sao dùng cho LSTM?** Các hàm kích hoạt bên trong mạng LSTM như $\tanh$ hay $Sigmoid$ hoạt động hiệu quả nhất trong khoảng $(-1, 1)$ hoặc $(0, 1)$. Nếu dữ liệu đầu vào quá lớn (ví dụ: 5000), các đạo hàm sẽ bị bão hòa, dẫn đến hiện tượng triệt tiêu đạo hàm, khiến mô hình không thể học được.
* **Kỹ thuật tránh Data Leakage (Rò rỉ dữ liệu):**
    * **Tập Train:** Dùng `fit_transform` để máy tính "ghi nhớ" giá trị Min và Max của tập này.
    * **Tập Val/Test:** Chỉ dùng `transform` dựa trên Min/Max của tập Train đã lưu trước đó.
* **Lý do:** Nếu lấy Max của tập Test để chuẩn hóa, mô hình sẽ vô tình "biết trước" các đỉnh tải trong tương lai, làm kết quả dự báo mất đi tính khách quan.

---

## 3. Tại sao phải sử dụng Sliding Window (Cửa sổ trượt)?
Mô hình LSTM cần "ngữ cảnh" để hiểu được xu hướng thay đổi thay vì nhìn vào từng dòng dữ liệu rời rạc.

* **Biến đổi bài toán:** Sliding window giúp biến dữ liệu chuỗi thời gian thành bài toán **Học có giám sát (Supervised Learning)** với đầu vào $X$ và nhãn $y$.
* **Cách hoạt động:** Với `sequence_length=12`, chúng ta gom 12 dòng liên tiếp (tương đương 60 phút quá khứ) thành một khối dữ liệu duy nhất.
* **Mục đích:** Giúp mô hình học được mối quan hệ động: "Nếu 60 phút vừa qua tải đang tăng dần, thì 5 phút tới tải sẽ tiếp tục tăng hay bắt đầu giảm?". Không có cửa sổ trượt, LSTM không thể hiểu được vận tốc hay xu hướng của lưu lượng.

---

### Tóm lại:
1. **60/20/20:** Huấn luyện và kiểm tra công bằng theo đúng trình tự thời gian.
2. **MinMaxScaler:** Giúp các phép toán bên trong LSTM ổn định và hội tụ nhanh.
3. **Fit trên Train, Transform trên Test:** Đảm bảo mô hình không "ăn gian" nhìn trước tương lai.
4. **Sliding Window:** Cung cấp ngữ cảnh quá khứ để đưa ra dự báo chính xác.