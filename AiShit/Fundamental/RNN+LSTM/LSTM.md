LSTM sẽ giải quyết nhược điểm về vanishing gradient của RNN bằng việc đưa ra bộ nhớ Cell State và các cổng: cổng quên, cổng nhớ, cổng cập nhật vào cell state và cổng đầu ra output gate. Cổng quên ft: $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$ sẽ quyết định mức độ quên các thông tin cũ dựa trên h (output) trước đó và input hiện tại và cổng nhớ $$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$ sẽ quyết định vị trí để nạp vào Cell State



$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$ sẽ đưa ra ứng viên mới để nạp vào Cell State (C ở đây sẽ là một vector nhiều chiều chứa các dữ liệu). Tiếp theo cổng cập nhật trạng thái ô $$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$ sẽ lấy kí ức cũ (mức độ quên ft nhân với Cell state trước đó) cộng với thông tin mới (sau khi đã quyết định được các vị trí nên nạp vào) -> chính phép cộng này giúp cho việc loại bỏ được vanishing gradient khi ở RNN ht sẽ được cập nhật bằng việc nhân chồng chất dần dần từ lớp cuối đến lớp đầu tiên các trọng số cố định dẫn đến việc không thể vừa ưu tiên thông tin quan trọng hiện tại vừa đưa được thông tin đi lâu dài. Mà thay vào đó LSTM nhờ vào phép cộng khi đạo hàm sẽ xấp xỉ ft nên việc cập nhật sẽ là chồng chất các phép nhân của ft và các ft này là khác nhau trong từng giai đoạn nên mạng có thể học và quyết định thông tin nào là quan trọng thông tin nào là không và học cách để truyền nó đi bằng cách điều chỉnh ft. Cuối cùng sẽ là đến cổng output $$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$ sẽ quyết định phần nào của bộ nhớ dài hạn sẽ được xất ra $$h_t = o_t * \tanh(C_t)$$ sẽ là kết quả cuối cùng của cell này và ht này có thể là output hoặc đi qua một hàm tuyến tính để đưa ra output


# 🧠 Tổng quan về "Bộ não" LSTM (Long Short-Term Memory)

LSTM được thiết kế như một hệ thống quản lý thông tin chủ động. Khác với RNN "thụ động" để thông tin bị bào mòn qua thời gian, LSTM có khả năng tự điều tiết dòng chảy dữ liệu.

---

## 1. Hai thành phần lưu trữ cốt lõi

* **Cell State ($C_t$):** Được ví như một đường cao tốc thông tin (**Long-term memory**). Nó chạy xuyên suốt các bước thời gian với rất ít tương tác phi tuyến tính, giúp bảo toàn dữ liệu quan trọng từ quá khứ xa xôi.
* **Hidden State ($h_t$):** Trí nhớ ngắn hạn (**Short-term memory**). Nó là kết quả được trích xuất từ Cell State để đưa ra dự báo ngay lập tức và làm ngữ cảnh cho bước tiếp theo.

---

## 🚪 Cơ chế vận hành của 3 Cánh cổng (Gates)

Mọi cổng trong LSTM đều sử dụng hàm **Sigmoid** để đưa ra giá trị từ $0$ (quên sạch/đóng) đến $1$ (nhớ hết/mở).

### A. Cổng Quên (Forget Gate - $f_t$)
Đây là "bộ lọc nhiễu". Nó nhìn vào quá khứ ($h_{t-1}$) và hiện tại ($x_t$) để quyết định những gì không còn giá trị trong "sổ cái" dài hạn.
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

### B. Cổng Nhớ (Input Gate - $i_t$ & $\tilde{C}_t$)
Quyết định nạp thêm kiến thức mới vào bộ nhớ.
* **$i_t$**: Quyết định vị trí (chiều đặc trưng) nào trong vector Cell State sẽ được cập nhật.
* **$\tilde{C}_t$**: Tạo ra nội dung mới để nạp vào (Candidate).
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

### C. Cổng Xuất (Output Gate - $o_t$)
Quyết định xem từ "sổ cái" dài hạn, thông tin nào là cần thiết để đưa ra câu trả lời cho hiện tại.
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

---

## 🛠 Giải pháp cho lỗi "Mất trí nhớ" (Vanishing Gradient)

Sự cải tiến của LSTM nằm ở việc thay thế phép nhân chồng chất bằng **phép cộng tuyến tính** thông tin mới vào Cell State.

| Đặc điểm | RNN (Truyền thống) | LSTM (Cải tiến) |
| :--- | :--- | :--- |
| **Cập nhật bộ nhớ** | Nhân ma trận trọng số $W$ cố định liên tiếp. | Cộng tuyến tính thông tin mới vào Cell State. |
| **Dòng chảy đạo hàm** | Bị triệt tiêu về $0$ do nhân lũy thừa $W^n$. | Xấp xỉ bằng tích các cổng quên $f_t$. |
| **Tính linh hoạt** | "Tĩnh" - Đạo hàm luôn bị giảm nếu $W < 1$. | "Động" - Mạng tự học cách đặt $f_t \approx 1$ để giữ đạo hàm. |