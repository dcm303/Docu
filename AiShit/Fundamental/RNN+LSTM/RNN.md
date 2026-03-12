RNN được sinh ra trong bối cảnh MLP coi các đầu vào như các dữ liệu hoàn toàn độc lập và không có khái niệm về thứ tự hoặc thời gian và được dùng để giải quyết các bài toàn có dữ liệu dạng chuỗi như NLP, Chuỗi thời gian (Time-series), ...


À thế thì tôi hiểu về RNN rồi: RNN là mạng cải tiến từ mạng MLP bằng việc thêm thông tin từ quá khứ để đưa ra output. Cụ thể là việc đưa vào một lớp hidden state là quá trình cập nhật và tổng hợp kiến thức nhờ công thức $$h_t = \sigma(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)$$ trong đó h sẽ là tổng hợp kiến thức từ đầu cho đến thời điểm t là kết quả của hidden state tại t - 1 nhân trọng số cùng input hiện tại nhân trọng số và bias được đưa qua hàm kích hoạt. Và từ đó output tại mỗi thời điểm t đó sẽ là hidden state tại t nhân trọng số cùng bias đi qua hàm kích hoạt. RNN sẽ xử lý tuần từ từng điểm dữ liệu và luôn dùng một bộ trọng số Why, Whh, Wxh trong suốt thời gian xử lý từng điểm dữ liệu đó. Việc này giúp cho số lượng lớp neural không phụ thuộc vào độ dài input (vì xử lý tuần tự) và giúp cho mạng học được quy luật chung cố định với toàn bộ điểm (thay vì học trọng số ưu tiên của từng điểm dữ liệu) phù hợp với các bài toán dữ liệu dạng chuỗi như NLP và time-series. Tuy nhiên, khi nói đến quá trình lan truyền ngược là phép nhân liên tiếp các đạo hàm của hidden state từ lớp hiện tại dần dần về lớp đầu tiên (việc này diễn ra tại mỗi điểm t sau khi đã đưa ra output và tính loss). Việc xử lý tuần từ từng điểm này khiến cho việc lan truyền ngược phụ thuộc vào độ dài input (độ dài input có thể quá lớn) có thể dẫn đến vấn đề vanishing gradient hoặc exploding gradient khi nhân liên tiếp các giá trị bé hơn 1 hoặc lớn hơn 1 dẫn đến việc cập nhập trọng số bị ảnh hưởng

# Kiến trúc và Cơ chế vận hành của Mạng Neural Hồi quy (RNN)

Tài liệu này đi sâu vào cấu trúc "trí nhớ" của RNN, lợi thế của việc dùng chung trọng số và những thách thức toán học trong quá trình huấn luyện.

---

## 🏗️ 1. Kiến trúc và Cơ chế Lan truyền xuôi (Forward Propagation)

RNN (Recurrent Neural Network) được coi là mạng Neural có "trí nhớ", khác biệt hoàn toàn với MLP truyền thống chỉ xử lý dữ liệu độc lập.

### Hidden State ($h_t$): Bản tóm tắt kiến thức
Như bạn đã nêu, $h_t$ không chỉ là kết quả của một phép tính, nó là trạng thái ẩn tích lũy ngữ cảnh từ quá khứ.

$$h_t = \sigma(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b_h)$$

* **Ý nghĩa thực tế:** Tại mỗi thời điểm $t$, mạng thực hiện một phép "hợp nhất": lấy thông tin mới ($x_t$) trộn với ký ức cũ ($h_{t-1}$).
* **Hàm kích hoạt ($\sigma$):** Thường là `tanh` hoặc `ReLU`. Nó đóng vai trò tạo ra các "nếp gấp" phi tuyến tính, giúp mạng học được các quy luật biến thiên phức tạp trong dữ liệu chuỗi thời gian của NASA.

### Output ($y_t$): Quyết định dựa trên ngữ cảnh
Đầu ra tại mỗi bước được tính dựa trên "bản tóm tắt" $h_t$ vừa được cập nhật:

$$y_t = \sigma(W_{hy} \cdot h_t + b_y)$$

Việc này đảm bảo rằng mọi quyết định (ví dụ: dự báo tải server tăng hay giảm) đều dựa trên toàn bộ lịch sử trước đó chứ không chỉ là con số request hiện tại.

---

## 🔄 2. "Weight Sharing": Bí mật của sự linh hoạt

Việc dùng chung một bộ trọng số ($W_{xh}, W_{hh}, W_{hy}$) cho mọi bước thời gian mang lại những lợi thế khổng lồ:

1. **Tính bất biến theo thời gian (Temporal Invariance):** Mạng học được quy luật chung. Ví dụ, quy luật "Request tăng đột biến dẫn đến quá tải" là như nhau dù nó xảy ra lúc 8 giờ sáng hay 2 giờ chiều trong tập dữ liệu log.
2. **Tối ưu hóa tham số:** Số lượng tham số không thay đổi bất kể đầu vào dài hay ngắn. Điều này giúp mô hình nhẹ hơn, tránh **Overfitting** và có khả năng xử lý các chuỗi dữ liệu có độ dài linh hoạt (điều mà MLP không thể làm được).

---

## 📉 3. "Gót chân Achilles": BPTT và Vấn đề Gradient

Dù kiến trúc rất thông minh, nhưng phương pháp học của RNN — **Lan truyền ngược qua thời gian (BPTT)** — lại chứa đựng một thảm họa toán học.

### Cơ chế nhân liên tiếp
Tại mỗi thời điểm $t$, sau khi tính Loss ($L_t$), ta cần cập nhật trọng số chung $W$. Vì $h_t$ phụ thuộc vào $h_{t-1}$, $h_{t-1}$ lại phụ thuộc vào $h_{t-2}$... nên khi tính đạo hàm, ta phải áp dụng quy tắc chuỗi (**Chain Rule**) qua rất nhiều bước.

### Vanishing & Exploding Gradient
* **Vanishing (Biến mất):** Nếu các đạo hàm trung gian (liên quan đến trọng số $W$) nhỏ hơn 1, việc nhân liên tiếp qua 100-200 bước thời gian sẽ khiến đạo hàm tiến về 0.
    * **Hệ quả:** Trọng số không được cập nhật cho các thông tin ở xa. Mạng "quên" mất những gì xảy ra ở đầu chuỗi.
* **Exploding (Bùng nổ):** Nếu đạo hàm lớn hơn 1, kết quả nhân liên tiếp sẽ vọt lên cực lớn, khiến trọng số bị cập nhật quá mạnh, làm mô hình mất ổn định hoặc gặp lỗi `NaN`.