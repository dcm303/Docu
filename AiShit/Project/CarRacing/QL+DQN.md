# Từ Q-Learning truyền thống đến Thách thức của Môi trường Phức tạp

Tài liệu này hệ thống hóa cơ chế hoạt động của Q-Learning thông qua khái niệm Q-Table và phân tích lý do tại sao thuật toán này gặp bế tắc trước các bài toán có không gian trạng thái lớn như CarRacing.

---

## 1. Cơ chế: "Sổ tay ghi chép" (Q-Table)

Q-Learning hoạt động như một cái bảng tra cứu khổng lồ, nơi Agent lưu trữ "kinh nghiệm" của mình cho mọi tình huống có thể xảy ra.

* **Hàng:** Tất cả các trạng thái có thể có ($s$).
* **Cột:** Tất cả các hành động có thể làm ($a$).
* **Giá trị $Q(s, a)$:** Tổng phần thưởng dự kiến thu được nếu thực hiện hành động $a$ tại trạng thái $s$.



### Quy trình "Thử và Sai"
1.  **Exploration (Khám phá):** Ban đầu, nhờ vào tham số $\epsilon$ (Epsilon), Agent sẽ thực hiện "đi liều" (chọn ngẫu nhiên) để khám phá các ngóc ngách của môi trường.
2.  **Học hỏi:** Sau khi nhận phần thưởng $R$ và quan sát trạng thái mới $s'$, nó cập nhật lại "tri thức" vào bảng bằng phương trình Bellman:

$$Q(s, a) \leftarrow Q(s, a) + \alpha [R + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$

> **Ý nghĩa:** Đây chính là cách tri thức được lan truyền ngược từ những kết quả ở tương lai về hành động ở hiện tại.

---

## 2. Điểm yếu: "Bùng nổ chiều dữ liệu" (Curse of Dimensionality)

Đây chính là lý do cốt yếu khiến Q-Learning thuần túy không thể áp dụng cho dự án CarRacing hay các bài toán xử lý hình ảnh.

### Bài toán định lượng
Hãy cùng tính toán quy mô của cái "sổ tay" này đối với môi trường CarRacing:
* **Trạng thái:** Ảnh xám kích thước $84 \times 84$ pixel.
* **Mỗi Pixel:** Có 256 giá trị cường độ sáng khác nhau (từ 0 đến 255).
* **Tổng số trạng thái có thể có:** $$256^{(84 \times 84)}$$

Đây là một con số khổng lồ, thậm chí còn **nhiều hơn cả số nguyên tử trong vũ trụ**.



### Hệ quả thực tế
* **Quá nặng (Storage Limit):** Không có thiết bị lưu trữ nào hiện nay có thể chứa nổi một cái bảng có hàng tỷ tỷ hàng như vậy.
* **Học quá chậm (Convergence Issue):** Agent sẽ không bao giờ khám phá hết được các trạng thái để cập nhật giá trị Q. Nếu mỗi giây gặp một trạng thái mới, nó cần đến hàng tỷ năm để "thuộc lòng" được cái bảng này.

---

**Kết luận:** Để giải quyết bài toán này, chúng ta cần một "bộ não" có khả năng khái quát hóa thay vì ghi nhớ máy móc từng pixel. Đó là lúc Deep Q-Network (DQN) xuất hiện để thay thế bảng Q-Table bằng một mạng nơ-ron tích chập (CNN).