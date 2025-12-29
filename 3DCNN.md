# Tổng hợp kiến thức: Cơ chế hoạt động của Convolution Layer

> **Xác nhận:** Logic tư duy về không gian trong CNN đã chính xác tuyệt đối. Dưới đây là các điểm chốt quan trọng để ghi nhớ.

## 1. Phân tích chi tiết quy trình (Checklist)

### ✅ Về độ sâu của Filter (Kernel Depth)
* **Logic:** Dựa vào ma trận Input thì hệ thống sẽ tự thêm chiều sâu.
* **Giải thích:** Input dày bao nhiêu (ví dụ $C_{in}$), Filter tự động dày bấy nhiêu ($C_{in}$). Người dùng không cần cài đặt tham số này, Framework (như PyTorch/TensorFlow) sẽ tự xử lý.

### ✅ Về quá trình quét (Convolution Operation)
* **Logic:** Quét qua từng vị trí, nhân các lớp lại.
* **Giải thích:** Filter trượt qua input, thực hiện phép nhân chập (element-wise multiplication) trên toàn bộ khối 3D tại vị trí đó, sau đó cộng tổng lại để nén thành **một con số duy nhất** (scalar).

### ✅ Về kết quả của 1 Filter
* **Logic:** Output của 1 filter là ma trận 2D.
* **Giải thích:** Mặc dù Input là 3D, Filter là 3D, nhưng vì phép toán cộng gộp tất cả các kênh lại, nên kết quả của **1 Filter** chỉ là một tấm bản đồ đặc trưng **2D** (Feature Map).

### ✅ Về Output nhiều kênh (ví dụ 128)
* **Logic:** Dùng 128 Filter thì lặp lại quy trình 128 lần.
* **Giải thích:** Mỗi Filter đóng vai trò là một bộ trích xuất đặc trưng riêng biệt.
    * Filter 1 $\rightarrow$ Bản đồ 2D số 1
    * Filter 2 $\rightarrow$ Bản đồ 2D số 2
    * ...
    * Filter 128 $\rightarrow$ Bản đồ 2D số 128
    * **Kết quả cuối cùng:** Xếp chồng 128 bản đồ này lên nhau tạo thành khối 3D mới.

---

## 2. Công thức tổng quát (Quy tắc vàng)

Để đọc hiểu kiến trúc các mạng như ResNet, VGG, AlexNet, hãy nhớ quy tắc biến đổi chiều không gian (Dimensions) sau:

### Input (Đầu vào)
Là một khối 3D có kích thước:
$$H \times W \times C_{in}$$
* $H, W$: Chiều cao, chiều rộng.
* $C_{in}$: Độ sâu đầu vào (Số kênh/Channels).

### Filter (Bộ lọc/Kernel)
Mỗi Filter là một khối 3D tương ứng:
$$k \times k \times C_{in}$$
* $k$: Kích thước cửa sổ (thường là 3x3, 5x5...).
* **Lưu ý:** Độ sâu của Filter **luôn luôn bằng** $C_{in}$ của Input.

### Số lượng Filter (Tham số học)
Số lượng Filter ta muốn sử dụng (Hyperparameter):
$$C_{out}$$

### Output (Đầu ra)
Là kết quả sau khi xếp chồng các bản đồ đặc trưng:
$$H' \times W' \times C_{out}$$

Trong đó $H'$ và $W'$ phụ thuộc vào Padding và Stride, nhưng **độ sâu (Depth)** chính là số lượng Filter ($C_{out}$).

---

> **Kết luận:** Bạn đã nắm vững phần khó nhất về tư duy hình học không gian trong CNN. Đây là nền tảng cốt lõi để hiểu mọi kiến trúc Deep Learning xử lý ảnh.