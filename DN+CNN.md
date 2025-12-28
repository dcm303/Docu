
# Giải Mã Backpropagation: Ví dụ Số liệu Cụ thể & Công thức

Để hiểu tường tận "ma thuật" của Backpropagation, chúng ta cần đi sâu vào số liệu và công thức toán học. Dưới đây là mô phỏng quy trình trên một **Mô hình CNN tí hon**.

---

## 🐯 Kịch bản ví dụ: "Phát hiện sọc hổ"

**Mục tiêu:** Huấn luyện mạng nhận diện xem ảnh có "Sọc dọc" hay không.

* **Input ($X$):** Một mảnh ảnh bé xíu 3x1 pixel. Giá trị `[10, 10, 0]` (Sáng - Sáng - Tối $\to$ Có biên dọc).
* **Nhãn thật ($Y_{true}$):** $1$ (Là Hổ).
* **Kiến trúc:** 1 Lớp Conv (1 Filter 2x1) $\to$ 1 Lớp Output (Tổng Feature Map).

---

### PHẦN 1: FORWARD PASS (TÍNH TOÁN XUÔI)
*Mục tiêu: Xem mạng hiện tại (đang "ngu ngơ") đoán ra cái gì.*

**Bước 1: Convolution**
Máy tính khởi tạo ngẫu nhiên một Filter ($W_1$) kích thước 2x1. Giả sử khởi tạo: `[0.5, -0.5]`.
Ta trượt Filter này qua Input `[10, 10, 0]`:

* **Vị trí 1 (Pixel 1, 2):** `[10, 10]` nhân với `[0.5, -0.5]`
    $$(10 \times 0.5) + (10 \times -0.5) = 5 - 5 = 0$$
* **Vị trí 2 (Pixel 2, 3):** `[10, 0]` nhân với `[0.5, -0.5]`
    $$(10 \times 0.5) + (0 \times -0.5) = 5 - 0 = 5$$

$\Rightarrow$ **Feature Map:** `[0, 5]`

**Bước 2: Ra quyết định (Output Layer)**
Cộng tổng Feature Map để ra điểm số cuối cùng.

$$Y_{pred} = 0 + 5 = 5$$

---

### PHẦN 2: TÍNH TOÁN SAI SỐ (LOSS)
*Mục tiêu: Biết mạng đang sai bao nhiêu.*

Ta mong muốn kết quả là **10** (Ví dụ quy định), nhưng máy chỉ ra **5**.
Công thức Loss (Bình phương sai số):

$$L = (Y_{pred} - Y_{true})^2$$

Tính toán:
$$L = (5 - 10)^2 = (-5)^2 = 25$$

$\Rightarrow$ **Kết luận:** Lỗi là 25. Rất lớn! Cần sửa Filter `[0.5, -0.5]` ngay.

---

### PHẦN 3: BACKPROPAGATION (TRUY TÌM TRÁCH NHIỆM)
*Mục tiêu: Biết cần sửa số 0.5 hay -0.5, và sửa bao nhiêu.*

Ta cần tính Đạo hàm của Lỗi theo Trọng số ($W$): $\frac{\partial L}{\partial W}$.
Áp dụng **Quy tắc chuỗi (Chain Rule)**:

$$\frac{\partial L}{\partial W} = \underbrace{\frac{\partial L}{\partial Y_{pred}}}_{\text{Lỗi do Dự đoán}} \times \underbrace{\frac{\partial Y_{pred}}{\partial W}}_{\text{Dự đoán do Trọng số}}$$

**Khúc 1: Lỗi thay đổi thế nào theo Dự đoán?**
Hàm Loss: $L = (Y - 10)^2$. Đạo hàm là $2 \times (Y - 10)$.
$$\frac{\partial L}{\partial Y_{pred}} = 2 \times (5 - 10) = -10$$
*(Ý nghĩa: Số âm nghĩa là Dự đoán đang thấp quá, cần tăng lên).*

**Khúc 2: Dự đoán thay đổi thế nào theo Trọng số?**

Nhìn lại công thức Forward tại vị trí tạo ra kết quả 5:
$$\text{Kết quả} = (\text{Pixel}_2 \times W_{\text{trái}}) + (\text{Pixel}_3 \times W_{\text{phải}})$$
$$5 = (10 \times 0.5) + (0 \times -0.5)$$

* Với $W_{\text{trái}}$ (0.5): Nhân với Pixel 2 (giá trị 10) $\to$ Đạo hàm là **10**.
* Với $W_{\text{phải}}$ (-0.5): Nhân với Pixel 3 (giá trị 0) $\to$ Đạo hàm là **0**.

**TỔNG HỢP (Gradient):**

1.  **Gradient cho $W_{\text{trái}}$:**
    $$\text{Grad}_1 = (-10) \times 10 = -100$$
    *(Cần tăng trọng số này thật mạnh).*

2.  **Gradient cho $W_{\text{phải}}$:**
    $$\text{Grad}_2 = (-10) \times 0 = 0$$
    *(Không đóng góp vào lỗi sai, không cần sửa).*

---

### PHẦN 4: WEIGHT UPDATE (SỬA SAI)
*Mục tiêu: Ra được Filter mới xịn hơn.*

Công thức cập nhật (với Learning Rate $\eta = 0.01$):
$$W_{\text{mới}} = W_{\text{cũ}} - (\eta \times \text{Gradient})$$

* **Sửa $W_{\text{trái}}$ (Cũ là 0.5):**
    $$W_{\text{mới}} = 0.5 - (0.01 \times -100) = 0.5 - (-1) = 1.5$$
* **Sửa $W_{\text{phải}}$ (Cũ là -0.5):**
    $$W_{\text{mới}} = -0.5 - (0.01 \times 0) = -0.5$$

$\Rightarrow$ **Filter Mới:** `[1.5, -0.5]`.

---

### TỔNG KẾT
Công thức cốt lõi để mạng Deep Learning tìm ra tham số tối ưu:

$$W_{new} = W_{old} - \eta \cdot \nabla Loss$$

Logic: Nếu Input $X$ lớn mà gây ra lỗi, thì $W$ phải chịu trách nhiệm lớn. Nếu Input $X=0$, thì $W$ vô tội.