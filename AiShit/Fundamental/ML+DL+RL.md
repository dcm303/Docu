# Tổng quan về Machine Learning, Deep Learning và Reinforcement Learning

Nội dung này hệ thống lại các khái niệm cốt lõi và các dự án thực tế đã triển khai, từ học máy truyền thống đến các kiến trúc mạng thần kinh phức tạp và học tăng cường.

---

## 1. Machine Learning (ML) - Học máy truyền thống

ML là quá trình sử dụng các thuật toán thống kê để tìm ra các mẫu (patterns) trong dữ liệu mà không cần các chỉ dẫn lập trình cụ thể cho từng trường hợp.

**Quy trình cốt lõi:**
Dữ liệu đầu vào ($X$) $\rightarrow$ Trích xuất đặc trưng (Feature Engineering) $\rightarrow$ Mô hình ($f$) $\rightarrow$ Dự đoán ($y$).

**Điểm đặc trưng:** Bạn phải là người "dạy" máy biết cái gì là quan trọng. Ví dụ, trong dự án **NASA-HTTP workload**, việc thực hiện Feature Engineering thủ công là then chốt:
* **Lag features:** Giúp máy hiểu dữ liệu quá khứ ảnh hưởng thế nào đến tương lai.
* **Cyclic time encoding:** Giúp máy hiểu được tính chu kỳ của thời gian (ví dụ: giờ thứ 23 và giờ thứ 00 rất gần nhau).

**Toán học nền tảng:**
* **Hàm mất mát (Loss Function):** Đo lường sự sai khác giữa dự đoán và thực tế, ví dụ MSE: 
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
* **Tối ưu hóa:** Sử dụng **Gradient Descent** để tìm bộ tham số giúp cực tiểu hóa hàm mất mát.

---

## 2. Deep Learning (DL) - Học sâu

Học sâu là một nhánh của ML nhưng tập trung vào việc sử dụng các **Mạng thần kinh nhân tạo (Artificial Neural Networks)** với nhiều lớp ẩn (hidden layers).

**Sự khác biệt lớn:** DL có khả năng **Representation Learning** (Học biểu diễn). Thay vì trích xuất đặc trưng thủ công, các lớp trong mạng thần kinh sẽ tự động học cách lấy ra những thông tin quan trọng nhất từ dữ liệu thô.

**Các kiến trúc quan trọng đã sử dụng:**
* **LSTM (Long Short-Term Memory):** Một dạng RNN đặc biệt giúp xử lý dữ liệu chuỗi thời gian mà không bị mất mát thông tin quá xa trong quá khứ, dùng để dự đoán request tải hệ thống.
* **Embeddings & Transformers:** Được sử dụng trong dự án **RAG-powered AI Chatbot** để biến văn bản thành các vector toán học trong không gian nhiều chiều.
* **Cơ chế lan truyền ngược (Backpropagation):** Là thuật toán giúp máy tính tự điều chỉnh hàng triệu tham số bên trong dựa trên lỗi mà nó mắc phải ở đầu ra.

---

## 3. Reinforcement Learning (RL) - Học tăng cường

RL là một hướng tiếp cận hoàn toàn khác, nơi một **Agent** học cách đưa ra quyết định bằng cách tương tác với một **Môi trường**.

**Khung toán học MDP (Markov Decision Process):** Bao gồm bộ 5 tham số $\{S, A, P, R, \gamma\}$.
* **State ($s$):** Trạng thái hiện tại của môi trường (ví dụ: ảnh từ camera xe đua).
* **Action ($a$):** Hành động agent thực hiện (lái trái, phải, đạp ga).
* **Reward ($r$):** Phản hồi từ môi trường (xe chạy đúng đường được cộng điểm, đâm vào cỏ bị trừ điểm).

**Mục tiêu:** Tìm ra một **Policy ($\pi$)** — tức là một "chiến lược" — sao cho tổng phần thưởng tích lũy (Return $G_t$) là lớn nhất:
$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots$$

**Dự án tiêu biểu:** **Car-Racing Autonomous Agent**, nơi huấn luyện xe tự lái thông qua việc thử và sai (trial and error).