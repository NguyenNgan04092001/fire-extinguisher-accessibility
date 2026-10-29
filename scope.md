# Phạm vi nghiên cứu và câu hỏi nghiên cứu

## 0.2 — Câu hỏi nghiên cứu

### Câu hỏi nghiên cứu chính

**Việc phân tích đồng thời mức độ che khuất của bình chữa cháy và mức độ cản trở của vùng tiếp cận có giúp nâng cao độ chính xác trong đánh giá khả năng tiếp cận so với các phương pháp chỉ dựa trên phát hiện hoặc mức độ che khuất của bình chữa cháy hay không?**

### Giả thuyết nghiên cứu

**Việc bổ sung thông tin về mức độ cản trở vùng tiếp cận sẽ giúp giảm các trường hợp đánh giá sai khi bình chữa cháy vẫn nhìn thấy rõ nhưng trên thực tế bị vật thể chắn phía trước.**

### Các phương pháp so sánh

Nghiên cứu sẽ kiểm chứng giả thuyết thông qua bốn phương pháp:

- **Phương pháp A** — chỉ phát hiện bình chữa cháy.
- **Phương pháp B** — chỉ sử dụng mức độ che khuất `q`.
- **Phương pháp C** — chỉ sử dụng mức độ cản trở vùng tiếp cận `r`.
- **Phương pháp D** — kết hợp `q + r`.

### Mục tiêu thực nghiệm

Mục tiêu chính là kiểm tra xem phương pháp kết hợp có cải thiện chất lượng phân loại hay không:

```text
Macro-F1(D) > Macro-F1(B)
```

Hoặc ít nhất, phương pháp D cải thiện khả năng phát hiện các trường hợp bị cản trở nghiêm trọng:

```text
Recall_D(severely_blocked) > Recall_B(severely_blocked)
```

---

## 0.3 — Phạm vi MVP

Phiên bản MVP chỉ tập trung vào **ảnh RGB tĩnh**, sử dụng segmentation và phân tích quan hệ không gian.

### Pipeline tổng thể

```text
Ảnh RGB
    ↓
Segmentation
    ↓
Phát hiện bình chữa cháy + vật cản
    ↓
Tính mức độ che khuất q
    ↓
Tạo vùng tiếp cận
    ↓
Tính mức độ cản trở r
    ↓
Đánh giá khả năng tiếp cận
```

### Các thành phần có trong MVP

- Ảnh RGB tĩnh.
- Segmentation bình chữa cháy.
- Segmentation vật cản.
- Tính mức độ che khuất `q`.
- Xác định vùng tiếp cận.
- Tính mức độ cản trở `r`.
- Phân loại khả năng tiếp cận.
- So sánh các phương pháp A/B/C/D.
- Đánh giá bằng:
  - Accuracy
  - Macro-F1
  - Recall
- Phân tích lỗi.

### Các thành phần chưa làm trong MVP

Các nội dung sau **không thuộc phạm vi triển khai chính** và chỉ được xem là hướng mở rộng nếu còn thời gian:

- Video.
- Object tracking.
- Depth estimation.
- Camera RGB-D.
- Phân tích 3D.
- SLAM.
- Đo khoảng cách thực tế.
- VLM/LLM.
- Tự thiết kế backbone neural network mới.

---

## Tóm tắt phạm vi nghiên cứu

Nghiên cứu tập trung vào việc đánh giá khả năng tiếp cận bình chữa cháy từ **một ảnh RGB duy nhất**, dựa trên hai tín hiệu chính:

- `q` — mức độ che khuất của bình chữa cháy.
- `r` — mức độ cản trở của vùng tiếp cận.

Phương pháp đề xuất sẽ kiểm tra xem việc kết hợp hai tín hiệu này có giúp đánh giá khả năng tiếp cận tốt hơn so với các phương pháp chỉ sử dụng thông tin về bình chữa cháy hay không.
