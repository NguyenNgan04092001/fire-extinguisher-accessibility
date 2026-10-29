# Đề tài nghiên cứu

## Tên đề tài

**Tự động đánh giá khả năng tiếp cận bình chữa cháy từ hình ảnh dựa trên phân tích đồng thời mức độ che khuất và vùng tiếp cận**

## Mô tả ngắn

Nghiên cứu tập trung xây dựng một phương pháp tự động đánh giá khả năng tiếp cận bình chữa cháy từ ảnh RGB trong môi trường trong nhà bằng các kỹ thuật thị giác máy tính.

Khác với các phương pháp chỉ phát hiện sự xuất hiện hoặc vị trí của bình chữa cháy, nghiên cứu này xem xét thêm các vật cản và mối quan hệ không gian giữa chúng với bình chữa cháy.

Phương pháp đề xuất phân tích đồng thời:

- mức độ bình chữa cháy bị che khuất;
- mức độ vùng tiếp cận phía trước bình chữa cháy bị vật cản chiếm dụng.

Giả thuyết chính của nghiên cứu là việc kết hợp hai loại thông tin này sẽ giúp đánh giá khả năng tiếp cận chính xác hơn so với việc chỉ đánh giá mức độ nhìn thấy của bình chữa cháy.

## Ý tưởng cốt lõi

**Khả năng nhìn thấy không đồng nghĩa với khả năng tiếp cận.**

Một bình chữa cháy có thể được nhìn thấy hoàn toàn trong ảnh nhưng vẫn khó tiếp cận nếu có các vật thể như thùng carton, ghế hoặc thiết bị khác chắn phía trước.

Ngược lại, một phần nhỏ của bình chữa cháy có thể bị che khuất nhưng vẫn không ảnh hưởng đáng kể đến khả năng tiếp cận.

## Tên repository

`fire-extinguisher-accessibility`