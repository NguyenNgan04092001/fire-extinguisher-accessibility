# Đề tài nghiên cứu

## Tên đề tài

**Автоматическая оценка доступности огнетушителей по изображениям на основе совместного анализа визуального перекрытия и зоны подхода**

**Tự động đánh giá khả năng tiếp cận bình chữa cháy từ hình ảnh dựa trên phân tích đồng thời mức độ che khuất và vùng tiếp cận**

## Mô tả ngắn

Nghiên cứu tập trung xây dựng một phương pháp tự động đánh giá khả năng tiếp cận bình chữa cháy từ ảnh RGB trong môi trường trong nhà bằng các kỹ thuật thị giác máy tính.

Khác với các phương pháp chỉ phát hiện sự xuất hiện hoặc vị trí của bình chữa cháy, nghiên cứu xem xét thêm các vật cản và mối quan hệ không gian giữa chúng với bình chữa cháy.

Phương pháp đề xuất phân tích đồng thời:

- mức độ bình chữa cháy bị che khuất;
- mức độ vùng tiếp cận / thao tác lấy bình bị vật cản chiếm dụng.

## Ý tưởng cốt lõi

**Khả năng nhìn thấy không đồng nghĩa với khả năng tiếp cận.**

Một bình chữa cháy có thể được nhìn thấy hoàn toàn trong ảnh nhưng vẫn khó lấy nếu có vật thể hoặc cấu trúc cản trở thao tác tiếp cận, cầm, nhấc hoặc rút bình khỏi vị trí đặt.

Ngược lại, một vật thể nằm gần bình hoặc một cabinet chứa bình không tự động có nghĩa là bình không tiếp cận được.

## Định nghĩa làm việc của vùng tiếp cận

Trong phiên bản hiện tại, "vùng tiếp cận" không chỉ là vùng sàn phía trước bình. Nó được hiểu theo nghĩa chức năng hơn, bao gồm không gian cần thiết để:

1. tiếp cận vị trí bình;
2. đưa tay tới phần tay cầm;
3. cầm / thao tác với bình;
4. nhấc hoặc rút bình ra khỏi giá, cabinet hoặc vị trí lắp đặt.

Do đó, hệ thống có thể cần xét:

- vật cản phía trước;
- vật cản bên hông / ở góc thao tác;
- vật cản quanh tay cầm;
- giới hạn không gian phía trên hoặc theo hướng rút bình;
- cấu trúc cabinet / holder khi chúng thực sự cản trở việc lấy bình.

Cabinet hoặc cấu trúc cố định **không được mặc định xem là obstacle** nếu chúng không làm hạn chế việc lấy bình.

## Giả thuyết nghiên cứu

Việc kết hợp thông tin về mức che khuất với mức cản trở vùng tiếp cận chức năng sẽ giúp đánh giá khả năng tiếp cận chính xác hơn so với phương pháp chỉ dựa vào mức độ nhìn thấy của bình chữa cháy.

## Tên repository

`fire-extinguisher-accessibility`
