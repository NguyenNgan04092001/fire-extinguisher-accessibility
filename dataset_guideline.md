# Hướng dẫn thiết kế và gán nhãn dataset

## 1.1 — Định nghĩa 3 lớp accessibility

Mỗi ảnh được gán một trong ba nhãn đánh giá khả năng tiếp cận bình chữa cháy.

### Class 0 — Accessible

Bình chữa cháy có thể được tiếp cận trực tiếp mà không cần di chuyển vật thể khác.

Đặc điểm:

- đường tiếp cận phía trước bình tương đối thông thoáng;
- không có vật thể đáng kể chắn trực tiếp phía trước;
- có thể có đồ vật ở gần nhưng không ảnh hưởng rõ ràng đến việc tiếp cận;
- bình có thể bị che một phần nhỏ nhưng việc tiếp cận vẫn không bị ảnh hưởng đáng kể.

Tên class:

`accessible`

---

### Class 1 — Partially blocked

Khả năng tiếp cận bị hạn chế một phần nhưng bình chữa cháy vẫn có thể được tiếp cận.

Đặc điểm:

- có vật thể nằm một phần trong vùng tiếp cận;
- hoặc bình bị che khuất một phần;
- người sử dụng có thể phải né vật cản hoặc thực hiện thêm thao tác nhỏ;
- vật cản chưa làm mất hoàn toàn đường tiếp cận trực tiếp.

Tên class:

`partially_blocked`

---

### Class 2 — Severely blocked

Khả năng tiếp cận bị hạn chế rõ rệt.

Đặc điểm:

- vật thể chắn trực tiếp khu vực phía trước bình;
- người sử dụng có khả năng phải di chuyển vật cản trước khi lấy được bình;
- hoặc bình bị che khuất đáng kể đồng thời khu vực tiếp cận cũng bị chắn;
- đường tiếp cận trực tiếp không còn thông thoáng.

Tên class:

`severely_blocked`

---

### Nguyên tắc gán nhãn accessibility

Nhãn accessibility được xác định độc lập với các giá trị `q` và `r`.

Không sử dụng các quy tắc như:

`q > threshold → blocked`

hoặc:

`q + r > threshold → severely_blocked`

để tạo ground truth.

Các giá trị `q` và `r` chỉ được tính sau khi segmentation hoàn thành và được sử dụng làm đặc trưng của các phương pháp B, C và D.

Nếu có thể, một phần dữ liệu nên được hai người gán nhãn độc lập để kiểm tra mức độ thống nhất của tiêu chí.

---

# 1.2 — Định nghĩa object classes

Trong phiên bản MVP chỉ sử dụng hai lớp segmentation chính.

## Class 0 — fire_extinguisher

Bao gồm phần nhìn thấy của bình chữa cháy.

Tên class:

`fire_extinguisher`

Cần gán mask sát theo phần nhìn thấy thực tế của bình.

---

## Class 1 — obstacle

Bao gồm các vật thể có khả năng ảnh hưởng đến việc tiếp cận bình chữa cháy.

Ví dụ:

- thùng carton;
- ghế;
- túi;
- thùng rác;
- hộp;
- thiết bị;
- đồ dùng;
- các vật thể rắn khác đặt phía trước hoặc gần bình chữa cháy.

Tên class:

`obstacle`

Không gán các thành phần cố định của môi trường như:

- tường;
- sàn;
- trần;
- cửa nếu cửa không đóng vai trò vật cản trong tình huống nghiên cứu;
- các vật thể nền không liên quan.

---

## Metadata loại vật cản

Mặc dù tất cả vật cản đều thuộc class `obstacle`, nên lưu thêm loại vật thể dưới dạng metadata.

Ví dụ:

`box`

`chair`

`bag`

`bin`

`equipment`

`other`

Thông tin này không bắt buộc đối với YOLO nhưng hữu ích cho phân tích lỗi sau này.

---

# 1.3 — Định nghĩa scenario chụp

Dataset cần bao gồm các nhóm tình huống sau.

## S0 — Clear

Bình chữa cháy không bị che và vùng phía trước không có vật cản đáng kể.

Mục đích:

Tạo dữ liệu cho class `accessible`.

---

## S1 — Occlusion only

Bình chữa cháy bị che một phần nhưng vùng tiếp cận vẫn thông thoáng.

Ví dụ:

một vật nằm phía bên cạnh camera khiến một phần bình bị che trong ảnh nhưng không thực sự chắn đường tiếp cận.

Mục đích:

Kiểm tra trường hợp:

`q cao nhưng accessibility vẫn tốt`

---

## S2 — Approach obstruction only

Bình chữa cháy vẫn nhìn thấy rõ nhưng có vật cản phía trước.

Ví dụ:

- thùng carton;
- ghế;
- túi lớn;
- thiết bị.

Mục đích:

Đây là scenario quan trọng nhất để kiểm chứng giả thuyết nghiên cứu:

`q thấp nhưng r cao`

---

## S3 — Occlusion + approach obstruction

Bình vừa bị che khuất vừa có vật cản trong vùng tiếp cận.

Mục đích:

Tạo các trường hợp accessibility bị hạn chế rõ rệt.

---

## S4 — Nearby but not blocking

Có vật thể nằm gần bình chữa cháy nhưng không thực sự chắn đường tiếp cận.

Ví dụ:

một chiếc ghế nằm bên cạnh bình nhưng không nằm phía trước.

Mục đích:

Tạo hard negative và kiểm tra hệ thống có nhầm:

`near object = blocked`

hay không.

---

## S5 — Điều kiện quan sát khác nhau

Lặp lại các scenario trên với:

- góc camera khác nhau;
- khoảng cách khác nhau;
- ánh sáng khác nhau;
- background khác nhau;
- loại vật cản khác nhau;
- kích thước vật cản khác nhau.

Đây không phải một accessibility class mới mà là nhóm variation dùng để kiểm tra robustness.

---

## Dataset pilot

Trước khi thu thập toàn bộ dataset, nên chụp thử khoảng:

`15–20 ảnh / scenario`

để kiểm tra:

- tiêu chí gán nhãn;
- segmentation;
- pipeline;
- tính khả thi của `q`;
- cách tạo vùng tiếp cận.

Sau khi pilot hoạt động mới mở rộng dataset.

Mục tiêu ban đầu hợp lý:

`300–500 ảnh`

sau đó có thể mở rộng nếu cần.

---

# 1.4 — Thiết kế hard pairs

Hard pair là hai ảnh rất giống nhau nhưng có sự khác biệt quan trọng về accessibility.

Hard pairs đặc biệt hữu ích để kiểm tra xem thuật toán có thực sự sử dụng spatial context hay không.

## HP1 — Bình đều nhìn thấy rõ nhưng accessibility khác nhau

Ảnh A:

- bình nhìn thấy rõ;
- phía trước thông thoáng;
- `accessible`.

Ảnh B:

- bình vẫn nhìn thấy rõ tương tự;
- có vật cản trực tiếp phía trước;
- `partially_blocked` hoặc `severely_blocked`.

Đây là hard pair quan trọng nhất.

---

## HP2 — Mức che khuất tương tự nhưng vùng tiếp cận khác nhau

Hai ảnh có mức occlusion gần giống nhau.

Ảnh A:

- vùng tiếp cận thông thoáng.

Ảnh B:

- vùng tiếp cận bị chắn.

Mục tiêu:

kiểm tra xem `r` có bổ sung thông tin mà `q` không thể cung cấp hay không.

---

## HP3 — Vùng tiếp cận tương tự nhưng mức che khuất khác nhau

Hai ảnh có bố trí vật cản phía trước tương tự nhưng bình bị che ở mức khác nhau.

Mục tiêu:

đánh giá đóng góp của `q`.

---

## HP4 — Near vs Blocking

Ảnh A:

vật thể nằm cạnh bình nhưng không nằm trên đường tiếp cận.

Ảnh B:

cùng vật thể được dịch vào phía trước bình.

Mục tiêu:

kiểm tra hệ thống có phân biệt được:

`vật ở gần`

và

`vật thực sự cản trở`

hay không.

---

## HP5 — Perspective pair

Giữ nguyên bố trí vật thể nhưng thay đổi góc camera.

Mục tiêu:

kiểm tra ảnh hưởng của perspective đến `q`, `r` và accessibility prediction.

---

# 1.5 — Quy tắc chia Train / Validation / Test

Không chia ảnh hoàn toàn ngẫu nhiên.

Các ảnh gần giống nhau của cùng một cảnh không được xuất hiện đồng thời trong train và test.

Ví dụ không được phép:

`IMG_001 → train`

`IMG_002 → test`

nếu hai ảnh chỉ khác nhau vài giây hoặc vài centimet camera.

---

## Đơn vị chia dữ liệu

Mỗi nhóm ảnh cần có một `scene_group_id`.

Ví dụ:

`room01_ext01_session01`

Tất cả ảnh thuộc cùng một scene group phải nằm trong cùng một split.

---

## Tỷ lệ đề xuất

Sử dụng:

`70% train`

`15% validation`

`15% test`

Tỷ lệ có thể thay đổi nhẹ để đảm bảo phân bố class hợp lý.

---

## Train set

Dùng để:

- train segmentation model;
- học các tham số của model.

---

## Validation set

Dùng để:

- chọn model;
- điều chỉnh hyperparameter;
- chọn `α`, `β`;
- chọn threshold phân loại;
- kiểm tra các phiên bản approach zone.

Không được dùng test set để thực hiện các bước trên.

---

## Test set

Chỉ sử dụng sau khi phương pháp đã được cố định.

Test set dùng để báo cáo kết quả cuối cùng:

- Accuracy;
- Macro-F1;
- Recall;
- confusion matrix;
- comparison A/B/C/D.

Không điều chỉnh thuật toán dựa trên kết quả test.

---

## Quy tắc chống data leakage

Không để cùng một:

- chuỗi ảnh liên tiếp;
- setup vật cản giống hệt nhau;
- hard pair;
- burst photo;
- video frame gần nhau

xuất hiện ở nhiều split khác nhau.

Một hard pair phải nằm hoàn toàn trong:

`train`

hoặc:

`validation`

hoặc:

`test`

không được tách hai ảnh của cùng một pair sang hai tập khác nhau.

---

## Ưu tiên kiểm tra khả năng tổng quát hóa

Nếu thu thập đủ dữ liệu từ nhiều địa điểm, nên ưu tiên để test set chứa:

- phòng chưa xuất hiện trong train;
- bình chữa cháy chưa xuất hiện trong train;
- background mới;
- bố trí vật cản mới.

Điều này giúp đánh giá khả năng tổng quát hóa thực tế của phương pháp.

---

# Metadata nên lưu cho mỗi ảnh

Mỗi ảnh nên có các trường:

`image_id`

`room_id`

`extinguisher_id`

`session_id`

`scene_group_id`

`scenario_id`

`accessibility_class`

`obstacle_type`

`camera_angle`

`camera_distance`

`lighting_condition`

`hard_pair_id`

`split`

Ví dụ:

`IMG_0042, room02, ext01, session03, room02_ext01_session03, S2, severely_blocked, box, left, medium, indoor, HP01, test`