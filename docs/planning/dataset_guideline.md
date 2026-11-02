# Hướng dẫn thiết kế và gán nhãn dataset

> **Trạng thái:** tài liệu này mô tả quy tắc hiện tại sau khi mở rộng khái niệm accessibility. Trước khi train cần chốt một quyết định còn mở: `fire_extinguisher` dùng **modal mask** (phần nhìn thấy) hay **amodal/full mask** (hình dạng đầy đủ kể cả phần bị che). Không được trộn hai định nghĩa trong cùng một class. Xem `docs/repository_audit.md`.

## 1. Khái niệm accessibility

Trong đề tài này, **nhìn thấy bình chữa cháy không đồng nghĩa với lấy được bình chữa cháy**.

Accessibility cần phản ánh khả năng:

1. tiếp cận vị trí bình;
2. đưa tay tới vùng cầm/nắm;
3. thao tác quanh tay cầm và hai bên bình;
4. nhấc/rút bình ra khỏi giá, hộp hoặc vị trí lắp đặt.

Do đó **functional access / extraction zone** không chỉ là vùng sàn ngay phía trước bình. Nó có thể gồm:

- vùng phía trước;
- vùng bên hông cần cho thao tác tay;
- vùng quanh tay cầm;
- khoảng trống phía trên để nhấc/rút bình;
- phần không gian bị cabinet/holder giới hạn nếu cấu trúc đó thực sự cản thao tác lấy bình.

Một cabinet hoặc giá đỡ **không tự động được coi là obstacle**.

---

## 2. Ba mức accessibility dự kiến

Các mức này là nhãn đánh giá cuối của hệ thống, không phải class segmentation YOLO.

### Accessible

- có thể tiếp cận và lấy bình trực tiếp;
- vật ở gần nhưng không cản đáng kể;
- không cần di chuyển một vật cản khác trước khi lấy bình.

### Partially blocked

- vẫn có thể lấy bình nhưng thao tác bị hạn chế;
- có thể cần né vật, đổi góc tay hoặc thực hiện thao tác bổ sung nhỏ;
- vùng trước/bên/tay cầm/phía trên bị chiếm một phần.

### Severely blocked / not readily accessible

- đường tiếp cận hoặc thao tác lấy bình bị cản rõ rệt;
- có khả năng phải di chuyển vật khác trước;
- hoặc không đủ không gian cần thiết để cầm, nhấc hoặc rút bình thuận lợi.

### Nguyên tắc ground truth

Không tạo nhãn accessibility trực tiếp từ công thức `q`, `r` hoặc `q+r`.

Ground truth accessibility phải được xác định độc lập theo tiêu chí thao tác thực tế. Sau đó `q` và các đặc trưng vùng tiếp cận mới được dùng để dự đoán/giải thích nhãn đó.

---

## 3. Segmentation classes

Dataset YOLO hiện chỉ có:

```text
0 fire_extinguisher
1 obstacle
```

### Class 0 — `fire_extinguisher`

**Quy tắc modal/amodal phải được chốt trước khi train.**

Hai lựa chọn:

- **modal:** chỉ polygon phần thực sự nhìn thấy;
- **amodal/full:** polygon toàn bộ hình dạng ước lượng, kể cả phần bị obstacle che.

Khuyến nghị cho baseline YOLO tiêu chuẩn là **modal segmentation**, còn full/reference mask của synthetic nên lưu riêng nếu cần tính ground-truth occlusion.

### Class 1 — `obstacle`

Chỉ annotate vật/cấu trúc **thực sự liên quan tới khả năng lấy bình**, ví dụ:

- ghế, vali, thùng, xe đẩy hoặc thiết bị chắn phía trước;
- vật đặt sát bên làm hạn chế thao tác tay;
- vật/cấu trúc phía trên làm thiếu khoảng nhấc/rút;
- cửa hoặc cấu trúc cố định nếu trong ảnh nó thực sự cản tiếp cận/lấy bình.

Không annotate mọi đồ vật trong ảnh chỉ vì nó nằm gần bình.

Không mặc định annotate toàn bộ cabinet/holder thành obstacle.

---

## 4. Scenario synthetic hiện tại

Các tên S1–S3 dưới đây phản ánh **bộ manual composite thực tế hiện có**.

### S1 — nearby / minimal obstruction

Vật được đặt gần bình hoặc lệch khỏi vùng thao tác chính; tác động tới accessibility nhỏ.

Mục đích: hard negative, tránh quy tắc sai `near object = blocked`.

### S2 — access obstruction with extinguisher still relatively visible

Vật nằm trong vùng tiếp cận/thao tác nhưng bình vẫn nhìn thấy tương đối rõ.

Đây là nhóm quan trọng để kiểm tra trường hợp:

```text
visibility tốt nhưng accessibility kém
```

### S3 — access obstruction + visual occlusion

Vật vừa cản vùng lấy bình vừa che một phần bình trong ảnh.

### Các scenario mở rộng sau này

Nếu cần có thể thêm S4/S5..., ví dụ:

- upper / extraction-space obstruction;
- side / handle-space obstruction;
- cabinet/holder constraint;
- các variation về góc camera, ánh sáng và background.

Script augmentation đã được sửa để nhận diện tự động bất kỳ `_S<number>` nào.

---

## 5. Additional real cases

Bộ real bổ sung hiện có 14 ảnh được ghi cố định trong:

```text
metadata/additional_real_split.csv
```

Các ảnh này bổ sung:

- cabinet/reference cases;
- accessible controls;
- door/chair/clutter/equipment obstructions;
- upper/extraction-space obstruction.

Không thay đổi split của các ảnh này sau khi bắt đầu experiment chính thức.

---

## 6. Hard cases cần có trong evaluation

### H1 — visible but blocked

Hai ảnh đều nhìn thấy bình rõ, nhưng chỉ một ảnh có vật thật sự cản việc lấy bình.

### H2 — near vs blocking

Cùng loại vật thể:

- một ảnh vật ở gần nhưng không cản;
- một ảnh vật nằm trong functional access zone.

### H3 — front vs side vs upper obstruction

Mức độ nhìn thấy bình tương tự nhưng vị trí vật cản khác nhau:

- phía trước;
- bên hông/tay cầm;
- phía trên/hướng nhấc bình.

### H4 — cabinet/reference

Bình ở trong cabinet nhưng vẫn accessible so với cabinet/structure thực sự làm hạn chế extraction.

### H5 — perspective

Cùng bố trí vật thể nhưng góc camera khác nhau để kiểm tra độ nhạy với perspective.

---

## 7. Train / validation / test

Nguyên tắc quan trọng nhất là chống data leakage.

Không để các ảnh gần như cùng một scene/setup xuất hiện ở nhiều split khác nhau.

Các synthetic tạo từ cùng một base image phải kế thừa split của base. Bộ synthetic hiện tại được dùng ở train.

Additional real images dùng fixed split manifest.

### Vai trò các split

**Train**
- học tham số segmentation model;
- chỉ train mới được augmentation.

**Validation**
- chọn model/checkpoint;
- điều chỉnh threshold và thiết kế access zone;
- không dùng để báo cáo duy nhất kết quả cuối.

**Test**
- chỉ dùng sau khi phương pháp đã cố định;
- dùng cho kết quả cuối, confusion matrix và comparison giữa các phương pháp/model.

### Caveat hiện tại

Val và test hiện chỉ có 2 ảnh obstacle-positive mỗi split. Đây chỉ đủ cho smoke test, **không đủ cho kết luận khoa học cuối cùng**. Trước paper/defense cần bổ sung thêm các case real độc lập.

---

## 8. Augmentation

Augmentation hiện tại chỉ là **photometric**:

- brightness / contrast;
- gamma;
- HSV nhẹ;
- blur nhẹ;
- Gaussian noise;
- sharpen nhẹ.

Không rotate/crop/flip/resize trong offline augmentation, vì polygon được copy nguyên.

Val/test không augmentation.

1046 ảnh train augmented **không được mô tả như 1046 scene độc lập**.

---

## 9. Validation trước khi train

Luôn chạy:

```bash
python -m scripts.data_prep.build_source_dataset
python -m scripts.data_prep.validate_yolo_dataset data/fire_accessibility_source
python -m scripts.data_prep.augment_train_dataset
python -m scripts.data_prep.validate_yolo_dataset data/fire_accessibility_augmented
```

Expected counts với dataset hiện tại:

```text
source:
  train 182
  val    30
  test   31

augmented:
  train 1046
  val     30
  test    31
```

Nếu counts thay đổi ngoài chủ ý, dừng lại và kiểm tra trước khi train.

---

## 10. Metadata nên lưu khi mở rộng dataset

Ưu tiên các trường:

```text
image_id
source
scene_group_id
scenario_id
accessibility_class
obstacle_type
obstruction_region   # front / side / handle / upper / other
camera_angle
lighting_condition
split
source_url
license
```

`source_url` và `license` đặc biệt quan trọng đối với ảnh lấy từ web nếu sau này dataset được công khai hoặc dùng trong bài báo.
