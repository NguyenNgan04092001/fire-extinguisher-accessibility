# Methodology Specification: Fire Extinguisher Accessibility

## 1. Research objective

Mục tiêu của hệ thống không chỉ là phát hiện sự hiện diện của bình chữa cháy mà là đánh giá:

> **Bình chữa cháy có thể được tiếp cận, nắm và lấy ra để sử dụng hay không.**

Khả năng tiếp cận được mô hình hóa bằng hai yếu tố độc lập:

- **q — visual occlusion:** mức độ bình chữa cháy bị che khuất trong ảnh.
- **r — functional access obstruction:** mức độ không gian cần thiết để tiếp cận, nắm và lấy bình bị vật thể cản trở.

Giả thuyết chính:

> Việc kết hợp q và r sẽ đánh giá khả năng tiếp cận chính xác hơn so với chỉ sử dụng q hoặc chỉ sử dụng r.

---

## 2. Final accessibility classes

Đầu ra cuối cùng gồm ba mức:

### 0 — accessible

Bình có thể được tiếp cận, nắm và lấy ra trực tiếp mà không cần di chuyển vật cản đáng kể.

### 1 — partially_blocked

Bình vẫn có thể lấy được nhưng việc tiếp cận hoặc thao tác bị hạn chế. Người sử dụng có thể cần thay đổi tư thế, tiếp cận từ hướng khác hoặc thực hiện thêm thao tác nhỏ.

### 2 — severely_blocked

Vật cản gây trở ngại đáng kể cho việc tiếp cận, nắm hoặc lấy bình. Việc sử dụng bình đòi hỏi phải di chuyển vật cản trước hoặc rất khó thực hiện trực tiếp.

Các nhãn này phải được con người đánh giá độc lập dựa trên khả năng sử dụng thực tế.

**Không được tạo ground truth accessibility trực tiếp từ q hoặc r.**

---

## 3. Visual occlusion score q

### 3.1. Ý nghĩa

`q` biểu diễn tỷ lệ phần hình dạng của bình chữa cháy bị vật khác che khuất.

Giá trị:

\[
0 \leq q \leq 1
\]

Trong đó:

- `q ≈ 0`: bình gần như hoàn toàn nhìn thấy.
- `q → 1`: phần lớn bình bị che.

### 3.2. Synthetic images

Đối với ảnh synthetic, hình dạng đầy đủ của bình trước khi chèn obstacle đã được biết.

Ký hiệu:

- \(M_f^{full}\): full/amodal mask của bình.
- \(M_o\): mask của vật che.

Khi đó:

\[
q =
\frac{|M_f^{full}\cap M_o|}
{|M_f^{full}|}
\]

Đây là nguồn ground truth q đáng tin cậy nhất.

### 3.3. Real images

Không được tùy ý tưởng tượng phần bình bị che nếu hình dạng đầy đủ không thể suy luận đáng tin cậy.

Đối với ảnh real:

- nếu hình dạng đầy đủ có thể suy luận rõ ràng → có thể tạo amodal reference;
- nếu phần bị che quá lớn hoặc không chắc chắn → đánh dấu q là `unreliable`;
- ảnh `q_unreliable` không được dùng để đánh giá định lượng chính xác q.

Ảnh đó vẫn có thể được sử dụng để đánh giá accessibility và r.

---

## 4. Functional access obstruction score r

### 4.1. Ý nghĩa

`r` không biểu diễn việc một vật thể đơn thuần xuất hiện gần bình.

`r` biểu diễn:

> **Mức độ vật thể chiếm không gian cần thiết để tiếp cận, nắm và lấy bình chữa cháy.**

Giá trị:

\[
0 \leq r \leq 1
\]

Trong đó:

- `r ≈ 0`: không gian tiếp cận gần như thông thoáng.
- `r → 1`: vùng cần thiết để thao tác/lấy bình bị cản mạnh.

---

## 5. Potential obstacle

Class YOLO hiện tại có ID:

```text
1: obstacle
```

Về mặt phương pháp, class này được hiểu là:

> **potential_obstacle**

Nó biểu diễn vật thể có khả năng gây cản trở, chứ không khẳng định bản thân vật đó đang làm bình không thể tiếp cận.

Ví dụ:

- chair
- suitcase
- table
- bench
- trash bin
- plant pot
- structural object
- other physical objects near the extinguisher

Một vật chỉ trở thành **functional obstruction** khi nó chiếm vùng cần thiết để tiếp cận hoặc lấy bình.

Ví dụ:

```text
🔥                         🪑
```

Ghế ở xa:

\[
r \approx 0
\]

Trong khi:

```text
🔥
    🪑
```

ghế nằm trong vùng thao tác:

\[
r > 0
\]

Do đó:

> **Object identity ≠ access obstruction.**

Quan hệ không gian giữa vật thể và bình mới quyết định r.

---

## 6. Functional access region

Do nghiên cứu sử dụng ảnh RGB 2D, hệ thống không tuyên bố đo chính xác khoảng trống vật lý 3D.

Thay vào đó, một **2D functional access region** được xây dựng quanh mask của bình.

Ký hiệu:

- \(M_f\): mask của fire extinguisher.
- \(s\): kích thước đặc trưng của bình trong ảnh, lấy từ kích thước bounding box.

Tạo hai vùng:

### Z1 — manipulation/extraction zone

Vùng sát bình, đại diện cho không gian cần để:

- đưa tay tới bình;
- nắm tay cầm hoặc thân bình;
- thao tác và bắt đầu lấy bình ra.

\[
Z_1 = Dilate(M_f,d_1)-M_f
\]

Thiết lập ban đầu:

\[
d_1=0.25s
\]

### Z2 — approach clearance zone

Vùng rộng hơn, đại diện cho khoảng không gian hỗ trợ tiếp cận và di chuyển bình.

\[
Z_2= Dilate(M_f,d_2)-Dilate(M_f,d_1)
\]

Thiết lập ban đầu:

\[
d_2=0.60s
\]

Các hệ số này là **initial parameters**, không phải tiêu chuẩn vật lý cố định.

Chúng chỉ được điều chỉnh trên training/validation data, không dựa trên test set.

---

## 7. Calculation of r

Gọi \(M_o\) là union của các mask potential obstacle.

Tỷ lệ cản trở trong vùng gần:

\[
r_1=
\frac{|M_o\cap Z_1|}
{|Z_1|}
\]

Tỷ lệ cản trở trong vùng rộng:

\[
r_2=
\frac{|M_o\cap Z_2|}
{|Z_2|}
\]

Functional obstruction score ban đầu:

\[
\boxed{
r=0.7r_1+0.3r_2
}
\]

Z1 có trọng số lớn hơn vì vật nằm sát bình có khả năng ảnh hưởng trực tiếp đến thao tác lấy bình hơn vật nằm xa.

Các trọng số phải được kiểm tra bằng sensitivity analysis trên validation set.

---

## 8. Dataset scenarios

Các scenario không phải accessibility classes.

Chúng được dùng để đảm bảo dataset bao phủ đầy đủ các trường hợp cần nghiên cứu.

### S0 — Clear

- q thấp
- r thấp

Bình nhìn rõ và dễ tiếp cận.

### S1 — Occlusion only

- q cao
- r thấp

Bình bị che về mặt thị giác nhưng không gian tiếp cận vẫn tương đối thông thoáng.

### S2 — Access obstruction only

- q thấp
- r cao

Bình nhìn rõ nhưng vật thể cản việc tiếp cận hoặc lấy bình.

Đây là scenario đặc biệt quan trọng vì nó chứng minh:

> **visibility does not imply accessibility.**

### S3 — Occlusion + access obstruction

- q cao
- r cao

Bình vừa bị che khuất vừa bị cản trở tiếp cận.

### S4 — Hard negative

Có vật thể ở gần bình nhưng không thực sự cản vùng tiếp cận.

Ví dụ:

```text
🔥                    🪑
```

S4 giúp hệ thống học sự khác nhau giữa:

> “có vật thể gần bình”

và

> “vật thể thực sự cản việc lấy bình”.

---

## 9. Experimental comparison

Ba phương pháp chính:

### M1 — q only

Input:

\[
q
\]

→ predict accessibility class.

### M2 — r only

Input:

\[
r
\]

→ predict accessibility class.

### M3 — q + r

Input:

\[
(q,r)
\]

→ predict accessibility class.

M3 là phương pháp đề xuất chính.

Giả thuyết:

\[
Performance(q+r)

> Performance(q)
> \]

và

\[
Performance(q+r)

> Performance(r)
> \]

Một detection-only baseline có thể được báo cáo như baseline phụ nhưng không phải comparison chính.

---

## 10. Evaluation metrics

### Segmentation

Báo cáo riêng cho từng class:

- Precision
- Recall
- mAP@0.50
- mAP@0.50:0.95

Không chỉ báo cáo overall metric vì class imbalance có thể che giấu hiệu suất thấp của `potential_obstacle`.

### Accessibility classification

Các metric chính:

- Accuracy
- Macro-F1
- Precision per class
- Recall per class
- Confusion matrix

Đặc biệt theo dõi:

> Recall của `severely_blocked`

vì false negative trong trường hợp bình thực sự khó tiếp cận có ý nghĩa lớn hơn về mặt an toàn.

---

## 11. Data split policy

- Train: được sử dụng để huấn luyện.
- Validation: được sử dụng để chọn model, parameter và threshold.
- Test: chỉ sử dụng sau khi phương pháp đã được chốt.

Không được sử dụng test set để:

- lựa chọn hyperparameter;
- chỉnh q/r;
- lựa chọn dilation distance;
- lựa chọn threshold;
- quyết định sửa model.

---

## 12. Current research interpretation

YOLO segmentation không trực tiếp quyết định accessibility.

Vai trò của segmentation là tạo:

1. fire extinguisher mask;
2. potential obstacle masks.

Sau đó spatial analysis mới tính q và r.

Pipeline chính thức:

```text
RGB image
     ↓
Instance segmentation
     ↓
┌─────────────────────────┐
│ fire extinguisher mask  │
│ potential obstacle mask │
└────────────┬────────────┘
             │
       ┌─────┴─────┐
       ↓           ↓
       q     functional access region
                   ↓
                   r
       └─────┬─────┘
             ↓
       accessibility
             ↓
 ┌──────────────────────┐
 │ accessible           │
 │ partially_blocked    │
 │ severely_blocked     │
 └──────────────────────┘
```

---

## 13. Fixed methodological decisions

Các quyết định sau được coi là specification chính thức của project:

1. Bài toán cuối cùng có **3 accessibility classes**.
2. Accessibility được phân tích bằng **hai yếu tố q và r**.
3. `q` biểu diễn visual occlusion.
4. `r` biểu diễn functional access obstruction.
5. Có **5 dataset scenarios S0–S4**, không phải 5 output classes.
6. `obstacle` được hiểu về mặt phương pháp là `potential_obstacle`.
7. Việc một vật có phải obstruction hay không phụ thuộc vào quan hệ không gian với functional access region.
8. Ground truth accessibility phải độc lập với q và r.
9. Ba comparison chính là `q-only`, `r-only`, `q+r`.
10. Test set chỉ được sử dụng sau khi phương pháp và tham số đã được chốt.
11. Với real images, không ép tạo amodal q ground truth khi phần bị che không thể suy luận đáng tin cậy.
12. Functional access region là một **2D proxy**, không được mô tả như phép đo chính xác khoảng trống 3D.
