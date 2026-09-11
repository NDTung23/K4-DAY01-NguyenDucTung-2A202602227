# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python 3.13.15 / PyTorch 2.11.0+cpu / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.



## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
`class_id=468, class_name="cab", rank=1, score=0.510915, taxonomy_name="ImageNet-1K"`
- Record này mô tả toàn ảnh như thế nào?
Đây là nhãn cho toàn bộ bức ảnh "traffic", không phải cho một vật thể riêng lẻ nào. Mô hình cho
rằng cả bức ảnh giống với khái niệm "cab" (xe taxi) nhất, với độ tin cậy khoảng 51%.
- Ai định nghĩa class list mà checkpoint có thể dự đoán?
Không phải mô hình tự nghĩ ra. Danh sách 1000 lớp này đến từ bộ dữ liệu ImageNet-1K, do con
người xây dựng và gán nhãn từ trước khi huấn luyện mô hình.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
Vì tên lớp (ví dụ "cab") có thể trùng tên hoặc hiểu khác nhau giữa các bộ dữ liệu khác nhau. Giữ
cả `class_id` và `taxonomy_name` giúp xác định chính xác nhãn này đến từ định nghĩa nào, tránh
nhầm lẫn khi so sánh dữ liệu giữa các dự án khác nhau.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
Guideline cần nói rõ: khi ảnh có nhiều đối tượng khác nhau (ví dụ ảnh giao thông có cả ô tô, xe
máy, người đi bộ), người gán nhãn phải chọn nhãn theo chủ thể chính/nổi bật nhất hay theo toàn
cảnh chung, để tránh mỗi người hiểu và gán nhãn khác nhau.
- Vì sao model score không phải ground truth?  
Vì score chỉ thể hiện mức độ "tự tin" của mô hình dựa trên những gì nó đã học được, không đảm  
bảo đúng với thực tế. Ground truth phải do con người xem xét và xác nhận theo guideline của dự  
án, không thể lấy trực tiếp từ con số score này. 



## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
`class_name="person", score=0.912624, bbox_xyxy=[385.33, 69.24, 498.92, 348.92], bbox_width=113.58, bbox_height=279.68`
- Diễn giải vị trí box bằng lời:
Hộp này nằm ở phía bên phải của ảnh, cách mép trái khoảng 385px và cách mép trên khoảng 69px
(gốc tọa độ (0,0) ở góc trên bên trái ảnh). Hộp rộng khoảng 114px, cao khoảng 280px — là một
hộp cao và hẹp, phù hợp với hình dáng một người đứng.
- So sánh số prediction ở hai threshold:
*(Điền số liệu từ output của ô "sample_id = kitchen" trong phần phát hiện vật thể — notebook in
ra số vật thể phát hiện được ở threshold 0.20, 0.35 và 0.60.)*
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
Ngưỡng càng thấp (0.20) thì mô hình giữ lại càng nhiều box, kể cả những box có độ tin cậy thấp
→ bao phủ nhiều vật thể hơn nhưng cũng nhiều box sai/nhiễu hơn, khiến reviewer phải xem và loại
bỏ nhiều hơn. Ngưỡng càng cao (0.60) thì số box giảm, ít nhiễu hơn nhưng có nguy cơ bỏ sót vật
thể thật (như vật thể nhỏ hoặc bị che khuất một phần).
- Đề xuất một quy tắc box chặt:
Hộp giới hạn nên ôm sát vật thể ở cả 4 cạnh, không chừa khoảng trống nền xung quanh, và không
cắt mất bất kỳ phần nào của vật thể còn nhìn thấy được trong ảnh.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?  
Guideline cần quy định rõ: nếu vật thể bị che khuất một phần, có nên chỉ khoanh phần nhìn thấy  
được hay ước lượng cả phần bị che? Nếu vật thể bị cắt bởi mép ảnh, hộp có nên dừng đúng tại mép  
ảnh hay không? Trường hợp không rõ ràng, annotator nên báo cáo (escalate) cho người phụ trách  
guideline quyết định thay vì tự đoán.



## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
`instance_id="kitchen-001", class_name="person", score=0.899318`, polygon gồm 348 điểm; vài
điểm đầu: `[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]`
- Polygon bổ sung chi tiết gì so với box?
Bounding box chỉ là một hình chữ nhật bao quanh vật thể, gồm cả nền và khoảng trống xung quanh.
Polygon vẽ theo đúng đường viền/hình dạng thật của vật thể (ở đây là 348 điểm nối lại thành
đường viền cơ thể người), nên mô tả chính xác vùng ảnh nào thuộc về vật thể, vùng nào là nền.
- `instance_id` dùng để làm gì và không phải loại ID nào?
`instance_id` chỉ dùng để phân biệt các đối tượng khác nhau trong CÙNG MỘT ảnh (ví dụ
"kitchen-001", "kitchen-002"...). Nó không phải class ID (không cho biết vật thể thuộc lớp gì)
và cũng không phải tracking ID (không dùng để theo dõi cùng một vật thể qua nhiều khung hình/video).
- Đề xuất một quy tắc biên mask:
Đường viền mask nên bám sát rìa thật của vật thể, không lấn vào nền và không bỏ sót phần vật thể
còn nhìn thấy được, kể cả những chi tiết nhỏ ở mép (như ngón tay, tóc).
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?  
Guideline cần quy định: khi ranh giới giữa 2 vật thể tiếp xúc nhau bị mờ hoặc không rõ (ví dụ  
người đứng sát bàn), annotator có nên tách chính xác theo ước lượng hay đánh dấu để escalate  
cho người review quyết định? Cần thống nhất trước để mọi người gán nhãn nhất quán. 



## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`


| Tác vụ                | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --------------------- | ----------------------------- | --------------------------------- | ----------------- | ---------------- |
| Phân loại ảnh         |                               |                                   |                   |                  |
| Phát hiện vật thể     |                               |                                   |                   |                  |
| Instance segmentation |                               |                                   |                   |                  |




## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:



## 6. Danh sách bằng chứng

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không coi score/confidence là ground truth.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.