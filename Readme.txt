Thông tin các file/folder có trong project:
- Folder:
    + annotation: gồm các file .txt chứa face landmarks của các hình ban đầu dùng training
    (cấu trúc: dòng đầu tiên là tên hình và các dòng còn lại là landmarks).
    + annotation_test: gồm các file .txt chứa face landmarks của các hình ban đầu dùng để test
    (cấu trúc: dòng đầu tiên là tên hình và các dòng còn lại là landmarks).
    + checkpoint: gồm các file .pth chứa các tham số đã được lưu lại trong quá trình train.
    + face: chứa các hình ảnh sau quá trình preprocessing. Mỗi hình là một khuôn mặt.
    + images: chứa tất cả các hình ảnh dùng để train, test và validate.
    + new_annotation_test: gồm các file .txt chứa landmarks mới của các hình trong folder face dùng để test
    (cấu trúc: dòng đầu tiên là tên hình và các dòng còn lại là landmarks).
    + new_annotation_train: gồm các file .txt chứa landmarks mới của các hình trong folder face dùng để train
    (cấu trúc: dòng đầu tiên là tên hình và các dòng còn lại là landmarks).
    + new_annotation_val: gồm các file .txt chứa landmarks mới của các hình trong folder face dùng để validate
    (cấu trúc: dòng đầu tiên là tên hình và các dòng còn lại là landmarks).
- File:
    + dataset.py: file khai báo dataset.
    + demo.jpg: hình ảnh dùng để test trong file demo.py.
    + demo.py: file chứa các bước thực hiện khi sử dụng pre-train model để nhận diện face landmarks.
    + Figure_2.png: file ảnh kết quả sau khi test.
    + list_test.txt: chứa đường dẫn đến file .txt trong foler annotation_test dùng để test.
    + list_val.txt: chứa đường dẫn đến file .txt trong foler annotation_test dùng để validate.
    + list.txt: chứa đường dẫn đến file .txt trong foler annotation dùng để train.
    + loss.py: khai báo hàm loss L2.
    + model.py: khai báo model PFLD.
    + new_list_test.txt: chứa đường dẫn đến file .txt trong foler new_annotation_test.
    + new_list_train.txt: chứa đường dẫn đến file .txt trong foler new_annotation_train.
    + new_list_val.txt: chứa đường dẫn đến file .txt trong foler new_annotation_val.
    + preprocess.py: dùng để chuẩn bị dữ liệu là các khuôn mặt và landmarks tương ứng.
    + test.py: dùng để test pre-train model và đánh giá error.
    + test1pic.py: dùng để test và plot để kiểm tra tính chính xác.
    + train_result.png: biểu đồ thể hiện loss ở mỗi epoch khi train.
    + train.py: dùng để train model và lưu checkpoint.
    + utils.py: chứa các hàm hỗ trợ.

- Thứ tự chạy các file để thực hiện train:
    + Cài đặt file "requirements.txt"
    + Run file "preprocess.py" -> "train.py"

- Thứ tự chạy các file để thực hiện test:
    + Cài đặt file "requirements.txt" (nếu trước đó chưa cài)
    + Run file "preprocess.py" (nếu trước đó chưa run) -> "test.py"
