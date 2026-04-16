FULL REFACTOR SPECIFICATION: EEG EMOTION RECOGNITION SYSTEM
1. TỔNG QUAN (OVERVIEW)
Mục tiêu: Tái cấu trúc Dashboard Streamlit để hỗ trợ xử lý song song (Parallel Processing) cho hai đặc tính Arousal và Valence, đồng thời xây dựng một Virtual File Manager (Thanh quản lý thư mục ảo) ở bên phải để quản lý toàn bộ luồng dữ liệu.

Phạm vi chỉnh sửa: Mainstream/app/main.py và các hàm hỗ trợ trong Mainstream/src/.

Nguyên tắc: Các bước MRMR, Train, Predict không còn ràng buộc cứng theo thứ tự mà dựa trên sự tồn tại của file trong File Manager.

2. KIẾN TRÚC DỮ LIỆU (SESSION STATE)
Mọi dữ liệu phải được quản lý tập trung trong st.session_state.file_manager. Agent cần khởi tạo cấu trúc này ngay khi ứng dụng bắt đầu:

Python
if "file_manager" not in st.session_state:
    st.session_state.file_manager = {
        "raw_data": [],          # Danh sách đối tượng/tên file .dat đã load
        "processed_data": [],    # Dữ liệu sau FFT (dùng cho Train và Predict)
        "mrmr_selection": {
            "arousal": None,     # Đối tượng DataFrame hoặc đường dẫn file Excel
            "valence": None
        },
        "models": {
            "arousal": None,     # Đối tượng model đã train hoặc path .pth
            "valence": None
        }
    }
3. THIẾT KẾ GIAO DIỆN (UI LAYOUT)
Tất cả các trang chức năng (trừ Home) phải sử dụng cấu trúc 2 cột để giả lập thanh quản lý bên phải:

Python
col_main, col_manager = st.columns([3.5, 1.2], gap="medium")
Cột bên phải (Virtual File Manager):
Phải hiển thị các Expander hoặc Container tương ứng:

📂 Data: Hiển thị danh sách file .dat đã nạp.

⚡ Processed Data: Hiển thị danh sách dữ liệu FFT sẵn sàng để Train/Predict.

🔬 MRMR Selection:

Mục Arousal: Trạng thái (Có file/Trống) + Nút "Upload Excel" để nạp file thủ công.

Mục Valence: Trạng thái (Có file/Trống) + Nút "Upload Excel" để nạp file thủ công.

🎓 Model:

Mục Arousal: Trạng thái Model + Nút "Upload Model (.pth)".

Mục Valence: Trạng thái Model + Nút "Upload Model (.pth)".

4. LOGIC XỬ LÝ CHI TIẾT (FUNCTIONAL LOGIC)
A. Load Data & Preprocess
Gỡ bỏ việc chọn nhãn (Arousal/Valence) ở bước này.

Sau khi FFT, kết quả phải được đẩy vào file_manager["processed_data"].

B. MRMR Channel Selection
Giao diện: Cho phép chọn checkbox: [ ] Arousal và [ ] Valence.

Xử lý:

Nếu chọn cả hai: Sử dụng concurrent.futures.ThreadPoolExecutor để chạy run_mrmr_global_selection cho cả hai nhãn cùng lúc.

Tận dụng CPU đa nhân để tính toán song song.

Output: Tự động lưu vào file_manager["mrmr_selection"] và cho phép tải xuống 2 file Excel riêng biệt.

C. Train Model (Quan trọng)
Cơ chế nạp file: Không bắt buộc chạy MRMR trước. Nếu người dùng upload file Excel vào cột bên phải, Model sẽ dùng file đó.

Điều kiện: - Nếu train Arousal: Bắt buộc phải có file trong mrmr_selection["arousal"].

Nếu train Valence: Bắt buộc phải có file trong mrmr_selection["valence"].

Dữ liệu huấn luyện: Tự động lấy toàn bộ danh sách trong processed_data để train.

Xử lý song song: Nếu chọn train cả hai, phải khởi tạo 2 tiến trình huấn luyện BiLSTM song song để tận dụng GPU.

Output: Sau khi train xong, tự động cập nhật Model vào file_manager["models"].

D. Predict (Dự đoán)
Giao diện: Chọn 1 dữ liệu từ processed_data. Chọn checkbox nhãn muốn dự đoán.

Điều kiện: Để dự đoán nhãn nào, phải có đủ cả Model và file MRMR của nhãn đó trong File Manager.

Hiển thị:

Nếu dự đoán cả hai: Chia đôi màn hình bên dưới (st.columns(2)).

Cột trái: Kết quả Arousal. Cột phải: Kết quả Valence.

Quá trình dự đoán (Inference) của 2 model phải chạy song song.

5. YÊU CẦU KỸ THUẬT (TECHNICAL REQUIREMENTS)
Parallel Library: Sử dụng concurrent.futures.ThreadPoolExecutor cho các tác vụ I/O bound và tính toán logic.

UI Responsiveness: Sử dụng st.status hoặc st.spinner để thông báo trạng thái của các tiến trình đang chạy song song.

Error Handling: Phải có thông báo lỗi cụ thể nếu thiếu file MRMR hoặc Model trong thanh quản lý trước khi chạy Train/Predict.

No Hardcoding: Tuyệt đối không hardcode đường dẫn file. Mọi thứ phải thông qua st.session_state.

6. PROMPT DÀNH CHO AGENT (AGENT INSTRUCTION)
"Dựa trên file đặc tả refactor_eeg_specification.md, hãy thực hiện các thay đổi sau trong thư mục Mainstream/:

Thiết lập lại _init_state với cấu trúc file_manager mới.

Chỉnh sửa Layout của tất cả các trang để hiển thị Thanh quản lý thư mục ảo ở bên phải.

Triển khai logic chạy song song bằng ThreadPoolExecutor trong các hàm MRMR, Training và Predict.

Đảm bảo các nút Upload ở thanh bên phải hoạt động và cập nhật trực tiếp vào Session State."