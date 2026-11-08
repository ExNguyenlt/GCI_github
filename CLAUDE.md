# Quy tắc làm việc trong repo GCI_github

## Đồng bộ với nhánh `main` (BẮT BUỘC)
Người dùng có thể đẩy thay đổi trực tiếp lên `main` bất kỳ lúc nào. Vì vậy:

1. **Trước mỗi lần bắt đầu sửa/thêm file** (và khi người dùng nhắn "cập nhật main"):
   - Chạy `git fetch origin main`.
   - Xem thay đổi: `git log --oneline HEAD..origin/main` và `git diff --name-status HEAD...origin/main`.
2. **Báo cáo ngắn cho người dùng**: số commit mới + tên file thêm/sửa/xóa.
   - Chỉ báo TÊN file và commit, **không tự mở đọc nội dung** (tiết kiệm tín dụng) trừ khi người dùng yêu cầu hoặc việc sửa code cần đến.
3. **Gộp**: `git merge origin/main` vào nhánh làm việc, TRƯỚC khi sửa — không gộp giữa chừng.
4. **Gặp xung đột (merge conflict)**: DỪNG lại, báo file nào xung đột và hỏi người dùng giữ bản nào. Không tự chọn.
   - File nhị phân (`.docx`, `.pdf`, `.xlsx`): luôn hỏi người dùng giữ bản nào.
5. **Sau khi gộp**: nếu có code, chạy thử lại để phát hiện lỗi do thay đổi trên main (vd. đổi tên/di chuyển file dữ liệu).

## Quy ước khác
- Notebook `.ipynb`: xóa output (clear output) trước khi commit để giảm xung đột.
- Không thêm file dữ liệu lớn mới vào repo nếu không cần (thư mục `com/` đã ~68 MB).
- Không bao giờ push thẳng lên `main`; làm trên nhánh được chỉ định.

## Cấu trúc repo
- `com/` — dữ liệu & tài liệu competition (Home Credit: train_part1/2.csv, test.csv, sample_submission.csv, mô tả cột .xlsx, tutorial PDF).
- `ses6/`, `ses7/`, `ses8/` — slide + ghi chép các buổi học (ses7 có hướng dẫn bài cuối khóa).
