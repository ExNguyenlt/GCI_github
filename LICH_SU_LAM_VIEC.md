# Lịch sử làm việc của Claude trong repo GCI_github

Tài liệu ghi lại các đợt làm việc của Claude Code (phiên cloud, nhánh `claude/cool-noether-usxp6b`), dựng lại từ bản ghi hội thoại, lịch sử git và nhật ký thí nghiệm. Giờ ghi theo UTC (giờ Việt Nam = UTC + 7).

## Tổng quan

| Đợt | Thời gian (UTC) | Nội dung chính | Kết quả |
|---|---|---|---|
| 1 | 08/11 12:11–13:12 | Khảo sát repo, cài cơ chế theo dõi `main` | `CLAUDE.md` + hook SessionStart, đã gộp vào `main` |
| 2 | 08/11 13:15–16:50 | Mô hình hóa ban đầu, Optuna + stacking | AUC 0.76273 |
| 3 | 09/11 08:49–11:26 | Thiết lập vòng lặp 3 tác nhân, vòng 1 | AUC 0.76790 |
| 4 | 10/11–12/11 | Vòng 2 (gián đoạn do tạm dừng và hết hạn mức) | AUC 0.77081 |
| 5 | 12/11 18:13–13/11 08:33 | Tự động chạy lại, vòng 3 (vòng cuối) | AUC 0.77149, vòng lặp dừng |
| 6 | 13/11 19:39 | Viết tài liệu này | `LICH_SU_LAM_VIEC.md` |

AUC trong tài liệu là OOF AUC, kiểm định chéo 5 phần (StratifiedKFold, shuffle, random_state = 42) trên 171.202 dòng train, không phải điểm của ban tổ chức.

---

## Đợt 1 — Khảo sát repo và cơ chế theo dõi nhánh `main` (08/11)

**Đã làm gì**
- Liệt kê file trong repo; giải thích `github.txt` nằm trên `main` nên Claude chỉ thấy sau khi gộp.
- Cài cơ chế để mỗi lần làm việc Claude đều biết `main` đã thay đổi gì.
- Đổi tên commit thành `000` theo yêu cầu, sau đó force push (được người dùng cho phép) để nhánh chỉ còn một commit `000` dựng trên `origin/main`.

**Làm thế nào**
- `CLAUDE.md`: quy tắc bắt buộc — `git fetch origin main` → báo tên commit/file mới → `git merge origin/main` trước khi sửa → gặp xung đột thì dừng và hỏi; xóa output notebook; không push thẳng lên `main`.
- `.claude/settings.json` + `.claude/hooks/check-main.sh`: hook SessionStart tự fetch và báo thay đổi của `main` (chỉ báo, không gộp, luôn thoát mã 0).

**Kết quả**
- Người dùng đã gộp cơ chế này vào `main`. Lỗi push 403 ban đầu được khắc phục sau khi người dùng kết nối lại GitHub.

---

## Đợt 2 — Mô hình hóa ban đầu (08/11)

**Yêu cầu:** tạo `com/submission.csv` cho bài Home Credit (dự đoán khả năng vỡ nợ, thước đo ROC-AUC), mục tiêu ≥ 0.80, chỉ dùng phương pháp hợp lệ.

**Đã làm gì và làm thế nào**
1. Đọc dữ liệu: `train_part1.csv` + `train_part2.csv` (171.202 dòng, ~8% vỡ nợ), `test.csv` (61.500 dòng), 33 cột hồ sơ vay; tên cột có BOM nên đọc bằng `encoding="utf-8-sig"`.
2. Baseline LightGBM → thêm ~40 đặc trưng (tỷ lệ tiền vay/thu nhập, tổ hợp EXT_SOURCE, …) → kết hợp LightGBM/XGBoost/CatBoost bằng trung bình hạng.
3. Optuna (TPE) tối ưu tham số và stacking bằng hồi quy logistic (theo yêu cầu "chạy optuna và stacking").
4. Đọc 5 tài liệu PDF trong `com/advanced techniques/` (chỉ trả lời, không sửa gì); giải thích ý nghĩa AUC và vì sao dữ liệu là giới hạn trên của điểm.

**Kết quả**

| Bước | AUC |
|---|---|
| LightGBM baseline | 0.7615 |
| + 40 đặc trưng | 0.7615 |
| Kết hợp 3 mô hình | 0.7625 |
| Optuna + stacking | 0.76273 |

- Kiểm tra rò rỉ: AUC của riêng mã khách ≈ 0.503 (không mang thông tin); kiểm tra lệch train/test ≈ 0.512 (hai tập đồng nhất).

---

## Đợt 3 — Vòng lặp 3 tác nhân, vòng 1 (09/11)

**Thiết kế vòng lặp (theo yêu cầu người dùng)**
- Tác nhân 1: nghiên cứu phương pháp tối ưu MỚI. Tác nhân 2: rút bài học từ vòng trước. Hai tác nhân này chạy song song.
- Tác nhân 3: cài đặt theo đánh giá của 1 và 2, chạy và so sánh kết quả.
- Điều kiện dừng (chỉ một trong ba): (1) AUC ≥ 0.80; (2) Tác nhân 1 hết ý tưởng mới VÀ mọi bài học của Tác nhân 2 đã được thực hiện, sau khi chạy nốt vòng cuối; (3) người dùng ra lệnh dừng. Không dừng vì bão hòa hay giới hạn số vòng.
- Bài học được theo dõi bằng bảng "CHECKLIST BÀI HỌC" (mã L<vòng>.<số>, trạng thái CHƯA LÀM / ĐÃ LÀM / KHÔNG THỰC THI ĐƯỢC).

**Vòng 1 — kỹ thuật được giữ**
- Điền khuyết EXT_SOURCE bằng hồi quy.
- Dựng lại lãi suất ngầm từ phương trình trả góp (AMT_CREDIT, AMT_ANNUITY).
- Dựng lại kỳ hạn vay.

**Bị loại:** đặc trưng ngày tháng, trung bình nhãn theo láng giềng KNN, mạng MLP, đưa mô hình vòng 0 vào kết hợp.

**Kết quả:** AUC **0.76790** (LightGBM 0.76696, XGBoost 0.76643, CatBoost 0.76545). Đã cập nhật `com/submission.csv`, commit `000`, push.

---

## Đợt 4 — Vòng 2 (10/11 – 12/11)

**Diễn biến:** người dùng ra lệnh "Tạm dừng mọi thứ" (09/11), rồi cho chạy tiếp (10/11). Các tác nhân nhiều lần dừng vì hết hạn mức 5 giờ và được cho chạy tiếp từ chỗ dừng, không làm lại từ đầu.

**Đã làm gì**
- Tác nhân 1: "hồ sơ quần thể sản phẩm vay" — với mỗi sản phẩm (khóa kỳ hạn × lãi suất dựng lại), tính đặc trưng trung bình của mọi khách cùng sản phẩm trên train + test, không dùng nhãn.
- Tác nhân 2: mã hóa sản phẩm theo nhãn, tính lồng trong từng fold; chỉ ra các nhóm khách mô hình còn yếu (trên 60 tuổi, hưu trí, thiếu 2 điểm EXT).
- Tác nhân 3: cài đặt cả hai cách, thực hiện 10 mục checklist còn tồn, chạy đủ 5 fold cho 3 mô hình, rồi kết hợp.

**Kết quả**
- Giữ: hồ sơ sản phẩm (+0.0028); mã sản phẩm làm biến phân loại cho CatBoost (+0.0040 trên CatBoost); mã hóa theo nhãn trong LightGBM (+0.0003).
- Loại (≈ 0): mô hình riêng cho nhóm yếu, pseudo-label, EBM, láng giềng KNN ở tầng kết hợp, …
- AUC **0.77081** (+0.0029, tăng ở 5/5 fold). Đã cập nhật `com/submission.csv`, commit `000`, push (12/11 18:32).

---

## Đợt 5 — Tự động chạy lại và vòng 3 (12/11 18:13 – 13/11 08:33)

**Tự động chạy lại khi hạn mức hồi**
- Tạo tác vụ định kỳ "Tự tiếp tục vòng lặp 3 tác nhân". Ban đầu kiểm tra mỗi giờ; sau đổi sang chu kỳ 5 giờ theo ý người dùng.
- Lần đầu bỏ lỡ mốc hạn mức hồi lúc 21:30 do ước lượng sai thời điểm bắt đầu chu kỳ; đã căn lại thành 03:40 / 08:40 / 13:40 / 18:40 / 23:40 UTC.
- Đã tắt theo yêu cầu người dùng (13/11 08:31).

**Vòng 3**
- Tác nhân 1: thử 9 hướng mới (OpenFE, tự chưng cất nhãn mềm, phần dư định giá, ràng buộc đơn điệu, …), không hướng nào vượt ngưỡng → **NO_NEW_METHODS**.
- Tác nhân 2: xác nhận không rò rỉ nhãn; OOF hơi lạc quan khoảng +0.00025 do dừng sớm; đề xuất 6 mục cuối.
- Tác nhân 3 (vòng cuối) bị dừng vì hết hạn mức tuần của model, sau khi đã chạy xong các mô hình; phiên chính tự hoàn tất bước kết hợp, ghi submission và nhật ký.

| Mục | Kết quả |
|---|---|
| Hồ sơ sản phẩm × 5 mức tiền vay | Tăng ổn định → giữ |
| CatBoost 5 seed | Tăng nhẹ → giữ |
| Mạng nơ-ron TabM (torch CPU) | Tăng khi kết hợp → giữ |
| LightGBM DART / extra trees / linear tree | Làm đa dạng tầng kết hợp → giữ |
| Bỏ XGBoost | Không đổi → không dùng XGBoost |
| Lọc biến Null Importance | Giảm điểm → loại |
| Phạt lãi suất cao khi dựng kỳ hạn | Không ổn định → loại |

**Kết quả**
- Kết hợp cuối: stacking hồi quy logistic lồng fold trên LightGBM, CatBoost, hồi quy logistic phân khoảng, DART, extra trees, linear tree, TabM.
- AUC **0.77149** (+0.00068 so với vòng 2, tăng ở 5/5 fold); ước lượng trung thực khi không dừng sớm ≈ 0.7713.
- Đã cập nhật `com/submission.csv`, commit `000`, push (13/11 08:33).
- **Vòng lặp dừng theo điều kiện 2.**

---

## Kết luận chung

| Vòng | AUC |
|---|---|
| 0 | 0.76273 |
| 1 | 0.76790 |
| 2 | 0.77081 |
| 3 | **0.77149** |

- Mục tiêu 0.80 chưa đạt. Mọi tác nhân qua các vòng đều ước lượng trần hợp lệ với 33 cột hồ sơ vay khoảng 0.771–0.772; để vượt mức đó cần thêm dữ liệu (lịch sử tín dụng, các khoản vay trước) vốn không có trong bài.
- Mã nguồn và log thí nghiệm nằm trong thư mục tạm của phiên, không đưa vào repo theo yêu cầu "không tạo file nào khác"; chúng sẽ mất khi phiên kết thúc.
