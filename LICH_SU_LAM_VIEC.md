# Lịch sử làm việc của Claude trong repo GCI_github

Tài liệu ghi lại các đợt làm việc của Claude Code (phiên cloud, nhánh `claude/cool-noether-usxp6b`), dựng lại từ bản ghi hội thoại, lịch sử git và nhật ký thí nghiệm. Giờ ghi theo UTC (giờ Việt Nam = UTC + 7).

# PHẦN I — Tổng quan theo đợt làm việc

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

---

# PHẦN II — CHI TIẾT TỪNG TÁC NHÂN TẠI MỖI VÒNG (bản đầy đủ, không tóm gọn)

Phần này chép **nguyên văn** báo cáo của từng tác nhân (file `loop/round<vòng>_agent<1|2>.md`) và mục do Tác nhân 3 ghi trong sổ chung `loop/history.md`; chỉ hạ cấp tiêu đề để vừa bố cục. Ký hiệu `$S` trong các báo cáo là thư mục tạm của phiên:
`/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad` (không nằm trong repo). Các mã `L<vòng>.<số>` là mã bài học trong checklist của Tác nhân 2; `R<số>` là quy tắc ghép kỹ thuật; `A1 #<số>` là đề xuất số <số> của Tác nhân 1 vòng 1.

## Luật vòng lặp (`loop/rules.md`)
#### Quy trình mỗi vòng
1. Tác nhân 1 (nghiên cứu phương pháp MỚI, chưa có trong history.md) và Tác nhân 2 (rút bài học từ vòng trước) chạy SONG SONG.
2. Tác nhân 3 cài đặt theo đánh giá của 1 và 2, chạy, so sánh kết quả.
3. Nếu chưa thỏa điều kiện dừng → quay lại bước 1.

#### Điều kiện DỪNG — chỉ một trong ba
1. OOF AUC ≥ 0.80.
2. Tác nhân 1 báo `STATUS: NO_NEW_METHODS` VÀ đồng thời mọi bài học/hướng đề xuất của Tác nhân 2 đều đã được Tác nhân 3 thực hiện
   (→ chạy nốt vòng cuối với kiến thức đó rồi dừng).
3. Người dùng ra lệnh dừng.

KHÔNG có điều kiện dừng vì bão hòa hay giới hạn số vòng.

#### Sổ theo dõi bài học (bắt buộc để kiểm điều kiện 2)
- Tác nhân 2: cuối báo cáo mỗi vòng phải có bảng `CHECKLIST BÀI HỌC` — mỗi dòng: mã (L<vòng>.<số>), nội dung hướng/bài học có thể thực thi, trạng thái `CHƯA LÀM` / `ĐÃ LÀM (vòng X, kết quả)` / `KHÔNG THỰC THI ĐƯỢC (lý do)`. Mang theo các dòng chưa làm từ các vòng trước.
- Tác nhân 3: trong mục VÒNG của history.md phải ghi rõ đã thực hiện những mã bài học nào và kết quả.

## Bối cảnh chung và Vòng 0 (trước khi có vòng lặp — do phiên chính thực hiện, không có tác nhân)
Thư mục làm việc chung (scratchpad): `/tmp/claude-0/-home-user-GCI-github/3ba76d1c-216b-5212-94e6-41029488cb7c/scratchpad`
(viết tắt `$S` bên dưới). Sổ này nằm ở `$S/loop/history.md`.

### Bài toán
- Dữ liệu: `/home/user/GCI_github/com/` — `train_part1.csv` + `train_part2.csv` (gộp = 171,202 dòng), `test.csv` (61,500 dòng),
  `sample_submission.csv` (cột `SK_ID_CURR,TARGET`), `HomeCredit_columns_description.xlsx`.
- 33 cột (CHỈ bảng application, KHÔNG có bureau/previous_application/installments...). Tỉ lệ TARGET=1 ≈ 8.0%.
- Lưu ý: file CSV có BOM — bỏ `﻿` khỏi tên cột. pandas 3.x: cột chữ có dtype `str`, dùng `select_dtypes(exclude='number')`.
- Metric: ROC-AUC. Mục tiêu người dùng: AUC ≥ 0.80.
- Tham chiếu: leaderboard mẫu của khóa học hạng 1 = 0.748. Home Credit gốc (Kaggle) top ≈ 0.80 nhưng dùng 7 bảng; chỉ bảng application ≈ 0.76–0.77.
- Máy: 4 CPU, không GPU. Đã cài: lightgbm, xgboost, catboost, scikit-learn, optuna 5, scipy, pandas 3.

### Ràng buộc bắt buộc
- Chỉ phương pháp HỢP LỆ: không rò rỉ nhãn, không dò ngược leaderboard, không dùng nhãn test.
  (Đã kiểm tra: AUC theo SK_ID_CURR = 0.503 → ID không rò rỉ.)
- Kỹ thuật dùng dữ liệu test KHÔNG nhãn (pseudo-label, adversarial validation, thống kê gộp train+test) là hợp lệ nếu không dùng nhãn test.
- Không tạo file trong repo `/home/user/GCI_github` ngoại trừ `com/submission.csv`. Mọi script/kết quả trung gian để trong `$S`.
- Đánh giá bằng 5-fold StratifiedKFold(shuffle=True, random_state=42) để OOF so sánh được với các file sẵn có.
  Stacking/blending phải đánh giá bằng CV lồng nhau (meta-model chấm trên phần chưa thấy).

### File sẵn có trong `$S`
- `fe.py` — hàm `load()` trả `(df, y, n, cats, ids)`: bộ đặc trưng hiện tại (~80 cột). `run.py <lgb|cat|xgb> $S`, `opt.py <model> $S <n_trials>`.
- OOF/test preds (cùng folds rs=42): `oof_{lgb,cat,xgb,lgb_opt,cat_opt,xgb_opt}.npy`, `pt_*.npy`; `y.npy`; `ids.npy` (ID test đúng thứ tự sample_submission).
- Log: `log_cx.txt` (CatBoost/XGB), `log_opt.txt` (Optuna).

### VÒNG 0 (đã làm trước khi có vòng lặp) — kết quả
| # | Thay đổi | OOF AUC | Ghi chú |
|---|---|---|---|
| 0.1 | LightGBM baseline + ~20 FE (EXT mean/std/prod/min/max/weighted, credit/annuity, annuity/income, credit/income, ratios /DAYS_BIRTH, group mean by ORG/OCC), 3 seed × 5 fold | 0.7615 | `hc.py` |
| 0.2 | Thêm ~40 FE nữa (EXT×age, cặp tích EXT, freq encoding, diff so với mean theo ORG/OCC/INCOME_TYPE/EDU, income/median(ORG,OCC), TERM_ROUND, REQ_SUM) + LGB lr 0.01 leaves16 ff0.3 | 0.76152 | KHÔNG tăng so với 0.1 |
| 0.3 | CatBoost (depth6 lr0.03, cat native) | 0.7608 | |
| 0.4 | XGBoost (depth5, colsample0.3, categorical native) | 0.7610 | |
| 0.5 | Rank-average 3 model | 0.76246 | +0.001 |
| 0.6 | Grid trọng số blend | 0.76258 | chỉ +0.0001 → không dùng |
| 0.7 | Optuna (LGB 30, XGB 20, Cat 12 trials; 3 fold, lr cao rồi retrain lr thấp) | LGB 0.76153 / XGB 0.76118 / Cat 0.76063 | ≈ 0; tham số tốt nhất đều cây NÔNG (depth 3–4) + regularization mạnh |
| 0.8 | Rank-avg 6 model (cũ + opt) | 0.76266 | |
| 0.9 | Stacking LogisticRegression trên logit OOF 6 model (CV lồng nhau) | **0.76273** | BẢN ĐANG NỘP. Trọng số XGB ≈ 0 |

Phương pháp ĐÃ dùng (Tác nhân 1 không được đề xuất lại như "mới"):
GBDT (LGB/XGB/Cat), feature ratio/interaction cơ bản, frequency encoding, group-mean diff, rank averaging, weighted blend,
Optuna TPE, stacking LogisticRegression, multi-seed averaging, xử lý DAYS_EMPLOYED=365243.
Từ tài liệu khóa học (đã biết): log transform, label/count/one-hot encoding, OWN_CAR_AGE>60 outlier, fill missing, class_weight balanced, bagging/boosting/stacking lý thuyết.

Quan sát chính vòng 0: 3 thuật toán khác nhau hội tụ ~0.761 → nghi trần thông tin của dữ liệu ~0.76–0.77.

---

---

## VÒNG 1

### Tác nhân 1 — Nghiên cứu phương pháp mới (`loop/round1_agent1.md`)
STATUS: NEW_METHODS_FOUND

#### VÒNG 1 — Tác nhân 1 (Nghiên cứu): phương pháp mới đề xuất

Bối cảnh đã đọc: `loop/history.md` (OOF tốt nhất 0.76273, 3 thuật toán GBDT hội tụ ~0.761) và `fe.py` (~80 cột).
Mọi phương pháp dưới đây CHƯA có trong danh sách "đã dùng". Tất cả hợp lệ: không dùng nhãn test; những gì dùng nhãn train
(KNN-target, target encoding) PHẢI tính lồng trong fold (nested OOF), mô tả cụ thể ở từng mục.

Ghi chú mạng: kaggle.com, medium.com, speakerdeck.com bị chặn trong phiên này, nên một số nguồn chỉ đọc được qua đoạn trích của
công cụ tìm kiếm; mã nguồn giải pháp hạng 2 (KazukiOnodera, GitHub) đọc trực tiếp được.

##### Kiểm tra nhanh trên dữ liệu (không huấn luyện mô hình, script ở `$S/loop/a1/explore*.py`)
- C/A (AMT_CREDIT/AMT_ANNUITY) của Cash loans tập trung thành các CỤM RỜI RẠC (vd. 34.1: 5,757 dòng; 23.5: 4,841; 19.5: 4,467; 20.7; 12.6; 30.8...),
  tức là các "sản phẩm" có (kỳ hạn, lãi suất) cố định. Revolving loans: C/A = 20.000 ở 99.8% dòng → C/A vô nghĩa với revolving.
- Tái dựng lãi suất (chọn kỳ hạn n có mật độ lãi suất cao nhất trên train+test) → trong từng bin C/A, AUC đơn biến của lãi suất ước lượng
  lên tới 0.646 (bin 4/20) và 0.56 (bin 0) → có thông tin MỚI mà C/A (TERM_ROUND) chưa nắm hết.
- Tuổi khi được cấp hộ chiếu (DAYS_ID_PUBLISH − DAYS_BIRTH) dồn cục ở 20.0–20.5 tuổi (16.5k) và 45.0–45.5 tuổi (39.7k) đúng quy định hộ chiếu Nga
  (đổi ở 20 và 45 tuổi). "Độ trễ đổi hộ chiếu" có AUC 0.53–0.546 TRONG từng nhóm tuổi.
- Khách lặp lại: khóa bất biến theo thời gian (DAYS_BIRTH−DAYS_ID_PUBLISH, DAYS_BIRTH−DAYS_REGISTRATION, DAYS_BIRTH−DAYS_EMPLOYED, giới tính)
  cho 2.2% dòng nằm trong nhóm >1 (ví dụ 2 dòng test lệch đúng 113 ngày ở cả DAYS_BIRTH lẫn DAYS_EMPLOYED → cùng một người). Độ phủ nhỏ.

##### Bảng xếp hạng (điểm ≈ mức tăng kỳ vọng × khả năng thành công / chi phí)

| Hạng | Phương pháp | Mức tăng OOF kỳ vọng | Chi phí CPU | Rủi ro chính |
|---|---|---|---|---|
| 1 | Tái dựng kỳ hạn & lãi suất khoản vay (term/rate reconstruction) + "mã sản phẩm" | +0.002 → +0.008 | vài giây | gán nhầm kỳ hạn (chỉ làm nhiễu, trees tự lọc) |
| 2 | KNN-target-mean 500 láng giềng (đặc trưng mạnh của đội vô địch) — nested OOF | +0.001 → +0.004 | 5–15 phút | rò rỉ nếu không lồng fold |
| 3 | Điền khuyết EXT_SOURCE_1/2/3 bằng mô hình hồi quy (giải pháp hạng 2) + phần dư | +0.001 → +0.003 | 5–10 phút | thấp (không dùng TARGET) |
| 4 | Đặc trưng ngày/tuổi theo nghiệp vụ (tuổi tại các mốc, trễ đổi hộ chiếu 20/45, tuổi khi trả xong) | +0.0005 → +0.002 | ~0 | thấp |
| 5 | Mạng nơ-ron dạng bảng hiện đại RealMLP / TabM (TabArena 2025) để đa dạng hóa stacking | +0.001 → +0.003 (ở mức ensemble) | 30–90 phút | cài torch ~3 GB; NN đơn lẻ yếu hơn GBDT |
| 6 | Target encoding OOF cho tổ hợp danh mục (ORG×OCC, sản phẩm×loại HĐ…) | +0.0003 → +0.0015 | phút | rò rỉ nếu không lồng fold |
| 7 | Chọn đặc trưng bằng Null Importance (kernel Home Credit của Olivier) | 0 → +0.001 | 20–40 phút | overfit lựa chọn vào CV |
| 8 | Ghép khách hàng lặp lại (client matching) qua khóa ngày bất biến | +0.0002 → +0.001 | phút | ghép nhầm; độ phủ 2% |
| 9 | Biến thể boosting để đa dạng: LightGBM DART / extra_trees / linear_tree | 0 → +0.001 | 10–30 phút | DART chậm, không early-stop |
| 10 | (Không khuyến nghị lúc này) TabICL v2 / TabPFN-2.5 | ? | rất cao trên CPU | không khả thi ở 171k dòng không GPU |

---

##### 1. Tái dựng kỳ hạn & lãi suất khoản vay (term/rate reconstruction)
- **Nguồn**: Giải pháp hạng 2 Home Credit (đội ikiri_DS, slide "Home Credit Default Risk 2nd place solutions",
  https://speakerdeck.com/hoxomaxwell/home-credit-default-risk-2nd-place-solutions — slide 9, 13): lãi suất suy ra từ (tiền gốc, annuity, kỳ hạn)
  có độ quan trọng "cực cao", nhóm nghi đó là META-FEATURE từ mô hình chấm điểm nội bộ của Home Credit (lãi suất = giá rủi ro);
  bảng application thiếu kỳ hạn nên họ xây "Interest Rate Estimator" (R² 0.998). Mã nguồn: https://github.com/KazukiOnodera/Home-Credit-Default-Risk (py/000.py có
  `AMT_CREDIT-d-AMT_ANNUITY`, `AMT_GOODS_PRICE-d-AMT_ANNUITY`, `age_finish_payment`).
- **Cơ chế**: Home Credit định giá theo rủi ro → lãi suất được duyệt chứa đánh giá rủi ro của chính ngân hàng. C/A trộn lẫn kỳ hạn và lãi suất
  (C/A=34.1 có thể là 36 tháng @3.6%/năm, 48 tháng @18% hoặc 60 tháng @25%). Ta không có CNT_PAYMENT, nhưng vì sản phẩm dùng một lưới
  (kỳ hạn, lãi suất) cố định, cặp đúng là cặp mà lãi suất suy ra rơi vào một "đỉnh mật độ" trên toàn bộ train+test. Kiểm tra nhanh:
  AUC của lãi suất ước lượng trong cùng bin C/A tới 0.646 → thông tin mới thật sự.
- **Cài đặt** (chỉ dùng AMT_*, train+test gộp, không dùng TARGET):
  ```python
  TERMS=[6,10,12,18,24,30,36,48,60]   # kỳ hạn thực tế của HC (CNT_PAYMENT trong previous_application); thử thêm 42,54
  def rate(C,A,n):   # giải A = C*r/(1-(1+r)^-n) bằng chia đôi, r∈[0,0.2]/tháng; NaN nếu A*n<C
  for n in TERMS: r_n = rate(C,A,n)*12
      key = np.round(r_n*1000)   # bin 0.1%/năm (thử 0.25%)
      dens_n = key.map(key.value_counts())   # mật độ trên train+test (cash loans)
  term_hat = argmax_n dens_n ; rate_hat = r_{term_hat}; dens_max; dens_ratio = top1/top2 (độ chắc chắn)
  overpay = A*term_hat - C ; overpay_rel = overpay/C ; age_end = -DAYS_BIRTH/365 + term_hat/12
  PRODUCT = round(C/A, 2) (+ round(C/GOODS,3)) → freq encoding + dùng làm categorical cho CatBoost
  với revolving loans: đặt term_hat/rate_hat = NaN (C/A=20 cố định)
  ```
  Bản nâng cao (giống "estimator" của hạng 2): chọn n bằng cực đại xác suất hậu nghiệm = prior(n) × mật độ(r_n), prior(n) lấy từ tần suất các
  kỳ hạn chuẩn; hoặc làm mịn mật độ bằng KDE 1 chiều. Giữ cả r_12, r_24, r_36, r_48, r_60 dạng thô (để GBDT tự chọn).
- **Kỳ vọng**: +0.002…+0.008 (cao nhất trong danh sách; ở cuộc thi gốc lãi suất là đặc trưng quan trọng hàng đầu nhóm previous).
- **Chi phí**: vài giây (vector hóa numpy). **Rủi ro**: gán nhầm kỳ hạn cho cụm hiếm → chỉ là nhiễu; không rò rỉ.

##### 2. KNN-target-mean (500 láng giềng) — đặc trưng của đội vô địch
- **Nguồn**: bình luận của đội thắng cuộc trên Kaggle (https://www.kaggle.com/c/home-credit-default-risk/discussion/64480, dẫn tới bài viết
  hạng 1 discussion/64821): "one interesting feature we used that was very strong was the mean targets of the 500 nearest neighbors, where
  neighbors were defined by the three external credit ratings and the credit amount/annuity ratio". Được nhắc lại trong case-study
  https://medium.com/thecyphy/home-credit-default-risk-part-2-84b58c1ab9d5 (`TARGET_NEIGHBORS_500_MEAN`, độ quan trọng cao nhất).
- **Cơ chế**: ước lượng xác suất vỡ nợ phi tham số, trơn trong không gian 4 chiều mạnh nhất (EXT1..3, C/A). GBDT xấp xỉ bề mặt này bằng các
  bậc thang song song trục; KNN cho bề mặt trơn, tương tác đầy đủ → bổ sung, đặc biệt cho dữ liệu ít cột như ở đây.
- **Cài đặt** (sklearn đã có):
  ```python
  Z = [EXT1, EXT2, EXT3, C/A]  -> QuantileTransformer(normal) fit trên train+test; NaN: điền bằng EXT dự đoán ở mục 3
       (hoặc median + thêm cột cờ); thử thêm biến thể Z2=[EXT_MEAN, EXT2, C/A, DAYS_BIRTH]
  for outer fold k (5-fold rs=42):
      Tk = các dòng train ngoài fold k
      # đặc trưng cho dòng thuộc Tk: inner 5-fold trên Tk (KDTree fit trên inner-train, query inner-valid)
      # đặc trưng cho fold k (valid) và cho test: KDTree fit trên toàn bộ Tk
      feat = mean(y[neighbors]) với K∈{100, 500, 2000}; thêm trung bình có trọng số 1/khoảng cách
  # nộp bài: train cuối trên toàn train dùng OOF 5-fold; test query toàn train
  ```
  Dùng `sklearn.neighbors.KDTree` (4 chiều → rất nhanh), `n_jobs=4`.
- **Kỳ vọng**: +0.001…+0.004. **Chi phí**: 5–15 phút (30 lần fit/query KDTree ~137k điểm).
- **Rủi ro**: RÒ RỈ nếu tính OOF một tầng rồi dùng trong CV (đặc trưng của tập train chứa nhãn fold valid) → bắt buộc lồng như trên.
  Có thể lệch nhẹ phân phối giữa train (inner 80%) và valid (100% Tk) — chấp nhận được, giống target encoding chuẩn.

##### 3. Điền khuyết EXT_SOURCE bằng mô hình (giải pháp hạng 2) + phần dư
- **Nguồn**: mã hạng 2 `py/011_EXT_1.py … 016_EXT_3.py` (https://github.com/KazukiOnodera/Home-Credit-Default-Risk/tree/master/py):
  LGB regression (lr 0.01, num_leaves 63, max_depth 6, min_data_in_leaf 150, colsample 0.9) học EXT_SOURCE_k từ các cột khác trên các dòng
  train+test CÓ giá trị, dự đoán cho dòng thiếu; không dùng TARGET.
- **Cơ chế**: EXT_SOURCE_1 thiếu ~56%, EXT_SOURCE_3 ~20% và là 3 đặc trưng mạnh nhất; các tổng hợp EXT_MEAN/PROD/W hiện đang tính trên phần
  không thiếu → nhiễu. Giá trị điền hợp lý làm EXT_MEAN/PROD/KNN (mục 2) chính xác hơn. Phần dư (EXT_k − dự đoán, tính OOF trên dòng có giá trị)
  đo mức "điểm tín dụng bất thường so với hồ sơ" — tín hiệu mới.
- **Cài đặt**: với k=1,2,3: X = tất cả cột gốc + FE (bỏ EXT_k và các cột dẫn xuất từ EXT_k), dữ liệu = train+test có EXT_k; 5-fold KFold →
  `EXT_k_PRED` (OOF cho dòng có giá trị, trung bình fold cho dòng thiếu), `EXT_k_FILL = EXT_k.fillna(PRED)`, `EXT_k_RES = EXT_k − PRED`.
  Tính lại EXT_MEAN_F / EXT_PROD_F / EXT_W_F trên bản FILL; giữ NGUYÊN cờ thiếu.
- **Kỳ vọng**: +0.001…+0.003. **Chi phí**: 3 mô hình × 5 fold LGB hồi quy ~5–10 phút. **Rủi ro**: thấp; không dùng nhãn.

##### 4. Đặc trưng ngày/tuổi theo nghiệp vụ
- **Nguồn**: hạng 2 (py/000.py: `DAYS_X-m-DAYS_BIRTH` các hiệu, `age_finish_payment`); quy định hộ chiếu Nga (cấp 14 tuổi, đổi 20 và 45 tuổi) —
  xác nhận trên dữ liệu: đỉnh ở 20.0–20.5 và 45.0–45.5 tuổi.
- **Cơ chế**: các tỷ lệ /DAYS_BIRTH hiện có không biểu diễn được "tuổi tại sự kiện" và "độ trễ so với mốc pháp lý". Người đổi hộ chiếu trễ
  (thiếu kỷ luật hành chính) vỡ nợ nhiều hơn: AUC trong nhóm tuổi 0.53–0.546.
- **Cài đặt**:
  ```python
  age = -DAYS_BIRTH/365.25
  for c in [DAYS_ID_PUBLISH, DAYS_REGISTRATION, DAYS_EMPLOYED, DAYS_LAST_PHONE_CHANGE]: df[c+'_AGE_AT'] = (c - DAYS_BIRTH)/365.25
  ida = ID_PUBLISH_AGE_AT; df['ID_LATE'] = where(ida>=45, ida-45, where(ida>=20, ida-20, ida-14))
  df['ID_ON_TIME'] = ID_LATE < 0.1 ; df['ID_OVERDUE'] = (age>=45)&(ida<45)  | (age>=20)&(ida<20)
  df['YEARS_TO_45'] = 45-age ; df['AGE_END_LOAN'] = age + CREDIT_ANNUITY/12 (hoặc term_hat/12 ở mục 1)
  df['PHONE_CHANGED_TODAY'] = DAYS_LAST_PHONE_CHANGE==0 ; df['GOODS_ANNUITY'] = AMT_GOODS_PRICE/AMT_ANNUITY
  df['EMP_START_AGE'] = (DAYS_EMPLOYED-DAYS_BIRTH)/365.25 ; df['ID_MINUS_EMP'] = DAYS_ID_PUBLISH-DAYS_EMPLOYED
  ```
- **Kỳ vọng**: +0.0005…+0.002. **Chi phí**: ~0. **Rủi ro**: thấp.

##### 5. Mạng nơ-ron dạng bảng hiện đại (RealMLP, TabM) để đa dạng hóa stacking
- **Nguồn**: TabArena (Erickson et al., NeurIPS 2025, https://arxiv.org/abs/2506.16791): với tuning + ensembling, RealMLP (Elo ~1569) và TabM (~1552)
  ngang hoặc vượt LightGBM (~1532), CatBoost (~1489). RealMLP: Holzmüller et al. 2024 "Better by default" (package `pytabkit`);
  TabM: Gorishniy et al., ICLR 2025 (package `tabm`). Đội vô địch Home Credit cũng dùng NN (DAE+NN của M. Jahrer) làm mô hình tầng 1.
- **Cơ chế**: lỗi của NN ít tương quan với GBDT (bề mặt trơn, embedding số dạng periodic/PLR) → stacking LR hiện tại (XGB trọng số ≈0 vì quá giống
  LGB) có thêm thành phần đa dạng thật. Kỳ vọng NN đơn lẻ ~0.752–0.760, nhưng blend tăng.
- **Cài đặt**: `pip install torch` (PyPI torch 2.15.0, ~3 GB kèm thư viện CUDA — download.pytorch.org/whl/cpu bị proxy chặn; ổ đĩa còn 26 GB),
  `pip install pytabkit==1.8.0` (đã kiểm tra tải được), hoặc `tabm==0.0.3`.
  ```python
  from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier
  m = RealMLP_TD_Classifier(device='cpu', n_threads=4, random_state=0, val_metric_name='1-auc_ovr', n_epochs=256)
  # tiền xử lý: dùng cột gốc + FE quan trọng; cats để pytabkit tự one-hot/embedding; NaN: thêm cờ + điền median
  # giữ đúng folds rs=42; mỗi fold tách 10% làm val nội bộ cho early stopping
  # TabM_D_Classifier(device='cpu', tabm_k=16) nếu thời gian cho phép
  ```
  Rồi đưa logit OOF vào stacking LR lồng nhau như vòng 0.9.
- **Kỳ vọng**: +0.001…+0.003 ở mức ensemble. **Chi phí**: RealMLP ~3–8 phút/fold, TabM ~10–25 phút/fold trên 4 nhân.
  **Rủi ro**: cài đặt nặng; NN nhạy với tiền xử lý (cần QuantileTransformer/robust scaling, đã có trong RealMLP-TD).

##### 6. Target encoding OOF cho tổ hợp danh mục
- **Nguồn**: hạng 2 `py/003_TargetEncoding.py`, `004_TargetEncoding.py` (OOF mean target theo 5-fold); Micci-Barreca (2001) smoothing.
- **Cơ chế**: ORGANIZATION_TYPE (58 mức) × OCCUPATION_TYPE (19) × NAME_INCOME_TYPE tạo tổ hợp nhiều mức mà GBDT khó cắt; frequency encoding /
  group-mean-diff hiện có không chứa thông tin nhãn. Thêm "PRODUCT" (mục 1) × NAME_CONTRACT_TYPE.
- **Cài đặt**: smoothing m=50–200; LỒNG: trong mỗi outer fold, TE của dòng train tính bằng inner 5-fold, TE của valid/test tính trên toàn Tk.
  Hoặc rẻ hơn: CatBoost với các cột tổ hợp dạng chuỗi + `max_ctr_complexity=3` (CatBoost tự làm ordered TS, không rò rỉ).
- **Kỳ vọng**: +0.0003…+0.0015. **Chi phí**: phút. **Rủi ro**: rò rỉ nếu không lồng fold; tổ hợp quá hiếm → nhiễu (dùng min_count≥30).

##### 7. Chọn đặc trưng bằng Null Importance
- **Nguồn**: kernel "Feature Selection with Null Importances" (Olivier Grellier) viết cho chính cuộc thi Home Credit
  (https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances).
- **Cơ chế**: vòng 0.2 thêm 40 đặc trưng mà OOF không tăng → nhiều cột nhiễu/trùng; với cây nông (depth 3–4, theo Optuna) cột nhiễu chiếm split.
  So độ quan trọng thật với phân phối độ quan trọng khi xáo trộn nhãn (50–80 lần) → bỏ cột có gain ≤ phân vị 75% của null.
- **Cài đặt**: LGB nhanh (lr 0.1, 200 cây, feature_fraction 0.8) × 60 lần xáo trộn TARGET; score = log(1e-10 + imp_thật/(1+P75(imp_null)));
  thử các ngưỡng {0,10,20,30} và chọn bằng 5-fold rs=42. Làm sau khi thêm mục 1–4.
- **Kỳ vọng**: 0…+0.001. **Chi phí**: 20–40 phút. **Rủi ro**: chọn ngưỡng theo CV → overfit nhẹ; giữ ngưỡng thô.

##### 8. Ghép khách hàng lặp lại (client matching)
- **Nguồn**: suy luận từ cấu trúc dữ liệu (ngày tính tương đối so với ngày nộp đơn → hiệu giữa hai cột DAYS là bất biến theo người); kỹ thuật
  "tìm user ẩn" phổ biến trong các cuộc thi Kaggle (IEEE-CIS Fraud 2019 — uid từ D-columns). Kiểm tra: 2.2% dòng có ≥1 bản ghi trùng khóa.
- **Cài đặt**: uid = (DAYS_BIRTH−DAYS_ID_PUBLISH, DAYS_BIRTH−DAYS_REGISTRATION, DAYS_BIRTH−DAYS_EMPLOYED, CODE_GENDER); đặc trưng:
  UID_CNT (train+test), khoảng cách thời gian tới đơn trước/sau (hiệu DAYS_BIRTH), thay đổi income/credit so với đơn khác, và (tùy chọn) OOF
  mean TARGET của các đơn KHÁC của cùng uid trong train (lồng fold như mục 2; với test chỉ dùng nhãn train).
- **Kỳ vọng**: +0.0002…+0.001 (độ phủ nhỏ). **Rủi ro**: ghép nhầm (DAYS_EMPLOYED=365243 → loại khỏi khóa), rò rỉ nếu TE không lồng fold.

##### 9. Biến thể boosting để đa dạng hóa
- LightGBM `boosting='dart'` (drop_rate 0.1, skip_drop 0.5, ~1500–3000 vòng, lr 0.02, không early stopping — chọn số vòng bằng CV),
  `extra_trees=True`, `linear_tree=True` (+ `linear_lambda`); `path_smooth`. Dùng như mô hình tầng 1 mới cho stacking.
- Nguồn: tài liệu LightGBM (https://lightgbm.readthedocs.io/en/latest/Parameters.html); DART (Rashmi & Gilad-Bachrach, AISTATS 2015);
  nhiều kernel Home Credit dùng DART.
- **Kỳ vọng**: 0…+0.001 (chủ yếu qua đa dạng). **Chi phí**: DART chậm hơn 3–5× (~10–30 phút).

##### 10. (Không khuyến nghị ở vòng này) Mô hình nền tảng dạng bảng TabICL v2 / TabPFN-2.5
- TabPFN-2.5 (Prior Labs, arXiv 2511.08667): thiết kế cho ≤50k dòng (thử tới 100k) và cần GPU (160k dòng ≈ GPU 80 GB); trọng số cần chấp nhận
  license trên HuggingFace. `tabpfn` 9.2.0, `tabicl` 2.2.0 có trên PyPI, nhưng chi phí attention ~O(n²) trên 137k dòng × 5 fold × nhiều ensemble
  member trên CPU 4 nhân là nhiều giờ. Chỉ đáng thử dạng phụ: ngữ cảnh lấy mẫu 10–20k dòng, làm thêm một mô hình tầng 1 — kỳ vọng thấp.

##### Không khuyến nghị
- Pseudo-labeling trên test: với AUC và 8% dương, mức tăng thường ≈0, thêm rủi ro tự xác nhận lỗi.
- Adversarial validation: chỉ dùng để chẩn đoán (AUC theo ID = 0.503 → nhiều khả năng chia ngẫu nhiên, không có lợi ích).

##### Thứ tự thử đề xuất cho Tác nhân 2
1+4 (rẻ, cùng một lần chạy LGB) → 3 → 2 (dùng EXT đã điền) → 6 → chạy lại LGB/Cat với bộ mới → 5 và 9 để thêm đa dạng → stacking LR lồng nhau → 7 nếu còn thời gian.
So sánh từng bước với LGB 3 seed (OOF 0.7615); coi mức tăng < 0.0005 là nhiễu (độ lệch giữa seed).

##### Đánh giá trung thực về mức trần AUC
- Cuộc thi gốc: hạng 1 đạt ~0.805 private LB, nhưng dùng đủ 7 bảng (bureau, previous, installments… đóng góp phần lớn mức tăng từ ~0.76 lên ~0.80).
  Các notebook phổ biến chỉ dùng application_train (122 cột, 307k dòng) thường đạt khoảng 0.755–0.77.
- Ở đây chỉ có 33/122 cột (mất FLAG_DOCUMENT_*, thông tin nhà ở, social circle, REGION_RATING_W_CITY, giờ nộp đơn…) và 171k dòng (≈56% số dòng gốc),
  nên trần thực tế thấp hơn. 3 thuật toán GBDT khác nhau đều dừng ~0.761 và stacking chỉ lên 0.7627 cũng xác nhận điều đó.
- Nếu các mục 1–5 cộng dồn tốt, kỳ vọng hợp lý là OOF **~0.765–0.770**, lạc quan ~0.772. Mức **0.80 gần như chắc chắn không đạt được
  bằng phương pháp hợp lệ** với chỉ bảng application 33 cột. Chỉ mục 1 (lãi suất) có tiềm năng tạo bước nhảy >0.005, vì nó khôi phục một phần
  đánh giá rủi ro nội bộ của Home Credit.

##### Nguồn
- Kaggle Home Credit discussion 64480 (bình luận đội thắng, KNN-500): https://www.kaggle.com/c/home-credit-default-risk/discussion/64480 ; bài viết hạng 1: /discussion/64821
- Slide hạng 2 (ikiri_DS): https://speakerdeck.com/hoxomaxwell/home-credit-default-risk-2nd-place-solutions
- Mã hạng 2: https://github.com/KazukiOnodera/Home-Credit-Default-Risk (py/000.py, 003_TargetEncoding.py, 011–016_EXT_*.py)
- Case-study KNN feature: https://medium.com/thecyphy/home-credit-default-risk-part-2-84b58c1ab9d5
- Giải pháp top 17 (#1 solo): https://github.com/NoxMoon/home-credit-default-risk
- TabArena: https://arxiv.org/abs/2506.16791 ; TabPFN-2.5: https://arxiv.org/abs/2511.08667
- Null importance kernel: https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances
- Gói pip đã kiểm tra: pytabkit 1.8.0, tabm 0.0.3, tabicl 2.2.0, tabpfn 9.2.0, gplearn 0.4.3, torch 2.15.0 (PyPI; index CPU của pytorch bị chặn).

### Tác nhân 2 — Rút bài học từ Vòng 0 (`loop/round1_agent2.md`)
#### VÒNG 1 — Tác nhân 2 (Rút kinh nghiệm)

Nguồn: `history.md`, `fe.py`, `run.py`, `opt.py`, `log_cx.txt`, `log_opt.txt`, các `oof_*/pt_*.npy`.
Phân tích nhẹ của tôi: script trong `$S/a2/` (`a1.py` … `a6.py`). Không sửa file nào trong repo.

Quy ước số liệu bên dưới:
- "OOF" = AUC trên 5 fold rs=42 (so sánh được với bảng vòng 0).
- "Thử nhanh F0/F1" = LightGBM lr 0.05, cây nông (leaves 14, depth 4, ff 0.2, l1 10), early stopping 200,
  chạy fold 0 và fold 1 × 2 seed = 4 lần chấm có ghép cặp. Điểm tuyệt đối thấp hơn OOF thật (lr cao) nhưng
  CHÊNH LỆCH ghép cặp thì dùng được.

---------------------------------------------------------------------------------------------------
##### 1. Thất bại & nguyên nhân

| Thử nghiệm | Kết quả | Bằng chứng | Nguyên nhân |
|---|---|---|---|
| 0.2: thêm ~40 FE (EXT×age, cặp tích EXT, freq enc, diff theo nhóm, TERM_ROUND, REQ_SUM) | 0.7615 → 0.76152 (≈ 0) | Thử nhanh fold 0 (trung bình 3 seed): **33 cột gốc + 8 FE lõi = 0.75749** vs **toàn bộ 93 cột = 0.75768** → 85 cột thêm chỉ đóng góp ~+0.0002, nhỏ hơn nhiễu seed (0.0012). 23/93 cột có gain < 0.3%, 2 cột gain = 0. | Các FE này là phép biến đổi lại của CÙNG thông tin (EXT_SOURCE, AMT_*, DAYS_*) mà GBDT tự học được qua phân nhánh. Không đưa thông tin MỚI. |
| 0.7: Optuna (LGB 30, XGB 20, Cat 12 trial) | LGB 0.76152→0.76153, XGB 0.76100→0.76118, Cat 0.76080→0.76063 | Chênh ghép cặp theo fold lgb_opt − lgb: [+0.0002, −0.0004, +0.0004, −0.0009, +0.0006], TB −0.00003, std 0.00054 → **bằng 0**. Cat_opt thua cat 5/5 fold (TB −0.0009 so với lgb). Khoảng điểm giữa các trial (0.7583–0.7616) ≈ nhiễu seed. | (a) Bề mặt tham số phẳng: mọi cấu hình cây nông + regularize đều ra ~0.761. (b) Hàm mục tiêu tối ưu (3 fold, lr 0.05) ≠ cấu hình cuối (5 fold, lr 0.01) → tham số "tốt nhất" không chuyển giao được. (c) Lời nguyền người thắng: trial tốt nhất được chọn trên chính 3 fold đó nên điểm lạc quan. (d) Dữ liệu bị giới hạn thông tin, không phải mô hình. |
| 0.6 / 0.9: blend có trọng số, stacking LR | 0.76246 → 0.76258 / 0.76273 (+0.0001–0.0003) | SE bootstrap của AUC tổng = 0.0024; chênh ghép cặp rank6 − lgb SE ≈ 0.0002 → +0.0001 nằm trong nhiễu. Stacking cho XGB trọng số ≈ 0. | Tương quan Spearman OOF: LGB–XGB **0.995**, LGB–lgb_opt 0.992, XGB–xgb_opt 0.993 → các mô hình gần như trùng nhau; không còn gì để tối ưu trọng số. |
| Thêm "mô hình" bằng cách tune lại cùng thuật toán (lgb_opt, xgb_opt, cat_opt) | rank 3 → rank 6: 0.76246 → 0.76266 (+0.0002) | corr OOF giữa bản gốc và bản opt cùng thuật toán 0.988–0.993 | Cùng đặc trưng + cùng họ GBDT → không tạo đa dạng. |
| Chọn đặc trưng theo gain (top-40) | fold 0: 0.75449 vs 0.75713–0.75835 (đầy đủ) | giảm ~0.003, lớn hơn nhiễu seed | Gain trên 1 fold không ổn định; bỏ cột gain thấp làm mất thông tin tương tác. → KHÔNG cắt đặc trưng theo gain. |

##### 2. Mẹo từ các lần thành công

1. **FE lõi quanh EXT_SOURCE và khoản vay là đòn bẩy lớn nhất đã thấy**: fold 0, 33 cột gốc = 0.74737 → thêm 8 FE lõi
   (CREDIT_ANNUITY, EXT_MEAN, EXT_PROD, EXT_W, EXT_NA, ANNUITY_INCOME, CREDIT_GOODS, DAYS_EMPLOYED_R_BIRTH) = 0.75749 (**+0.010**).
   Top gain hiện tại: ORGANIZATION_TYPE_EXT_MEAN_DIFF 8%, EXT_W 7.2%, EXT_MIN 6.6%, OCCUPATION_TYPE 3.4%.
   → Bài học: tăng điểm đến từ đặc trưng mang **thông tin mới/khó học bằng cây** (tổ hợp EXT bù chỗ thiếu, tỉ lệ tín dụng), không phải từ số lượng cột.
2. **Rank-average giữa các THUẬT TOÁN khác nhau là tăng thật**: rank6 − lgb theo fold = [+0.0015, +0.0011, +0.0011, +0.0008, +0.0012],
   TB +0.00114, std 0.00023 → dương 5/5 fold, t ≈ 10. Đóng góp chính đến từ CatBoost (corr với LGB chỉ 0.973, thấp nhất).
3. **Cây nông + regularization mạnh** (depth 3–4, l1 lớn, feature_fraction 0.1–0.3) là vùng tối ưu chung của cả 3 thuật toán → dùng làm mặc định, không cần tune lại.
4. **Rank-avg đơn giản ≈ stacking**: khi mô hình tương quan > 0.97, trọng số bằng nhau là đủ; stacking chỉ hợp lý khi có mô hình thật sự khác (corr < 0.95).

##### 3. Phát hiện bất ngờ (phân tích nhẹ vòng 1)

1. **Điền khuyết EXT_SOURCE bằng mô hình (không dùng TARGET) tăng rõ**: huấn luyện LGB hồi quy dự đoán EXT_SOURCE_k từ các cột khác
   trên train+test (OOF 3-fold cho dòng có giá trị). Tương quan dự đoán–thực: EXT1 **0.695**, EXT2 0.496, EXT3 0.369.
   Thêm `EXT_k_PRED`, `EXT_k_FILL` (giá trị thật, thiếu thì lấy dự đoán), `EXT_k_RES` (thật − dự đoán), `EXTF_MEAN/PROD/W`:
   thử nhanh F0/F1 TB 0.75919 → **0.76144 (+0.0022)**, ghép cặp [+0.0024, +0.0031, +0.0028, +0.0007] → dương 4/4, gấp ~2× nhiễu seed.
   Lý do: EXT1 thiếu 69.5% (cao hơn bộ Kaggle gốc ~56%), EXT3 thiếu 32%; nhóm thiếu EXT là nơi mô hình yếu nhất (xem mục 5).
2. **Lãi suất ngầm định của khoản vay (giá ngân hàng tự định theo rủi ro) tăng rõ**: với k = AMT_CREDIT/AMT_ANNUITY, giải
   (1 − (1+r)^−n)/r = k cho các kỳ hạn n ∈ {12, 24, 36, 48, 60} → `RATE_n` (lãi năm; NaN nếu k ≥ n).
   Thử nhanh F0/F1: 0.75919 → **0.76136 (+0.0022)**, ghép cặp [+0.0029, +0.0018, +0.0022, +0.0019] → dương 4/4, rất ổn định.
   Cây không tự học được hàm phi tuyến này từ CREDIT_ANNUITY (TERM_ROUND cũ chỉ là làm tròn k, không có thông tin lãi suất).
   Lưu ý: khoản Revolving có k = 20 cố định (annuity = 5% hạn mức) → RATE vô nghĩa cho 9.5% số dòng đó, nên để NaN.
   **Kết hợp cả 1 và 2**: thử nhanh F0/F1 = **0.76299 (+0.0038)**, ghép cặp [+0.0045, +0.0038, +0.0038, +0.0032] → gần như CỘNG DỒN (2 nguồn thông tin độc lập).
   Gain trong mô hình kết hợp: EXTF_W 15.4%, EXTF_MEAN 11.2%, EXTF_PROD 7.0%, EXT_SOURCE_3_PRED 3.5%, RATE_24/RATE_48 1.7% mỗi cột
   → các tổ hợp EXT ĐÃ ĐIỀN thay thế vị trí dẫn đầu của tổ hợp EXT cũ.
   (Code tái dùng: `$S/a2/a5.py` — phần điền khuyết và hàm `rate()`; `$S/a2/a6.py` — chạy kết hợp.)
3. **Train và test cùng phân phối**: adversarial AUC = **0.512** (gần ngẫu nhiên); tỉ lệ thiếu các cột train/test gần trùng (EXT1 0.695/0.698, EXT3 0.319/0.320).
   Không trùng dòng trong train, không trùng train–test. → CV đáng tin, không cần adversarial weighting; điểm LB ≈ OOF ± nhiễu mẫu.
4. **Hiệu năng theo nhóm (rank6 OOF, AUC tổng 0.76266)**:
   - Số EXT bị thiếu: 0 → 0.7735 (21%), 1 → 0.7668 (56%), **2 → 0.7399 (23%, tỉ lệ vỡ nợ 9.1% cao nhất)**.
   - EXT3 thiếu: 0.7430 vs có: 0.7711.
   - DAYS_EMPLOYED = 365243 (hưu trí/không việc): **0.7337** (18% dòng). Tuổi > 60: **0.7214**; < 30: 0.7354.
   - Hợp đồng: Cash 0.7618, Revolving 0.7573. Giới tính không khác biệt (0.758/0.758).
   → Điểm yếu nằm ở người thiếu EXT_SOURCE và nhóm hưu trí/già — chính là nơi điền khuyết EXT giúp.
5. **Nhiễu đo lường cụ thể** (dùng để đặt ngưỡng):
   - Std AUC giữa 5 fold của 1 mô hình: 0.0022–0.0034 (fold 2 luôn cao nhất ~0.766, fold 3 thấp nhất ~0.757 với mọi mô hình → khác biệt do fold, không do mô hình).
   - Nhiễu seed trên 1 fold (lr 0.05): 0.75713 / 0.75835 / 0.75755 → khoảng 0.0012.
   - SE bootstrap của AUC OOF tổng: 0.0024; nhưng SE của CHÊNH LỆCH ghép cặp giữa 2 mô hình trên cùng OOF chỉ ~0.0002–0.0005.
6. **Thiên lệch lạc quan nhỏ trong OOF**: `run.py`/`opt.py` dùng chính fold kiểm định để early stopping → OOF hơi lạc quan
   (vài 1e-4), và Optuna chọn trial trên chính 3 fold rồi chấm lại trên 5 fold chứa 3 fold đó. Không phải rò rỉ nhãn test, nhưng làm các
   chênh lệch +0.0001 càng vô nghĩa.
7. Dữ liệu là tập con 33/122 cột của Kaggle gốc (thiếu FLAG_DOCUMENT_*, chỉ số nhà ở, SOCIAL_CIRCLE, HOUR_APPR_PROCESS, REGION_RATING_W_CITY…),
   SK_ID_CURR đánh lại 0..232701. Mục tiêu 0.80 gần như chắc chắn **không đạt được** chỉ với dữ liệu này; trần thực tế ước ~0.765–0.770.

##### 4. Cách ghép kỹ thuật & tối ưu logic — quy tắc cho Tác nhân 3

**Đánh giá**
- R1. Luôn so sánh **ghép cặp** (cùng fold, cùng seed) với baseline, báo chênh lệch từng fold, không chỉ OOF tổng.
- R2. Ngưỡng có ý nghĩa: một thay đổi được giữ khi chênh OOF ≥ **+0.0005** VÀ dương ở **≥ 4/5 fold**
  (hoặc trong thử nhanh: dương ≥ 4/4 lần chấm và TB ≥ +0.001). Dưới +0.0003 coi như 0, đừng ghi là "tăng".
- R3. Sàng lọc nhanh trước: fold 0+1 × 2 seed, lr 0.05 (~10 s/lần) → chỉ ý tưởng qua sàng mới chạy đủ 5 fold lr 0.01.
  Tiết kiệm ~10× tính toán so với chạy full cho mọi ý tưởng.
- R4. Mọi đặc trưng dùng TARGET (target encoding, KNN target mean, stacking) phải tính TRONG fold (chỉ từ phần train của fold), kể cả khi dựng cho test.
  Thống kê không dùng TARGET (freq, group mean, mô hình điền khuyết) được phép dùng train+test.

**Phân bổ tính toán**
- R5. **Dừng Optuna/tune tham số** cho GBDT: đã chứng minh = 0. Dùng tham số opt hiện có làm mặc định. Chỉ tune nếu đổi bộ đặc trưng rất khác, và khi đó ≤ 15 trial, chấm trên 5 fold lr cố định.
- R6. Ưu tiên theo thứ tự: (1) **thông tin mới** (đặc trưng không suy ra được bằng cây: điền khuyết EXT, lãi suất ngầm định, KNN-target) >
  (2) **đa dạng mô hình** (thuật toán khác họ, corr OOF < 0.97) > (3) multi-seed > (4) trọng số blend > (5) tham số.
- R7. Đừng thêm hàng loạt đặc trưng cùng loại (ratio/diff/cặp tích) — vòng 0 đã chứng minh 85 cột như vậy ≈ 0. Thêm theo **nhóm có giả thuyết**, mỗi nhóm kiểm R2/R3 riêng.
- R8. Không cắt đặc trưng theo gain (top-40 làm giảm ~0.003).

**Ghép kỹ thuật**
- R9. Mô hình mới chỉ có ích cho ensemble nếu Spearman corr OOF với rank-avg hiện tại < ~0.97 VÀ AUC đơn lẻ ≥ ~0.755.
  Kiểm corr trước khi bỏ công tune.
- R10. Blend: rank-average trọng số bằng nhau làm mặc định; chỉ dùng stacking/trọng số khi có ≥ 1 mô hình corr < 0.95, và luôn đánh giá CV lồng nhau.
- R11. Khi đổi bộ đặc trưng, chạy lại cả 3 thuật toán (LGB/Cat/XGB) trên bộ mới rồi rank-avg — lợi ích FE (+0.002) và ensemble (+0.001) có vẻ cộng được vì khác nguồn.
- R12. Mỗi kết quả ghi vào history.md kèm: OOF, chênh từng fold so với baseline, số seed. Lưu oof/pt với tên riêng để tái blend.

##### 5. Chẩn đoán & hướng ưu tiên

**Điểm nghẽn: DỮ LIỆU/ĐẶC TRƯNG, không phải mô hình hay tham số.**
- Mô hình: 3 thuật toán × tune ≈ 0.7606–0.7615, corr 0.97–0.995 → đã bão hòa.
- Đánh giá: ổn (adversarial 0.512, train/test đồng phân phối); chỉ cần kỷ luật ghép cặp (R1–R3).
- Đặc trưng: hai nhóm thông tin mới đầu tiên thử (điền khuyết EXT, lãi suất ngầm định) đều cho +0.002 ổn định
  → dư địa còn nằm ở đây. Nhóm yếu nhất: thiếu ≥ 2 EXT (AUC 0.740), hưu trí (0.734), > 60 tuổi (0.721).

**Hướng ưu tiên cho vòng tới (theo thứ tự):**
1. **Đưa vào fe.py: điền khuyết EXT_SOURCE bằng mô hình + RATE_n**, chạy đủ 5 fold cho LGB/Cat/XGB, rank-avg.
   Kỳ vọng OOF đơn LGB ~0.764–0.765 (thử nhanh +0.0038), ensemble ~0.766. Mở rộng: điền khuyết bằng nhiều thuật toán, dùng thêm kỳ hạn n=6,10,18,30,42 cho RATE;
   tính RATE với AMT_GOODS_PRICE làm gốc (phần chênh credit−goods là bảo hiểm/phí); lọc RATE hợp lý theo NAME_CONTRACT_TYPE.
2. **KNN target-mean trong fold** (kiểu "neighbors_target_mean_500" của Kaggle): trung bình TARGET của k=50/500 hàng xóm gần nhất
   trong không gian chuẩn hóa {EXT_1/2/3 (bản đã điền), CREDIT_ANNUITY, DAYS_BIRTH}, tính chỉ từ phần train của fold. Thông tin cục bộ dạng
   "mượt" mà cây khó học; đồng thời cũng có thể dùng KNN làm mô hình riêng để đa dạng hóa.
3. **Mô hình khác họ để đa dạng hóa**: MLP với embedding cho biến phân loại + đặc trưng số chuẩn hóa quantile; hoặc Logistic Regression
   trên đặc trưng WOE/binning (thường corr ~0.9 với GBDT). Chỉ giữ nếu thỏa R9.
4. **Multi-seed (3 seed) cho cấu hình cuối** của mỗi thuật toán — rẻ, giảm nhiễu ~0.0003–0.0005; làm SAU khi chốt đặc trưng.
5. (Tùy chọn, rủi ro thấp lợi ích thấp) Pseudo-label test có độ tin cậy cao; mô hình riêng cho nhóm Revolving/hưu trí. Chỉ thử nếu 1–3 đã xong.

Kỳ vọng thực tế: trần ~0.765–0.770 với dữ liệu 33 cột; AUC 0.80 không khả thi một cách hợp lệ — nên báo cho người dùng.

### Tác nhân 3 — Thực hành (mục VÒNG 1 trong `loop/history.md`)
Script (trong `$S`): `fe_r1.py` (`load_r1()` = `fe.load()` + nhóm IMP/RATE/TERM/DATE; cache điền khuyết `r1_feats.pkl`), `r1_screen.py`, `r1_screen2.py` (sàng lọc),
`knn_r1.py` (KNN-target lồng fold → `r1_knn.pkl`), `run_r1.py <lgb|cat|xgb> IMP,RATE,TERM 0 <seeds> <tag>` (5 fold đầy đủ), `div_r1.py <lrbin|mlp>`,
`ens_final_r1.py` (so sánh ensemble + ghi submission). Log: `log_r1_{screen,screen2,knn,lgb,xgb,cat,lrbin,mlp}.txt`.
Bộ đặc trưng chốt: **fe.py (~80 cột) + IMP + RATE + TERM**.

Định nghĩa nhóm đặc trưng mới (tất cả KHÔNG dùng TARGET, tính trên train+test):
- **IMP**: LGB hồi quy (KFold 5, 800 vòng, lr 0.05) dự đoán EXT_SOURCE_k từ các cột không chứa EXT → `EXT_k_PRED` (OOF cho dòng có giá trị), `EXT_k_FILL`, `EXT_k_RES`, `EXTF_MEAN/PROD/MIN/MAX/STD/W` trên bản đã điền.
- **RATE**: lãi năm ngầm định `RATE_n` (n = 12/24/36/48/60) giải A = C·r/(1−(1+r)^−n) bằng chia đôi; Revolving → NaN.
- **TERM**: kỳ hạn dựng lại = argmax_n mật độ lãi suất (bin 0.25%/năm, n ∈ {6,10,12,18,24,30,36,42,48,54,60}) → `TERM_HAT, RATE_HAT, TERM_DENS, TERM_DENS_RATIO, OVERPAY_REL, AGE_END_LOAN, PRODUCT_FREQ (freq round(C/A,2)), CG_FREQ (freq round(C/GOODS,3))`.
- DATE (bị bỏ): tuổi tại các mốc DAYS_*, trễ đổi hộ chiếu 14/20/45, AGE_FRAC, YEARS_TO_45, ID_MINUS_EMP/REG, GOODS_ANNUITY.
- KNN (bị bỏ): mean TARGET của 50/200/500 láng giềng trong 2 không gian quantile-normal {EXT1/2/3_FILL, C/A} và {EXTF_MEAN, EXT2/3_FILL, C/A, DAYS_BIRTH}; lồng fold (inner 5-fold cho phần train, toàn Tk cho valid/test).

##### Sàng lọc nhanh (LGB lr 0.05, cây nông, fold 0/1 × 2 seed = 4 lần chấm ghép cặp)
| Thay đổi | TB 4 lần | Chênh ghép cặp | Dương | Quyết định |
|---|---|---|---|---|
| BASE (fe.py) | 0.75919 | — | — | |
| + IMP | 0.76157 | +0.00238 | 4/4 | GIỮ |
| + RATE | 0.76165 | +0.00246 | 4/4 | GIỮ |
| + IMP + RATE | 0.76321 | +0.00403 vs BASE (gần cộng dồn) | 4/4 | GIỮ |
| IMP+RATE + TERM | **0.76531** | +0.00210 vs IMP+RATE | 4/4 | GIỮ |
| IMP+RATE + DATE | 0.76278 | −0.00043 | 1/4 | BỎ |
| IMP+RATE+TERM + DATE | 0.76444 | −0.00087 vs IR+TERM | — | BỎ |
| IMP+RATE+TERM + KNN500 (2 cột) | 0.76477 | −0.00054 | 1/4 | BỎ |
| IMP+RATE+TERM + KNN tất cả (6 cột) | 0.76458 | −0.00074 | 1/4 | BỎ |
(KNN-500 đơn biến AUC ≈ 0.73 — mạnh nhưng trùng thông tin với EXT đã điền + C/A, và thêm nhiễu do lệch phân phối inner/outer.)

##### Chạy đầy đủ 5 fold (StratifiedKFold rs=42, early stopping trên fold valid như vòng 0)
| Mô hình (bộ IMP+RATE+TERM) | Seed | OOF AUC | AUC từng fold | Chênh ghép cặp vs bản vòng 0 cùng thuật toán |
|---|---|---|---|---|
| LGB (lr 0.01, leaves 16, ff 0.3) `lgb_r1` | 3 | **0.76696** | 0.76446 / 0.76733 / 0.77261 / 0.76322 / 0.76739 | +0.00545 +0.00477 +0.00618 +0.00540 +0.00537 → **+0.00543**, 5/5 |
| XGB (depth 5, colsample 0.3) `xgb_r1` | 1 | 0.76643 | 0.76373 / 0.76667 / 0.77194 / 0.76262 / 0.76739 | +0.00493 +0.00474 +0.00541 +0.00590 +0.00610 → **+0.00542**, 5/5 |
| CatBoost (depth 6, lr 0.03) `cat_r1` | 1 | 0.76545 | 0.76463 / 0.76483 / 0.77040 / 0.76142 / 0.76621 | +0.00465 vs cat (0.76080) |
| LR trên binning quantile 16 bin one-hot (GAM cộng tính, C=0.05; dùng cả DATE) `lrbin_r1` | — | 0.75688 | 0.75415 / 0.75628 / 0.76097 / 0.75394 / 0.75919 | mô hình mới; Spearman vs rank-avg GBDT r1 = 0.926 |
| MLP sklearn (128,64) trên quantile-normal `mlp_r1` | — | 0.75482 | 0.75288 / 0.75536 / 0.75999 / 0.75105 / 0.75747 | mô hình mới; Spearman = 0.937 |
→ Lợi ích FE ở mức đầy đủ (+0.0054) ≈ 90% mức sàng lọc (IMP+RATE+TERM − BASE = +0.0061 ở lr 0.05) — chuyển giao tốt.

##### Ensemble (meta-model chấm trên fold chưa thấy; mọi OOF tầng 1 cùng folds rs=42)
| Cách | OOF | AUC từng fold | Chênh vs stack vòng 0 (0.76273) |
|---|---|---|---|
| rank-avg LGB/Cat/XGB r1 | 0.76751 | 0.76544 / 0.76772 / 0.77285 / 0.76362 / 0.76812 | +0.00477, 5/5 (vs lgb_r1 đơn: +0.00055, 5/5) |
| stack LR LGB/Cat/XGB r1 | 0.76750 | | +0.00479 (≈ rank-avg, đúng R10) |
| rank-avg r1 GBDT + lrbin + mlp | 0.76747 | | +0.00478 (rank-avg trọng số bằng nhau KHÔNG hấp thụ được mô hình yếu hơn) |
| stack LR r1 GBDT + lrbin + mlp | 0.76786 | | +0.00517 |
| **stack LR r1 GBDT + lrbin** (CHỌN) | **0.76790** | **0.76579 / 0.76811 / 0.77310 / 0.76412 / 0.76872** | **+0.00517, 5/5** |
| stack LR r1 GBDT + mlp | 0.76746 | | mlp thêm ≈ 0 |
| stack LR 6 mô hình vòng 0 + r1 GBDT + lrbin + mlp | 0.76800 | | +0.00015 vs (G1+D1), dưới ngưỡng R2, hệ số âm cho xgb/xgb_opt → BỎ |
Đóng góp lrbin trong stacking (G1+lrbin − G1): +0.0003 +0.0004 +0.0002 +0.0005 +0.0006 → **+0.0004, 5/5** (mức biên R2 nhưng ổn định → GIỮ).
MLP: +0.00002 khi đã có lrbin → BỎ. Hệ số stack: lgb_r1 0.292, cat_r1 0.319, xgb_r1 0.265, lrbin_r1 0.175.

**AUC OOF cuối vòng 1 = 0.76790** (vòng 0: 0.76273, **+0.00517**, dương 5/5 fold).
Submission ứng viên: `$S/loop/sub_round1.csv` (61,500 dòng, thứ tự = sample_submission, TARGET = rank chuẩn hóa của logit stack, không NaN). Chưa ghi vào repo.

##### Phương pháp ĐÃ dùng (bổ sung vòng 1 — Tác nhân 1 không đề xuất lại)
Điền khuyết EXT_SOURCE bằng LGB hồi quy + phần dư + tổng hợp EXT đã điền; lãi suất ngầm định theo kỳ hạn cố định; dựng lại kỳ hạn bằng mật độ lãi suất
(TERM_HAT/RATE_HAT/độ chắc chắn/overpay/tuổi khi trả xong); freq của "mã sản phẩm" round(C/A), round(C/GOODS); đặc trưng ngày/tuổi nghiệp vụ & trễ đổi hộ chiếu (thử, bỏ);
KNN-target-mean 50/200/500 lồng fold (thử, bỏ); LR trên binning quantile (GAM/WOE-like); MLP sklearn quantile-normal; stacking LR có chọn mô hình theo ablation ghép cặp.

##### Phát hiện bất ngờ
1. Lợi ích FE chuyển giao nguyên vẹn sang cả 3 thuật toán (+0.0054 LGB, +0.0054 XGB, +0.0047 Cat) — thông tin mới thật, không phải nhiễu của một mô hình.
2. KNN-target-500 (đặc trưng "rất mạnh" của đội vô địch Kaggle) lại GIẢM điểm ở đây: khi EXT đã điền khuyết, KNN không còn thông tin mới; mặt khác lệch phân phối inner(80%)/outer(100%) làm cây tin nhầm.
3. Đặc trưng ngày/hộ chiếu (AUC đơn biến trong nhóm tuổi 0.53–0.55) làm giảm điểm — tín hiệu quá yếu so với nhiễu khi thêm 12 cột.
4. TERM (kỳ hạn dựng lại) cộng thêm +0.0021 trên RATE thô → mật độ lãi suất chọn đúng kỳ hạn mang thông tin mà RATE_n thô chưa có.
5. Đa dạng mô hình: chỉ LR-binning giúp (+0.0004); MLP corr 0.937 nhưng AUC thấp hơn nên không thêm gì; mô hình vòng 0 không còn giá trị.

##### Bài học đã thực hiện (đối chiếu round1_agent2.md và round1_agent1.md)
| Mã | Hướng / bài học | Trạng thái |
|---|---|---|
| L1.1 (A2 §5.1) | Điền khuyết EXT + RATE_n vào bộ đặc trưng, chạy đủ LGB/Cat/XGB, rank-avg | ĐÃ LÀM (vòng 1): LGB 0.76696, XGB 0.76643, Cat 0.76545; rank-avg 0.76751 |
| L1.1b (A2 §5.1 mở rộng) | Kỳ hạn bổ sung n=6,10,18,30,42; lọc RATE theo loại HĐ | ĐÃ LÀM một phần: dùng trong TERM (11 kỳ hạn), Revolving → NaN |
| L1.1c (A2 §5.1 mở rộng) | RATE tính với AMT_GOODS_PRICE làm gốc; điền khuyết bằng nhiều thuật toán | CHƯA LÀM |
| L1.2 (A2 §5.2, A1 #2) | KNN target-mean lồng fold (k=50/200/500, 2 không gian) | ĐÃ LÀM: −0.0005…−0.0007, 1/4 → BỎ. KNN làm mô hình riêng: CHƯA LÀM |
| L1.3 (A2 §5.3) | Mô hình khác họ: LR binning/WOE, MLP | ĐÃ LÀM: lrbin 0.75688 (giữ, +0.0004 trong stack), MLP 0.75482 (bỏ, ≈0) |
| L1.4 (A2 §5.4) | Multi-seed cấu hình cuối | ĐÃ LÀM cho LGB (3 seed); XGB/Cat 1 seed (giới hạn thời gian) |
| L1.5 (A2 §5.5) | Pseudo-label test; mô hình riêng nhóm Revolving/hưu trí | CHƯA LÀM |
| R1–R3 (A2 §4) | So sánh ghép cặp, ngưỡng +0.0005 & ≥4/5, sàng lọc fold0/1×2 seed | ĐÃ ÁP DỤNG cho mọi quyết định giữ/bỏ |
| R4 | Đặc trưng dùng TARGET tính trong fold | ĐÃ ÁP DỤNG (KNN lồng fold) |
| R5 | Không tune Optuna nữa | ĐÃ ÁP DỤNG (dùng tham số vòng 0) |
| R7 | Thêm theo nhóm có giả thuyết, kiểm riêng | ĐÃ ÁP DỤNG (IMP/RATE/TERM/DATE/KNN kiểm từng nhóm) |
| R8 | Không cắt đặc trưng theo gain | ĐÃ ÁP DỤNG |
| R9/R10 | Kiểm corr trước; stacking chỉ khi có mô hình corr < 0.95, CV lồng nhau | ĐÃ ÁP DỤNG (lrbin corr 0.926 → stacking > rank-avg) |
| R11 | Đổi đặc trưng → chạy lại cả 3 GBDT | ĐÃ LÀM |
| R12 | Lưu oof/pt tên riêng, ghi số seed | ĐÃ LÀM (`oof_/pt_{lgb,xgb,cat,lrbin,mlp}_r1.npy`) |
| A1 #1 | Tái dựng kỳ hạn & lãi suất + mã sản phẩm | ĐÃ LÀM (RATE + TERM): +0.0025 / +0.0021 sàng lọc. Bản "posterior prior×mật độ"/KDE: CHƯA LÀM |
| A1 #2 | KNN-target-mean 500 | ĐÃ LÀM → BỎ (xem L1.2) |
| A1 #3 | Điền khuyết EXT bằng mô hình + phần dư | ĐÃ LÀM (IMP): +0.0024 sàng lọc |
| A1 #4 | Đặc trưng ngày/tuổi nghiệp vụ, hộ chiếu 20/45 | ĐÃ LÀM → BỎ (−0.0004, 1/4) |
| A1 #5 | RealMLP / TabM (pytabkit, cần torch ~3 GB) | CHƯA LÀM (đã thay bằng MLP sklearn, không giúp) |
| A1 #6 | Target encoding OOF cho tổ hợp danh mục (ORG×OCC, PRODUCT×loại HĐ) | CHƯA LÀM |
| A1 #7 | Chọn đặc trưng Null Importance | CHƯA LÀM |
| A1 #8 | Ghép khách hàng lặp lại (uid từ hiệu DAYS) | CHƯA LÀM |
| A1 #9 | LightGBM DART / extra_trees / linear_tree để đa dạng | CHƯA LÀM |
| A1 #10 | TabICL / TabPFN | KHÔNG THỰC THI ĐƯỢC (O(n²), cần GPU, 171k dòng × CPU 4 nhân) |
| A1 (không khuyến nghị) | Adversarial validation | ĐÃ LÀM bởi A2 (AUC 0.512) — chỉ chẩn đoán |

---

## VÒNG 2

### Tác nhân 1 — Nghiên cứu phương pháp mới (`loop/round2_agent1.md`)
STATUS: NEW_METHODS_FOUND

#### VÒNG 2 — Tác nhân 1 (Nghiên cứu): phương pháp mới

Đã đọc: `rules.md`, `history.md` (mốc OOF 0.76790), `round1_agent1.md`, `fe_r1.py`, `round2_agent2.md`.
KHÔNG tính là của tôi: target encoding trong fold của PRODUCT=round(C/A,2) (Tác nhân 2, L2.1/L2.2).
Script và dữ liệu trung gian: `$S/loop/a1r2/` (`h2.py`, `t1.py`, `t2.py`, `mkprof.py`, `screen2.py`, `x*.py`, `em_term.py`, log `log_screen2/3/4.txt`).
Mạng: tìm kiếm web vòng này không cho nguồn mới có giá trị; phát hiện chính đến từ phân tích dữ liệu.

##### Tóm tắt một dòng
Tìm được MỘT nhóm đặc trưng mới có bằng chứng mạnh: **"hồ sơ quần thể của sản phẩm vay" (không dùng nhãn, tính trên train+test)** — sàng lọc LGB theo đúng R3: **+0.0026 … +0.0031, 4/4 dương** trên REF IMP+RATE+TERM (ngang mức của cả IMP hoặc RATE ở vòng 1). Các hướng khác tôi thử đều ≈ 0 (ghi rõ bên dưới để khỏi lặp lại).

##### Hai công cụ đo đã dùng
1. **Harness "thông tin còn lại"** (`h2.py`): LGB nông (depth 3, lr 0.03) boosting TIẾP từ logit của stack vòng 1 (init_score = OOF stack đã hiệu chuẩn), chỉ trên nhóm đặc trưng ứng viên, 5 fold rs=42; báo chênh AUC ở 100 vòng. Đối chứng: 5 cột nhiễu = −0.00006; cột thô C/A, C/G, RATE_HAT, TERM_HAT = −0.0002; EXT đã điền = −0.0006 → harness không tự sinh điểm dương.
2. **Sàng lọc R3** (`screen2.py`, y hệt cấu hình sàng lọc vòng 1: fold 0/1 × 2 seed, LGB lr 0.05 depth 4). REF tái lập đúng 0.76531 như history.md.

##### Bảng xếp hạng

| Hạng | Phương pháp | Bằng chứng | Mức tăng OOF kỳ vọng | Chi phí | Rủi ro |
|---|---|---|---|---|---|
| 1 | Hồ sơ quần thể của sản phẩm vay (target-free, train+test) làm cột cho GBDT | sàng lọc R3: +0.0026…+0.0031, 4/4 (6 biến thể) | +0.002 … +0.003 ở mô hình đơn và ở stack | vài giây + chạy lại 3 GBDT | trùng tín hiệu với TE PRODUCT của A2 (không cộng dồn) |
| 2 | "Hồ sơ theo dự đoán của mô hình": trung bình PD dự đoán (vòng 1) theo sản phẩm, gồm cả dòng test | chưa đo; suy từ #1 | 0 … +0.001 thêm trên #1 | phút, cần lồng fold | rò rỉ gián tiếp nếu không lồng |
| 3 | EBM / GA2M (`interpret`) làm mô hình tầng 1 khác họ | chưa đo; lrbin cộng tính đã cho +0.0004 | 0 … +0.0005 ở stack | 20–60 phút | có thể corr quá cao với GBDT |
| — | Đã thử, KHÔNG có tín hiệu (đừng làm lại) | xem mục "Kết quả âm" | ≈ 0 | — | — |

---

##### 1. Hồ sơ quần thể của sản phẩm vay (product-population profile)

**Ý tưởng.** Home Credit định giá theo rủi ro: mỗi sản phẩm (kỳ hạn, lãi suất) được bán cho một phân khúc khách hàng do mô hình nội bộ của ngân hàng chọn. Vậy "ai mua sản phẩm này" (EXT trung bình, tuổi trung bình, thâm niên trung bình… của TẤT CẢ khách cùng sản phẩm trong train+test) là ước lượng của hạng rủi ro mà ngân hàng gán cho sản phẩm đó. Khác với TE của A2: KHÔNG dùng TARGET → không cần lồng fold, không có lệch phân phối inner/outer (nguyên nhân làm KNN-target thất bại ở vòng 1), và dùng được thêm 61.5k dòng test (+36% dữ liệu cho thống kê nhóm).
Khác với "group-mean diff theo ORG/OCC" đã dùng ở vòng 0: khóa nhóm ở đây là SẢN PHẨM, và chỉ khóa này có tín hiệu (xem bảng khóa).

**Bằng chứng 1 — harness thông tin còn lại (chênh AUC so với stack 0.76792, 100 vòng; số trong ngoặc = số fold dương/5):**

| Khóa nhóm | Số mức | Hồ sơ target-free | TE lồng fold (đối chiếu) |
|---|---|---|---|
| round(C/A, 2) (cash) | 2,204 | **+0.00129 (5)** | +0.00111 (5) |
| TERM_HAT × round(RATE_HAT, 0.5%) | 682 | **+0.00114 (5)** | +0.00110 (5) |
| round(A/C, 4) | 815 | +0.00103 (5) | +0.00111 (5) |
| round(C/A, 1) | 299 | +0.00095 (5) | +0.00091 (5) |
| round(C/A, 3) | 11,084 | +0.00084 (5) | +0.00052 (5) |
| C/A × gói bảo hiểm k | 3,298 | +0.00090 (5) | +0.00066 (5) |
| C/A × REGION_RATING | 3,489 | +0.00082 (5) | +0.00060 (4) |
| AMT_ANNUITY chính xác | 12,981 | +0.00030 (5) | +0.00022 (5) |
| REGION_POPULATION_RELATIVE (81 vùng) | 81 | −0.00002 | −0.00007 |
| gói bảo hiểm k, AMT_GOODS_PRICE, AMT_CREDIT, AMT_INCOME, ORG×OCC, REGION×ORG, năm sinh×giới | — | −0.0002 … −0.0006 (0–2) | ≤ 0 |

- Hồ sơ + TE cùng lúc (khóa C/A 2 chữ số): +0.00129 = bằng hồ sơ một mình → **cùng một tín hiệu với TE của A2, không cộng dồn**.
- Từng cột một (khóa C/A 2 chữ số): trung bình EXT_SOURCE_3 **+0.00116**, EXTF_MEAN +0.00093, EXT_1_PRED +0.00067, tuổi +0.00061, DAYS_EMPLOYED +0.00054, EXT_2 +0.00052, tỉ lệ nam +0.00026; thu nhập, số tiền vay, tỉ lệ có bảo hiểm, số dòng của nhóm, (EXT cá nhân − EXT nhóm) ≤ 0.

**Bằng chứng 2 — sàng lọc R3 (LGB, thêm cột vào REF IMP+RATE+TERM = 0.76531):**

| Biến thể | Số cột | TB 4 lần | Chênh ghép cặp | Dương |
|---|---|---|---|---|
| TR16: khóa TERM×RATE(0.5%), 16 cột (`prof_tr.pkl`) | 16 | 0.76844 | **+0.00312** | 4/4 |
| RATEn: 5 khóa round(RATE_n, 0.5%) n=12/24/36/48/60 × {e, e3, age, emp} (`pv_RATEn.pkl`) | 20 | 0.76838 | **+0.00306** | 4/4 |
| RATEn + TR16 | 36 | 0.76819 | +0.00287 | 4/4 |
| TRmulti: khóa TERM×RATE ở 3 độ mịn (2% / 0.5% / 0.1%) | 24 | 0.76812 | +0.00281 | 4/4 |
| TR + CA2 (8 cột mỗi khóa) | 16 | 0.76802 | +0.00271 | 4/4 |
| RATEn + TR16 + CA2 | 44 | 0.76799 | +0.00268 | 4/4 |
| TRwide: khóa TERM×RATE, 43 cột (thêm tỉ lệ nghề, học vấn, nhà ở…) | 43 | 0.76795 | +0.00264 | 4/4 |
| TRcore: khóa TERM×RATE, 8 cột | 8 | 0.76793 | +0.00261 | 4/4 |
| CA2S: khóa round(C/A,2), 8 cột | 8 | 0.76725 | +0.00194 | 4/4 |

Kết luận: mọi biến thể đều dương 4/4; **bão hòa quanh +0.003** — thêm khóa hay thêm cột không giúp nữa (chênh giữa các biến thể 0.0005 nằm trong nhiễu). Chọn bản gọn.
Lưu ý: harness đo trên stack (+0.0013) còn sàng lọc đo trên LGB đơn lr 0.05 (+0.003); mức tăng ở stack cuối chưa đo trực tiếp, kỳ vọng nằm giữa hai số.

**Phác thảo cài đặt (đúng mã đã chạy, `mkprof.py`):**
```python
# df = load_r1() (train+test gộp); KHÔNG dùng y
cash = df.NAME_CONTRACT_TYPE=='Cash loans'
core = DataFrame(e=EXTF_MEAN, e2=EXT_SOURCE_2, e3=EXT_SOURCE_3, e1p=EXT_SOURCE_1_PRED,
                 age=DAYS_BIRTH, emp=DAYS_EMPLOYED (365243→NaN), male=(CODE_GENDER=='M'))
def profile(key, cols, pre, minc=30):
    cnt = số dòng của mỗi khóa (train+test)
    out[pre+'_'+c] = groupby(key)[c].transform('mean')     # mean bỏ qua NaN
    out[pre+'_estd'] = groupby(key).e.transform('std')
    out[cnt < minc] = NaN                                   # nhóm hiếm → NaN
# Phương án A (khuyến nghị, 8–16 cột): key = TERM_HAT*100000 + round(RATE_HAT*200); Revolving → một khóa riêng (-5)
# Phương án B (không phụ thuộc TERM_HAT, 20 cột): với n ∈ {12,24,36,48,60}: key_n = round(RATE_n*200); cột {e, e3, age, emp}
```
File dựng sẵn (232,702 dòng, đúng thứ tự `load_r1()`): `$S/loop/a1r2/prof_tr.pkl` (A-16 cột), `pv_TRcore.pkl` (A-8 cột), `pv_RATEn.pkl` (B), `prof_ca2s.pkl`.

**Việc Tác nhân 3 cần làm:**
1. Thêm nhóm PROF (A hoặc B, một trong hai) vào `fe_r1.py`, chạy lại LGB/XGB/Cat đủ 5 fold (R11), rồi stack với lrbin.
2. So sánh có/không có TE PRODUCT của A2 ở tầng meta: vì cùng tín hiệu, khi GBDT đã có PROF thì cột TE ở meta nhiều khả năng mất tác dụng (R16) — báo cả hai cấu hình.
3. Đưa PROF vào cả lrbin (mô hình cộng tính hưởng lợi trực tiếp từ cột "hạng rủi ro của sản phẩm").

**Rủi ro.** Không rò nhãn (không dùng TARGET ở đâu). Dùng đặc trưng của dòng test là hợp lệ theo ràng buộc. Rủi ro thực: mức tăng ở 5 fold đầy đủ lr 0.01 nhỏ hơn sàng lọc (vòng 1 chuyển giao ~90%), và trùng với TE của A2.

##### 2. Hồ sơ theo dự đoán của mô hình (chưa đo)
Thay vì trung bình EXT, lấy trung bình logit dự đoán của mô hình vòng 1 trên mọi dòng cùng sản phẩm (train + test). Đây là dạng "self-training" ở mức nhóm: tận dụng test không nhãn, mượt hơn TE vì mỗi dòng đóng góp một xác suất thay vì nhãn 0/1.
Bắt buộc lồng fold: với fold ngoài k, dự đoán cho mọi dòng (train trong fold: inner OOF; valid + test: mô hình học trên Tk) rồi mới gộp theo sản phẩm; nếu dùng OOF 1 tầng thì nhãn fold valid lọt gián tiếp vào đặc trưng. Kỳ vọng 0…+0.001 thêm trên #1 (cùng nguồn tín hiệu), chi phí cao hơn nhiều → chỉ thử sau #1.

##### 3. EBM / GA2M (chưa đo)
Explainable Boosting Machine (`interpret`; wheel `interpret_core-0.6.9` đã tải được ở `$S/loop/a1r2_pip/`): mô hình cộng tính + tương tác cặp, boosting vòng tròn từng đặc trưng. lrbin (cộng tính thuần, 0.7569) đã cho +0.0004 trong stack; EBM có tương tác cặp nên AUC đơn kỳ vọng ~0.760–0.764. Chỉ giữ nếu Spearman với rank-avg GBDT < 0.95 (R9). Kỳ vọng 0…+0.0005. Cấu hình gợi ý: `ExplainableBoostingClassifier(interactions=20, max_bins=256, outer_bags=4, n_jobs=4)`, cùng folds rs=42.

##### Kết quả âm (đã đo, đừng lặp lại)
| Hướng | Kết quả |
|---|---|
| Dựng kỳ hạn bằng EM (hỗn hợp tiềm ẩn, có điều kiện theo gói bảo hiểm; `em_term.py`, 8 cột TERM_EM/RATE_EM/POST_MAX/TERM_EXP/RATE_EXP/TERM_ENT/INS_K/INS_PER_M) | sàng lọc R3: **−0.00001, 2/4** → không giúp. Liên quan L2.3 của A2 và phần "posterior" còn dở của A1 #1 vòng 1 |
| Phí bảo hiểm ẩn: (C/G − 1) là bội của 1.32% ở 89.8% khoản cash có C>G (k = 3…50); tỉ lệ vỡ nợ theo k từ 2.6% (k=3) đến 18.3% (k=32) | thông tin đã nằm trong CREDIT_GOODS: harness one-hot/hồ sơ/TE theo k đều ≤ 0 |
| Kỳ hạn suy từ gói bảo hiểm (độ rời rạc của lãi suất theo từng k; `x9.py`) | không phân định được kỳ hạn (HHI gần bằng nhau giữa các n) |
| Độ tròn của số tiền (annuity/thu nhập/giá hàng chia hết 4.5k/9k/22.5k/45k/225k) | harness: −0.0001 … 0 |
| Số lần lặp lại của giá trị EXT_SOURCE_1/2/3 | ≈ 0; hai dòng trùng EXT_SOURCE_2 KHÔNG phải cùng người (DAYS_BIRTH trùng 0%) |
| Vùng (REGION_POPULATION_RELATIVE = 81 mã vùng) làm khóa nhóm / TE | ≈ 0 |
| Phương pháp bảng mới 2025–2026 (TabPFN-3, TabICLv2, LimiX…) | đều cần GPU ở cỡ 171k dòng; TabArena ghi nhận TabM/ModernNCA giảm chất lượng khi chạy CPU có giới hạn giờ → không khả thi ở đây |

##### Nhận định trung thực về 0.80
- Sau vòng 2, ước lượng hợp lý: 0.7679 + (0.0015 … 0.003 từ tín hiệu sản phẩm, PROF và TE không cộng dồn) + (0 … 0.0005 từ đa dạng mô hình/multi-seed) ≈ **0.770 – 0.771**.
- Mọi phép thăm dò "thông tin còn lại" ngoài khóa sản phẩm đều cho ≈ 0 → dữ liệu 33 cột gần cạn. Khoảng cách tới 0.80 còn ~0.03, gấp ~10 lần mức tăng của một vòng tốt.
- **0.80 không đạt được bằng phương pháp hợp lệ với dữ liệu này.** Sau khi #1 được cài, tôi dự đoán vòng 3 sẽ rất khó tìm thêm phương pháp mới có bằng chứng; #2 và #3 là các hướng chưa đo còn lại.

### Tác nhân 2 — Rút bài học từ Vòng 1 (`loop/round2_agent2.md`)
#### VÒNG 2 — Tác nhân 2 (Rút kinh nghiệm từ vòng 1)

Nguồn: `rules.md`, `history.md` (vòng 0 + vòng 1), `round1_agent2.md`, `fe_r1.py`, `run_r1.py`, `div_r1.py`, `ens_final_r1.py`, `log_r1_*.txt`, OOF `*_r1.npy`.
Phân tích nhẹ của tôi: `$S/a2r2/b1.py … b6.py`, log `$S/a2r2/b1.log … b6.log`. Không sửa repo.
Mốc: stack LR (lgb_r1 + cat_r1 + xgb_r1 + lrbin_r1) = **0.76790** (tôi dựng lại: 0.76790; stack vòng 0 dựng lại: 0.76271).

Chú thích độ tin cậy:
- [ĐO] = tôi đo trực tiếp trên OOF 5 fold rs=42. [SUY] = suy luận, chưa kiểm chứng.
- Phần sàng lọc "TE làm đặc trưng cho LGB" KHÔNG chạy xong (máy bị tranh CPU với tác nhân khác, 2 lần hết giờ) → chưa có số, đừng coi là đã kiểm.

---------------------------------------------------------------------------------------------------
##### 1. Thất bại vòng 1 & nguyên nhân

| Thử nghiệm | Kết quả | Nguyên nhân |
|---|---|---|
| DATE (12 cột tuổi tại mốc, trễ đổi hộ chiếu…) | −0.0004 (1/4), −0.0009 khi có TERM | [SUY] Đều là tổ hợp tuyến tính của DAYS_* mà cây đã phân nhánh được; AUC đơn biến 0.53–0.55. Thêm 12 cột yếu cùng lúc với feature_fraction 0.2–0.3 làm loãng xác suất chọn cột mạnh (EXTF_MEAN chiếm 20% gain). Vi phạm tinh thần R7 (nhóm quá to, tín hiệu quá yếu). |
| KNN-target 50/200/500 làm đặc trưng | −0.0005…−0.0007 (1/4) | (a) Không gian KNN = {EXT đã điền, C/A, tuổi} → trùng thông tin với EXTF_* (đã 35.5% gain [ĐO]). (b) Lệch inner(80%)/outer(100%) giữa train và valid của đặc trưng. (c) KNN đơn biến 0.73 < mô hình 0.767 nên chỉ là bản "mờ" của chính mô hình. |
| MLP sklearn (128,64) | 0.75482; thêm vào stack ≈ 0 | [ĐO] corr với lgb_r1 0.932 (đủ khác) nhưng kém lgb_r1 **đều −0.010 ở MỌI nhóm**, và với lrbin corr 0.865. Mô hình vừa yếu hơn 0.012 vừa không giỏi hơn ở nhóm nào → stack không có gì để lấy. MLP sklearn không có embedding, không xử lý NaN (điền trung vị), không early-stop theo AUC. |
| LR-binning (giữ, nhưng lợi ít: +0.0004) | 0.75688 | [ĐO] lrbin − lgb_r1 theo nhóm: −0.0022 (TERM 24) … −0.018 (tuổi > 60), −0.015 (hưu trí). Không nhóm nào lrbin thắng → lợi ích chỉ đến từ trung bình hóa nhiễu, không phải bù điểm mù. Mô hình cộng tính không có tương tác EXT×tuổi×sản phẩm. |
| 6 mô hình vòng 0 thêm vào stack | +0.00015, hệ số âm | [ĐO] corr lgb(vòng 0)–lgb_r1 = 0.975 nhưng yếu hơn 0.0054 và là TẬP CON thông tin của r1 → meta cho hệ số âm (đang khớp nhiễu). Bài học: mô hình cũ trên bộ đặc trưng con KHÔNG phải nguồn đa dạng. |
| Rank-avg bằng trọng số khi thêm lrbin/mlp | 0.76747 < 0.76751 | Rank-avg đều chỉ hợp khi các mô hình ngang sức; mô hình yếu hơn 0.01 cần trọng số nhỏ (stack cho lrbin 0.175). |
| XGB/Cat chỉ 1 seed | — | Giới hạn thời gian (XGB 838 s, Cat ~17 phút/seed). Còn nợ (L1.4). |

Thất bại phương pháp luận (không phải kết quả):
- Sàng lọc chỉ dùng LGB. Một đặc trưng dùng nhãn (KNN) bị loại ở dạng "cột cho cây" nhưng chưa thử ở tầng meta — trong khi ở tầng meta, tín hiệu dùng nhãn kiểu khác (TE mã sản phẩm, mục 3) lại cho lợi ích lớn nhất hiện có.
- TERM được chấp nhận nguyên khối 8 cột; không ai kiểm TERM_HAT có đúng không (xem 3.4).

##### 2. Mẹo từ thành công vòng 1

1. **IMP hiệu quả vì nó chuyển "thiếu" thành "ước lượng"** — và EXT là 80% mô hình. [ĐO, fold 0, lr 0.05, AUC 0.76417]:
   hoán vị đồng thời 15 cột IMP → −0.156; mọi cột EXT (cũ + IMP) → **−0.208**; trong khi RATE+TERM → −0.017, tuổi → −0.013, ORG/OCC → −0.005, AMT cũ → −0.005.
   Gain: IMP 35.5%, RATE 5.6%, TERM 7.9%, 80 cột cũ 51%. `EXTF_MEAN` một mình 20.3% gain.
   Lợi ích theo nhóm khớp giả thuyết [ĐO, stack r1 − stack r0]: thiếu 2 EXT **+0.0086** (0.7401→0.7487), thiếu EXT3 **+0.0087**, thiếu 1 EXT +0.0044, đủ 3 EXT +0.0035.
2. **Trong IMP, phần có ích là `_PRED` và tổng hợp `EXTF_*`, không phải phần dư**: hoán vị 3 cột `_PRED` −0.0087; 3 cột `_RES` chỉ −0.0008 (gain 1%).
   → `_PRED` là một "điểm tín dụng nhân khẩu học" học KHÔNG nhãn (EXT1_PRED chủ yếu là tuổi/giới/việc làm). Hệ quả: lợi cả cho người ĐỦ EXT (+0.0035).
3. **RATE/TERM hiệu quả vì là thông tin về SẢN PHẨM, độc lập với EXT**: Spearman(RATE_HAT, EXTF_MEAN) = −0.125 [ĐO] → gần trực giao nên cộng dồn với IMP (+0.0024 + +0.0025 ≈ +0.0040).
   Lợi ích lớn nhất ở TERM_HAT=60 (**+0.0119**), tuổi 50–60 (+0.0084), công chức (+0.0096).
4. **Cộng dồn ba tầng khác nguồn**: FE (+0.0054 mỗi GBDT) + ensemble 3 thuật toán (+0.00055) + mô hình khác họ ở meta (+0.0004). Tương quan r1 [ĐO]: lgb–xgb **0.996** (gần như trùng), lgb–cat 0.978, GBDT–lrbin 0.91–0.93. XGB gần như thừa cạnh LGB; CatBoost mới là nguồn đa dạng trong GBDT.
5. **Sàng lọc 4 lần chấm ghép cặp dự báo đúng kết quả đầy đủ** (≈ 90% chuyển giao) → tiếp tục tin quy trình R3.

##### 3. Phát hiện bất ngờ (đều [ĐO], script `a2r2/`)

1. **Target encoding trong fold của "mã sản phẩm" là tín hiệu mà cả 4 mô hình đang bỏ sót — lớn nhất tìm được vòng này.**
   Khóa `PRODUCT = round(AMT_CREDIT/AMT_ANNUITY, 2)` (2,119 mức), TE làm mượt `(sum + α·prior)/(count + α)`.
   Thêm logit(TE) làm cột thứ 5 của meta-LR, lồng NGHIÊM NGẶT (TE của phần train meta tính bằng inner 5-fold trong phần train; TE của fold chấm tính từ toàn bộ phần train; nhãn fold chấm không bao giờ được dùng) — `b4.py`, `b5.py`:

   | Cột thêm vào meta | Chênh từng fold vs 0.76790 | TB | Dương |
   |---|---|---|---|
   | TE PRODUCT (2 chữ số), α=10 | +0.00138 +0.00159 +0.00121 +0.00190 +0.00165 | **+0.00155** | 5/5 |
   | TE PRODUCT, α=5 / 20 / 100 / 300 | | +0.00153 / +0.00149 / +0.00108 / +0.00072 | 5/5 |
   | TE PRODUCT 3 chữ số, α=20 / 100 | | +0.00110 / +0.00067 | 5/5 |
   | TE (TERM_HAT × RATE_HAT làm tròn 1%), α=20 / 100 | | +0.00125 / +0.00107 | 5/5 |
   | TE (AMT_CREDIT × AMT_ANNUITY chính xác), α=20 | | +0.00084 | 5/5 |
   | TE PRODUCT + TE TERM×RATE cùng lúc (α=100) | | +0.00111 | 5/5 (KHÔNG cộng dồn → cùng một tín hiệu) |
   | TE ORG×OCC / ORG×INC×EDU / CREDIT×GOODS | | −0.00005 / −0.00007 / +0.00003 | vô ích |

   → ước lượng stack mới ≈ **0.7694** chỉ với một cột, gấp ~4 lần đóng góp của lrbin. α nhỏ (5–20) tốt hơn α lớn.
   TE PRODUCT đơn biến AUC 0.669, Spearman với stack chỉ 0.49. Tín hiệu nằm ở sản phẩm phổ biến: nhóm PRODUCT có 200–1000 dòng AUC TE 0.690, 30–200 dòng 0.665; < 30 dòng ≈ 0.51–0.56 (nhiễu).
   Vì sao GBDT bỏ sót [SUY]: tỉ lệ vỡ nợ theo mã sản phẩm KHÔNG đơn điệu theo C/A (mỗi giá trị C/A rời rạc là một sản phẩm/chương trình riêng), cây nông depth 4 + feature_fraction 0.2 không cắt nổi ~2,000 khoảng hẹp; `PRODUCT_FREQ` chỉ cho biết độ phổ biến, không cho biết rủi ro.
   CẢNH BÁO: mới kiểm ở tầng META. Dạng "cột TE cho GBDT" chưa có kết quả (chạy không xong).
2. **Meta thông minh hơn KHÔNG giúp**: meta-LR thêm chỉ báo nhóm (thiếu 2 EXT, hưu trí, Revolving, > 60 tuổi, nam) và tích logit×nhóm: −0.00033, **0/5**. Meta LGB 4 lá trên logit + 6 biến gốc: −0.00040, 1/5.
   → Không tồn tại "trọng số theo nhóm" có lợi; cải thiện phải đến từ tín hiệu MỚI ở tầng 1 hoặc cột mới ở meta, không phải từ meta phức tạp hơn.
3. **Nhóm yếu vẫn là các nhóm cũ, khoảng cách hầu như không thu hẹp** (AUC stack r1): tuổi > 60 **0.7274**, hưu trí 0.7405, tuổi < 30 0.7397, thiếu 2 EXT 0.7487, thiếu EXT3 0.7517; so với đủ EXT 0.7770.
   Theo sản phẩm: TERM_HAT 48 → **0.7315** (17.7% dữ liệu), 10 → 0.7371, 18 → 0.7398 (tỉ lệ vỡ nợ 13.8%, cao nhất), 42 → 0.7465; trong khi 12 → 0.7946, 24 → 0.7833, 54 → 0.7790.
   Nam chỉ được +0.0031 từ vòng 1, nữ +0.0068. Revolving +0.0037 (không có RATE/TERM).
4. **TERM_HAT đáng ngờ, mô hình gần như không dùng nó**: gain của `TERM_HAT` = 0.05% (thấp nhất trong 28 cột mới); cây dùng `RATE_36`, `RATE_HAT`, `TERM_DENS_RATIO`, `AGE_END_LOAN`.
   58% khoản Cash có RATE_HAT > 35%/năm, trung vị 40%, p90 = 55% (sát trần lọc 60%) → thuật toán mật độ thiên về kỳ hạn DÀI/lãi CAO. Chỉ 18% dòng có TERM_DENS_RATIO > 2 (kỳ hạn rõ ràng), 64.5% có tỉ số ≤ 1.2 (gần như hòa).
   Dấu của RATE_HAT trong từng kỳ hạn không nhất quán: AUC(RATE_HAT→vỡ nợ) = 0.47 (n=12), 0.455 (24), 0.47 (36) nhưng 0.61 (54), 0.61 (60).
   → TERM hiện đóng vai "mã băm sản phẩm" hơn là kỳ hạn thật; đây cũng là lý do TE theo PRODUCT (khóa thô, không cần đoán kỳ hạn) lại mạnh.
5. **Lỗi lớn nhất là nhiễu không giảm được theo đặc trưng hiện có**: 683 người vỡ nợ (4.9% số dương) bị xếp ở 25% an toàn nhất — trung vị EXTF_MEAN 0.636 (cao hơn cả người KHÔNG vỡ nợ: 0.537), tuổi 48, 78% nữ, 24% hưu trí. Ngược lại 3,164 người không vỡ nợ ở top 3% rủi ro: EXTF_MEAN 0.275, tuổi 35.6, 55% nam, 89% học vấn thấp, 98% Cash.
   Mô hình đã "đúng theo EXT"; cái sai là những ca EXT nói ngược thực tế. Thập phân vị rủi ro cao nhất chứa 34.5% số vỡ nợ (tỉ lệ 27.9%), 30% trên cùng chứa 66.3%.
6. **Khách hàng lặp lại gần như không có**: khóa (giới, DAYS_ID_PUBLISH−DAYS_BIRTH, DAYS_REGISTRATION−DAYS_BIRTH) cho 9,650/232,702 dòng trùng (4,701 nhóm), nhưng thêm điều kiện EXT1 bằng nhau chỉ còn 124 dòng; trong train, nhóm trùng có tỉ lệ vỡ nợ 6.2% (thấp hơn nền 8.1%), 247/2,675 nhóm lẫn nhãn. → phần lớn là trùng ngẫu nhiên; hướng "uid" (A1 #8) kỳ vọng ≈ 0.

##### 4. Cách ghép kỹ thuật & tối ưu logic cho vòng 2 — quy tắc cho Tác nhân 3

Giữ nguyên R1–R12 của vòng 1. Bổ sung:

- **R13 (đặc trưng dùng nhãn → thử ở META trước).** Mọi tín hiệu dùng TARGET (TE, KNN, WOE) kiểm ở tầng meta trước (vài giây, không huấn luyện lại GBDT), theo đúng mẫu lồng nghiêm ngặt của `a2r2/b4.py` (`nested_te` + `metaF`). Chỉ khi meta dương ≥ 4/5 mới thử làm cột cho GBDT.
- **R14 (lồng TE đúng cách).** Với mỗi fold ngoài (a = train, b = valid): TE cho a bằng inner 5-fold TRONG a; TE cho b từ toàn a; TE cho test từ toàn train (hoặc trung bình 5 bản theo fold). Không bao giờ dùng TE "OOF 5 fold toàn cục" làm đặc trưng huấn luyện rồi chấm trên chính các fold đó. α chọn trong {5, 10, 20} — đừng dò mịn hơn (chênh giữa α 5/10/20 chỉ 0.00006).
- **R15 (một tín hiệu = một cột).** TE PRODUCT và TE TERM×RATE không cộng dồn (+0.00111 khi gộp < +0.00149 riêng) → chọn MỘT khóa (PRODUCT 2 chữ số, α=10). Đừng thêm 5 biến thể TE cùng lúc.
- **R16 (khi đưa TE vào GBDT).** Nếu thử: so ghép cặp với REF IMP+RATE+TERM theo R3; nếu GBDT đã hấp thụ TE thì cột TE ở meta sẽ mất tác dụng → báo cáo CẢ HAI cấu hình (TE chỉ ở meta / TE trong GBDT) và chọn theo OOF stack cuối, không theo OOF mô hình đơn.
- **R17 (không làm meta phức tạp).** Meta = LR trên logit, chuẩn hóa trong fold. Đã đo: tương tác nhóm −0.0003 (0/5), meta LGB −0.0004 (1/5). Không thử lại.
- **R18 (ensemble).** Không đưa mô hình là tập con thông tin của mô hình mạnh hơn (mô hình vòng 0, MLP sklearn). Điều kiện nhận mô hình mới vào stack: corr < 0.95 với rank-avg GBDT VÀ (AUC ≥ 0.760 HOẶC thắng GBDT ở ít nhất một nhóm ≥ 5% dữ liệu). XGB và LGB trùng 0.996 → nếu thiếu thời gian, bỏ multi-seed XGB trước, ưu tiên seed cho CatBoost/LGB.
- **R19 (nhóm đặc trưng nhỏ).** Mỗi nhóm thử ≤ 5 cột; nhóm lớn hơn phải tách ra sàng lọc từng phần (bài học DATE 12 cột, TERM 8 cột trong đó TERM_HAT gain 0.05%).
- **R20 (báo cáo theo nhóm).** Mỗi vòng in lại bảng AUC theo nhóm (`a2r2/b1.py`, chạy < 1 phút khi máy rảnh) cho stack mới: thiếu EXT, hưu trí, tuổi, TERM_HAT — để biết lợi ích đến từ đâu.
- **R21 (tài nguyên).** Máy 4 nhân đang bị nhiều tác nhân dùng chung: một lần sàng lọc LGB 35 s có thể thành > 5 phút. Không chạy song song quá 1 tiến trình `n_jobs=4`; ghi log `flush=True` từng bước.

Thứ tự ghép đề nghị cho vòng 2: (1) meta + TE PRODUCT → chốt mốc mới; (2) sàng lọc TE/sửa TERM trong GBDT; (3) nếu bộ đặc trưng đổi → chạy lại 3 GBDT (R11) rồi stack lại với lrbin + TE; (4) multi-seed cuối cùng.

##### 5. Chẩn đoán điểm nghẽn & hướng ưu tiên

**Điểm nghẽn vẫn là THÔNG TIN, cụ thể hơn: (i) tín hiệu theo mã sản phẩm chưa được khai thác đúng dạng; (ii) nhóm thiếu EXT / hưu trí / già / trẻ không có nguồn thông tin thay thế.**
- Mô hình/meta: bão hòa (lgb–xgb 0.996; meta phức tạp hơn âm).
- Đánh giá: ổn định; mọi kết luận trên đều 5/5 fold.
- Phần sai còn lại chủ yếu là ca "EXT cao mà vẫn vỡ nợ" → không sửa được bằng 33 cột. Trần thực tế ước ~0.770–0.772; **0.80 không khả thi hợp lệ** (chưa có bằng chứng nào ngược lại sau 2 vòng).

**Hướng ưu tiên (theo kỳ vọng lợi ích / chi phí):**
1. **TE trong fold của PRODUCT=round(C/A,2), α=10, làm cột thứ 5 của meta-LR** (L2.1). Đã đo +0.0015, 5/5 → ~0.7694. Chi phí: vài giây. Cần dựng thêm TE cho test (từ toàn train) để ra submission.
2. **Đưa tín hiệu sản phẩm vào tầng 1** (L2.2): cột TE PRODUCT lồng fold cho LGB/Cat/XGB, và/hoặc PRODUCT làm biến phân loại cho CatBoost (ordered target statistics có sẵn, chống rò rỉ tự nhiên). Kỳ vọng +0.001–0.002 nếu cây kết hợp được TE với EXT/tuổi; rủi ro: lệch inner/outer như KNN → phải qua R3, R16.
3. **Sửa dựng kỳ hạn** (L2.3): thêm tiên nghiệm lãi suất hợp lý (phạt lãi > 35–40%/năm) hoặc dùng EM/posterior (trùng A1 #1 phần chưa làm); kiểm bằng: tỉ lệ TERM_DENS_RATIO > 2 tăng, gain TERM_HAT tăng, sàng lọc dương. Lợi ích không chắc (TE PRODUCT có thể đã lấy hết), ưu tiên sau 1–2.
4. **Nhóm yếu** (L1.5b → cụ thể hóa L2.4): đặc trưng/mô hình cho hưu trí & > 60 tuổi (18% dữ liệu, AUC 0.7405/0.7274) và Revolving (9.5%, không có RATE/TERM, 0.7609): ví dụ TE theo (NAME_INCOME_TYPE × học vấn × nhóm tuổi 5 năm) hoặc GBDT huấn luyện riêng trên nhóm rồi đưa vào stack. Lưu ý meta-theo-nhóm đã âm → phải là mô hình/đặc trưng tầng 1.
5. **Multi-seed Cat/LGB** (L1.4 phần còn lại) và hoàn tất các mục tồn L1.1c, L1.2b, L1.5a — rẻ nhưng kỳ vọng nhỏ (≤ +0.0005 mỗi mục); làm sau cùng để đóng checklist.

##### 6. CHECKLIST BÀI HỌC

| Mã | Hướng / bài học có thể thực thi | Trạng thái |
|---|---|---|
| L1.1 | Điền khuyết EXT bằng mô hình + RATE_n vào bộ đặc trưng, chạy đủ LGB/Cat/XGB, rank-avg | ĐÃ LÀM (vòng 1: LGB 0.76696, XGB 0.76643, Cat 0.76545; rank-avg 0.76751) |
| L1.1b | Kỳ hạn bổ sung n=6,10,18,30,42; RATE theo loại hợp đồng | ĐÃ LÀM (vòng 1, trong TERM 11 kỳ hạn; Revolving → NaN; TERM +0.0021 sàng lọc) |
| L1.1c | RATE tính với AMT_GOODS_PRICE làm gốc; điền khuyết EXT bằng thuật toán thứ hai (vd. CatBoost/ridge) rồi lấy trung bình | CHƯA LÀM |
| L1.2 | KNN target-mean lồng fold làm đặc trưng (k=50/200/500) | ĐÃ LÀM (vòng 1: −0.0005…−0.0007, 1/4 → bỏ) |
| L1.2b | KNN làm MÔ HÌNH riêng / cột ở tầng meta (theo R13: thử logit(KNN-500) làm cột meta, dùng `r1_knn.pkl` sẵn có) | CHƯA LÀM |
| L1.3 | Mô hình khác họ: LR-binning, MLP | ĐÃ LÀM (vòng 1: lrbin 0.75688 giữ, +0.0004; MLP 0.75482 bỏ) |
| L1.4 | Multi-seed cấu hình cuối cho cả 3 thuật toán | ĐÃ LÀM một phần (vòng 1: LGB 3 seed). Phần còn lại — Cat ≥ 2 seed, XGB tùy thời gian (R18): CHƯA LÀM |
| L1.5a | Pseudo-label test độ tin cậy cao | CHƯA LÀM |
| L1.5b | Mô hình/đặc trưng riêng cho nhóm yếu (hưu trí, > 60 tuổi, Revolving) ở TẦNG 1 — xem L2.4 | CHƯA LÀM (gộp vào L2.4) |
| R1–R12 | Quy tắc đánh giá/phân bổ/ghép vòng 1 | ĐÃ LÀM (vòng 1, áp dụng cho mọi quyết định) |
| L2.1 | Thêm logit(TE trong fold của PRODUCT=round(C/A,2), α=10) làm cột meta-LR, lồng theo R14; dựng TE test từ toàn train; ra submission | CHƯA LÀM (A2 đã kiểm sơ bộ: +0.00155, 5/5 → ≈0.7694; T3 cần cài chính thức + test) |
| L2.2 | Đưa tín hiệu sản phẩm vào tầng 1: (a) cột TE PRODUCT lồng fold cho GBDT, sàng lọc R3; (b) PRODUCT làm biến phân loại CatBoost. So cả hai cấu hình theo R16 | CHƯA LÀM (sàng lọc của A2 không chạy xong vì tranh CPU) |
| L2.3 | Sửa dựng kỳ hạn: tiên nghiệm/phạt lãi > 35–40%/năm hoặc posterior; kiểm bằng TERM_DENS_RATIO, gain TERM_HAT, sàng lọc R3 | CHƯA LÀM |
| L2.4 | Nhóm yếu ở tầng 1: TE trong fold (INCOME_TYPE × EDU × nhóm tuổi 5 năm) ở meta theo R13, và/hoặc GBDT riêng cho hưu trí+>60 tuổi và cho Revolving đưa vào stack | CHƯA LÀM |
| L2.5 | Giữ meta = LR đơn giản; KHÔNG dùng meta có tương tác nhóm / meta GBDT (R17) | ĐÃ LÀM (vòng 2 bởi A2: −0.00033 0/5; −0.00040 1/5 → loại, không cần T3 làm lại) |
| L2.6 | Không đưa mô hình vòng 0 / MLP sklearn vào stack; tiêu chí nhận mô hình R18 | ĐÃ LÀM (vòng 1: +0.00015 và +0.00002 → đã loại; từ nay áp dụng R18) |
| L2.7 | Ghép khách hàng lặp lại (uid) | KHÔNG THỰC THI ĐƯỢC có ích (A2 đo: chỉ 124 dòng trùng khi buộc EXT1 bằng nhau; nhóm trùng 247/2,675 lẫn nhãn → không có tín hiệu) |
| L2.8 | In bảng AUC theo nhóm cho stack mới mỗi vòng (R20) và ghi vào history.md | CHƯA LÀM |
| L2.9 | Nhóm đặc trưng thử ≤ 5 cột; tách TERM để kiểm từng phần — cụ thể: sàng lọc bỏ `TERM_HAT`, `EXT_k_RES` (gain 0.05% / 1%) xem có giảm không (R19; khác R8 vì đây là ablation ghép cặp, không cắt theo gain) | CHƯA LÀM |

Tổng: **10 mục CHƯA LÀM** (L1.1c, L1.2b, L1.4-phần còn lại, L1.5a, L2.1, L2.2, L2.3, L2.4, L2.8, L2.9; L1.5b đã gộp vào L2.4).

### Tác nhân 3 — Thực hành (mục VÒNG 2 trong `loop/history.md`)
Script (trong `$S`): `fe_r2.py` (`load_r2()` = `load_r1()` + nhóm PROF/PROFX/PROFR/PROFC/RATEG/IMPR/IMPB; `nested_te`, `te_map`, `PRODUCT_KEY`; cache `r2_df.pkl`, `r2_imp_ridge.pkl`),
`r2_screen.py` (sàng lọc R3, kết quả `r2_screen.json`), `run_r2.py <lgb|cat|xgb> <spec[~TE][~PCAT]> <seeds> <tag> [nfold]` (5 fold, lưu tạm `ck_<tag>.npz`),
`div_r2.py <lrbin|ebm>`, `r2_grp.py` (mô hình riêng theo nhóm), `ens_r2.py` + `ens_final_r2.py` (ensemble, ghi submission), `r2_meta1.py`/`r2_meta2.py` (kiểm ở meta), `r2_groups.py` (bảng nhóm).
Log: `log_r2_{screen1..4,meta1,catscr,grp,lgb,lgbte,cat,xgb,lrbin,ebm,ens1,ens_final,groups}.txt`. EBM: `interpret-core 0.7.8` cài vào `$S/pylib` (bản 0.6.9 tải sẵn lỗi với numpy 2.5).
Bộ đặc trưng chốt: **IMP + RATE + TERM + PROF (8 cột)**; LGB thêm cột TE PRODUCT lồng fold; CatBoost thêm PRODUCT làm biến phân loại.

Định nghĩa mới:
- **PROF** (không dùng TARGET, train+test): khóa = TERM_HAT × round(RATE_HAT, 0.5%) (Revolving = 1 khóa riêng); trung bình theo khóa của {EXTF_MEAN, EXT_2, EXT_3, EXT_1_PRED, DAYS_BIRTH, DAYS_EMPLOYED, tỉ lệ nam} + std EXTF_MEAN; khóa < 30 dòng → NaN. Trùng khớp `pv_TRcore.pkl` của A1 (sai khác 0).
- **TE PRODUCT** (dùng TARGET, lồng nghiêm ngặt R14): PRODUCT = round(C/A, 2), α = 10; với fold ngoài (a, b): dòng a dùng inner 5-fold trong a; dòng b và test dùng toàn a. Nhãn fold chấm không bao giờ được dùng.
- **PCAT**: PRODUCT (mức < 30 dòng gộp thành `rare`) làm biến phân loại gốc cho CatBoost (ordered target statistics).

##### Sàng lọc nhanh (LGB lr 0.05, fold 0/1 × 2 seed; REF = IMP+RATE+TERM = 0.76531)
| Thay đổi | TB 4 lần | Chênh ghép cặp | Dương | Quyết định |
|---|---|---|---|---|
| + PROF (8 cột) | 0.76793 | +0.00261 vs REF | 4/4 | **GIỮ** (bản gọn nhất) |
| + PROF + PROFX (14 cột) | 0.76828 | +0.00296 vs REF; +0.00035 vs PROF (3/4) | 4/4 | BỎ PROFX (dưới ngưỡng) |
| + PROFR (20 cột, khóa RATE_n) | 0.76838 | +0.00306 vs REF; +0.00045 vs PROF (3/4) | 4/4 | BỎ (dưới ngưỡng, 20 cột) |
| + PROFC (khóa C/A 2 chữ số, 4 cột) | 0.76710 | +0.00178 vs REF | 4/4 | BỎ (kém PROF) |
| + TE PRODUCT lồng fold (L2.2a) | 0.76744 | +0.00212 vs REF | 4/4 | kém PROF |
| + PROF + TE PRODUCT | 0.76842 | +0.00310 vs REF; +0.00049 vs PROF (2/4) | 4/4 | sàng lọc không rõ → đo ở 5 fold đầy đủ (bên dưới): GIỮ cho LGB |
| + PROFR + TE | 0.76823 | +0.00291 vs REF | 4/4 | BỎ |
| PROF − TERM_HAT (L2.9) | 0.76806 | +0.00013 vs PROF | 3/4 | không đổi → giữ nguyên cột (bỏ hay giữ đều ≈ 0) |
| PROF − 3 cột `_RES` (L2.9) | 0.76797 | +0.00005 | 2/4 | như trên |
| PROF − TERM_HAT − `_RES` | 0.76807 | +0.00014 | 3/4 | như trên |
| PROF + RATEG (lãi suất với GOODS làm gốc, 5 cột; L1.1c) | 0.76789 | −0.00004 | 2/4 | BỎ |
| PROF + IMPR (EXT dự đoán bằng Ridge, 3 cột; L1.1c) | 0.76792 | −0.00001 | 3/4 | BỎ |
| PROF, IMP thay bằng IMPB (trung bình LGB + Ridge; L1.1c) | 0.76807 | +0.00015 | 2/4 | BỎ |
| PROF + TE (INCOME_TYPE × EDU × tuổi 5 năm) (L2.4) | 0.76791 | −0.00001 | 2/4 | BỎ |
| PROF + PP (A1 #2 vòng 2: trung bình logit dự đoán theo sản phẩm, lồng fold, gồm test) | 0.76785 | −0.00007 | 2/4 | BỎ |
| PROF + pseudo-label mềm toàn bộ test (trọng số p / 1−p; L1.5a) | 0.76790 | −0.00003 | 2/4 | BỎ |
| PROF + pseudo-label cứng (50% thấp nhất → 0, 1% cao nhất → 1; L1.5a) | 0.76820 | +0.00027 | 4/4 | BỎ (dưới ngưỡng +0.0005) |
| PROF + PRODUCT làm categorical cho **LGB** | 0.76499 | **−0.00294** | 0/4 | BỎ (LGB quá khớp biến 2,000 mức) |
(Pseudo-label trong sàng lọc dùng dự đoán test của mô hình chỉ học trên phần train của fold → không rò nhãn fold chấm.)

Sàng lọc CatBoost (lr 0.1, fold 0/1, 1 seed; L2.2b):
| Cấu hình | fold 0 / fold 1 | TB | Chênh vs PROF |
|---|---|---|---|
| PROF | 0.76484 / 0.76758 | 0.76621 | — |
| PROF + PCAT | 0.76694 / 0.76959 | 0.76826 | **+0.00205** (2/2) → GIỮ |
| PROF + cột TE lồng fold | 0.76580 / 0.76863 | 0.76722 | +0.00101 (2/2), kém PCAT |
| PROF + PCAT + khóa TERM×RATE làm cat | 0.76669 / 0.76851 | 0.76760 | −0.00066 vs PCAT → BỎ |
| PROF + PCAT + round(C/GOODS,3) làm cat | 0.76760 / 0.76750 | 0.76755 | −0.00071 → BỎ |
| PROF + PCAT + AMT_ANNUITY làm cat | 0.76587 / 0.76791 | 0.76689 | −0.00137 → BỎ |

Kiểm ở tầng meta trên OOF vòng 1 (stack r1 = 0.76790; lồng nghiêm ngặt):
| Cột thêm vào meta | OOF | Chênh TB | Dương |
|---|---|---|---|
| TE PRODUCT α=5 / 10 / 20 (L2.1) | 0.76944 / 0.76946 / 0.76940 | +0.00153 / +0.00155 / +0.00149 | 5/5 (tái lập đúng số của A2) |
| logit KNN1_500 / KNN2_500 / cả hai / KNN1_50 (L1.2b) | 0.76792 / 0.76789 / 0.76790 / 0.76790 | +0.00002 / 0 / +0.00001 / 0 | 4/5, 2/5, 3/5, 2/5 → BỎ |
| TE (INCOME × EDU × tuổi 5 năm) / thêm × giới × nghề (L2.4a) | 0.76788 / 0.76788 | −0.00002 | 1/5 → BỎ |

##### Chạy đầy đủ 5 fold (StratifiedKFold rs=42)
| Mô hình | Seed | OOF AUC | AUC từng fold | So với vòng 1 cùng thuật toán |
|---|---|---|---|---|
| LGB + PROF `lgb_r2` | 3 | 0.76978 | 0.76726 / 0.77063 / 0.77518 / 0.76604 / 0.76993 | +0.00280, 5/5 |
| LGB + PROF + TE PRODUCT lồng fold `lgbte_r2` | 3 | **0.77008** | 0.76748 / 0.77106 / 0.77563 / 0.76650 / 0.76996 | +0.00312; +0.00030 vs lgb_r2, 5/5 |
| CatBoost + PROF + PCAT `cat_r2` | 2 | 0.76947 | 0.76834 / 0.77021 / 0.77339 / 0.76586 / 0.76970 | **+0.00402**, 5/5 |
| XGB + PROF `xgb_r2` | 2 | 0.76929 | 0.76679 / 0.76987 / 0.77494 / 0.76547 / 0.76960 | +0.00286, 5/5 |
| LR-binning + PROF `lrbin_r2` | — | 0.75851 | 0.75619 / 0.75831 / 0.76240 / 0.75531 / 0.76043 | +0.00163 |
| EBM (interactions 10, outer_bags 4; A1 #3 vòng 2) `ebm_r2` | — | 0.76715 | 0.76519 / 0.76761 / 0.77196 / 0.76468 / 0.76662 | mới; Spearman vs rank-avg GBDT = 0.980 (> 0.95) |
| LGB riêng theo nhóm (L2.4b, lr 0.02, 2 seed) — AUC trong nhóm: mô hình riêng vs lgb_r2 | | | hưu trí/>60: 0.7340 vs 0.7421; Revolving: 0.7480 vs 0.7593; <30 tuổi: 0.7361 vs 0.7411; thiếu 2 EXT: 0.7455 vs 0.7520 | mô hình riêng KÉM mô hình chung 0.005–0.011 ở chính nhóm đó |

##### Ensemble (meta chấm trên fold chưa thấy; chênh so với stack vòng 1 = 0.76790)
| Cách | OOF | Chênh TB | Ghi chú |
|---|---|---|---|
| rank-avg lgb/cat/xgb r2 | 0.77042 | +0.00248, 5/5 | |
| stack LR lgb/cat/xgb r2 | 0.77047 | +0.00255 | hệ số xgb ≈ −0.06 |
| + lrbin_r2 | 0.77062 | +0.00270 | lrbin_r2 +0.00014 |
| + lrbin_r2 + TE PRODUCT ở meta | 0.77074 | +0.00282 | TE meta chỉ còn +0.00013 (5/5) khi tầng 1 đã có PROF (trước đó +0.00155) |
| + lrbin_r2 + EBM | 0.77063 | +0.00271 | EBM +0.00001 (2/5) → BỎ |
| + lrbin_r2 + mô hình riêng nhóm (old / rev / young / ext2na) | 0.77052 / 0.77061 / 0.77064 / 0.77060 | −0.00010 … +0.00001 | BỎ |
| + lrbin_r2 + lgbte_r2 (5 mô hình) | 0.77082 | +0.00290 | hệ số xgb âm |
| **stack LR lgbte_r2 + cat_r2 + xgb_r2 + lrbin_r2 (CHỌN)** | **0.77081** | **+0.00289, 5/5** | hệ số 0.506 / 0.396 / 0.036 / 0.123; +0.00019 (5/5) so với bản dùng lgb_r2 |
| như trên + TE ở meta | 0.77083 | +0.00291 | TE meta thừa khi LGB đã có TE (+0.00002) |
| 3 GBDT r2 + 3 GBDT r1 + lrbin_r2 + TE meta (8 cột) | 0.77090 | +0.00298 | +0.00009 so với bản chọn, trái R18 → BỎ |

Ba cấu hình tín hiệu sản phẩm (R16): chỉ TE ở meta (trên mô hình vòng 1) 0.76946; chỉ hồ sơ PROF ở tầng 1 0.77062; PROF + TE ở meta 0.77074; PROF + TE trong LGB + PCAT trong CatBoost **0.77081**. Các bản có PROF chênh nhau ≤ 0.0002 (dưới ngưỡng R2) — chọn bản 4 mô hình hệ số đều dương.

**AUC OOF cuối vòng 2 = 0.77081** (vòng 1: 0.76790, **+0.00289**, dương 5/5). Từng fold: 0.76875 / 0.77172 / 0.77550 / 0.76727 / 0.77105.
Submission ứng viên: `$S/loop/sub_round2.csv` (61,500 dòng, thứ tự = sample_submission, không NaN; Spearman với sub_round1 = 0.985). Chưa ghi vào repo.
Mục tiêu 0.80: CHƯA đạt (còn thiếu 0.029).

##### Bảng AUC theo nhóm (L2.8 / R20) — stack vòng 1 → stack vòng 2
| Nhóm | Mức | n | tỉ lệ | vỡ nợ | AUC stack r1 | AUC stack r2 | Chênh |
|---|---|---|---|---|---|---|---|
| ALL | all | 171202 | 1.000 | 0.081 | 0.7679 | 0.7708 | +0.0029 |
| EXT_NA | 0 | 36164 | 0.211 | 0.071 | 0.7770 | 0.7800 | +0.0030 |
| EXT_NA | 1 | 96310 | 0.563 | 0.080 | 0.7712 | 0.7734 | +0.0022 |
| EXT_NA | 2 | 38611 | 0.226 | 0.091 | 0.7487 | 0.7534 | +0.0047 |
| EXT3_NA | have | 116616 | 0.681 | 0.077 | 0.7749 | 0.7775 | +0.0026 |
| EXT3_NA | na | 54586 | 0.319 | 0.089 | 0.7517 | 0.7553 | +0.0036 |
| CONTRACT | Cash loans | 154988 | 0.905 | 0.083 | 0.7671 | 0.7703 | +0.0032 |
| CONTRACT | Revolving loans | 16214 | 0.095 | 0.056 | 0.7609 | 0.7606 | -0.0003 |
| PENSION | 0 | 140304 | 0.820 | 0.087 | 0.7679 | 0.7707 | +0.0028 |
| PENSION | 1 | 30898 | 0.180 | 0.054 | 0.7405 | 0.7445 | +0.0040 |
| AGE | (0, 30] | 25125 | 0.147 | 0.115 | 0.7397 | 0.7422 | +0.0025 |
| AGE | (30, 40] | 45673 | 0.267 | 0.096 | 0.7692 | 0.7716 | +0.0024 |
| AGE | (40, 50] | 42732 | 0.250 | 0.077 | 0.7617 | 0.7649 | +0.0031 |
| AGE | (50, 60] | 37981 | 0.222 | 0.060 | 0.7637 | 0.7675 | +0.0038 |
| AGE | (60, 100] | 19691 | 0.115 | 0.049 | 0.7274 | 0.7340 | +0.0066 |
| GENDER | F | 112549 | 0.657 | 0.070 | 0.7650 | 0.7681 | +0.0031 |
| GENDER | M | 58649 | 0.343 | 0.101 | 0.7606 | 0.7635 | +0.0029 |
| INC_TYPE | Commercial associate | 40007 | 0.234 | 0.075 | 0.7610 | 0.7631 | +0.0021 |
| INC_TYPE | Pensioner | 30895 | 0.180 | 0.054 | 0.7402 | 0.7442 | +0.0040 |
| INC_TYPE | State servant | 12007 | 0.070 | 0.059 | 0.7800 | 0.7844 | +0.0044 |
| INC_TYPE | Working | 88268 | 0.516 | 0.095 | 0.7655 | 0.7684 | +0.0030 |
| TERM_HAT | -1 | 16214 | 0.095 | 0.056 | 0.7609 | 0.7606 | -0.0003 |
| TERM_HAT | 10 | 5754 | 0.034 | 0.067 | 0.7371 | 0.7390 | +0.0019 |
| TERM_HAT | 12 | 14120 | 0.082 | 0.047 | 0.7946 | 0.7974 | +0.0029 |
| TERM_HAT | 18 | 14712 | 0.086 | 0.138 | 0.7398 | 0.7438 | +0.0040 |
| TERM_HAT | 24 | 9501 | 0.055 | 0.103 | 0.7833 | 0.7880 | +0.0046 |
| TERM_HAT | 30 | 9854 | 0.058 | 0.095 | 0.7637 | 0.7672 | +0.0036 |
| TERM_HAT | 36 | 10480 | 0.061 | 0.070 | 0.7571 | 0.7590 | +0.0020 |
| TERM_HAT | 42 | 19542 | 0.114 | 0.084 | 0.7465 | 0.7486 | +0.0021 |
| TERM_HAT | 48 | 30241 | 0.177 | 0.072 | 0.7315 | 0.7350 | +0.0035 |
| TERM_HAT | 54 | 27869 | 0.163 | 0.086 | 0.7790 | 0.7818 | +0.0028 |
| TERM_HAT | 60 | 12915 | 0.075 | 0.076 | 0.7595 | 0.7640 | +0.0045 |
Nhận xét: lợi ích lớn nhất ở tuổi > 60 (+0.0066), thiếu 2 EXT (+0.0047), TERM_HAT 24/60 (+0.0046/+0.0045), công chức (+0.0044), hưu trí (+0.0040). Revolving KHÔNG được lợi (−0.0003: không có khóa sản phẩm). Nhóm yếu nhất vẫn là > 60 tuổi (0.7340), TERM_HAT 48 (0.7350), 10 (0.7390), < 30 tuổi (0.7422).

##### Phương pháp ĐÃ dùng (bổ sung vòng 2 — Tác nhân 1 không đề xuất lại)
Hồ sơ quần thể của sản phẩm vay (group-mean không nhãn theo khóa TERM×RATE, RATE_n, C/A; train+test; 8/14/20/4 cột); target encoding lồng fold của PRODUCT=round(C/A,2) ở meta VÀ làm cột cho LGB/CatBoost;
PRODUCT làm biến phân loại gốc cho CatBoost (ordered TS) và cho LGB (thử, bỏ); thêm khóa phân loại TERM×RATE / C÷GOODS / AMT_ANNUITY cho CatBoost (thử, bỏ);
lãi suất ngầm định với AMT_GOODS_PRICE làm gốc (thử, bỏ); điền khuyết EXT bằng thuật toán thứ hai (Ridge trên bin one-hot) và trung bình hai bộ điền khuyết (thử, bỏ);
ablation TERM_HAT / `_RES`; TE theo INCOME_TYPE×EDU×nhóm tuổi (meta và tầng 1; thử, bỏ); GBDT riêng cho nhóm hưu trí/>60, Revolving, <30 tuổi, thiếu 2 EXT (thử, bỏ);
KNN-target làm cột meta (thử, bỏ); pseudo-label test mềm và cứng (thử, bỏ); hồ sơ theo dự đoán của mô hình lồng fold (thử, bỏ); EBM/GA2M `interpret` (thử, bỏ);
multi-seed LGB 3 / CatBoost 2 / XGB 2; bảng AUC theo nhóm.

##### Phát hiện bất ngờ
1. Tín hiệu sản phẩm là MỘT tín hiệu, bão hòa quanh +0.003: PROF, TE, PCAT, TE ở meta thay thế lẫn nhau. Khi tầng 1 đã có PROF, cột TE ở meta rơi từ +0.00155 xuống +0.00013.
2. CatBoost hưởng lợi nhiều nhất (+0.0040 so với vòng 1, hơn LGB/XGB +0.0028) nhờ PRODUCT làm biến phân loại (sàng lọc +0.0021 trên PROF); cùng biến đó làm LGB GIẢM −0.0029 (0/4). Thêm khóa phân loại thứ hai lại giảm (R15 đúng).
3. Sàng lọc PROF+TE cho LGB không rõ (2/4) nhưng ở 5 fold đầy đủ lr 0.01 lại dương 5/5 (+0.0003) — mức quá nhỏ để sàng lọc 4 lần chấm phân giải; nhiễu sàng lọc giữa các biến thể tương đương ≈ ±0.0005.
4. Mô hình riêng theo nhóm yếu kém mô hình chung 0.005–0.011 ngay trong nhóm đó: nhóm yếu là do thiếu thông tin, không phải do mô hình chung bỏ quên.
5. EBM đạt 0.76715 chỉ sau ~1 phút/fold nhưng tương quan 0.980 với GBDT → không thêm gì; lrbin (0.7585, corr 0.93) vẫn có ích hơn dù yếu hơn 0.009.
6. Mọi hướng ngoài khóa sản phẩm đều ≈ 0 (RATEG, điền khuyết thứ hai, pseudo-label, KNN meta, TE nhân khẩu học, PP, EBM) — khớp nhận định của A1/A2 rằng 33 cột đã gần cạn; trần thực tế ≈ 0.771.

##### Bài học đã thực hiện (đối chiếu round2_agent2.md §6 và round1/2_agent1.md)
| Mã | Hướng / bài học | Trạng thái |
|---|---|---|
| L1.1 / L1.1b | Điền khuyết EXT + RATE_n; kỳ hạn bổ sung | ĐÃ LÀM (vòng 1) |
| L1.1c | RATE với AMT_GOODS_PRICE làm gốc; điền khuyết EXT bằng thuật toán thứ hai rồi trung bình | ĐÃ LÀM (vòng 2): RATEG −0.00004 (2/4); Ridge PRED +3 cột −0.00001; trung bình LGB+Ridge +0.00015 (2/4) → BỎ cả ba |
| L1.2 | KNN target-mean làm đặc trưng | ĐÃ LÀM (vòng 1, bỏ) |
| L1.2b | KNN làm cột ở tầng meta | ĐÃ LÀM (vòng 2): +0.00002 … 0 → BỎ. (KNN làm mô hình đứng riêng = chính cột này) |
| L1.3 | LR-binning, MLP | ĐÃ LÀM (vòng 1); vòng 2 lrbin + PROF = 0.75851, +0.00014 trong stack → GIỮ |
| L1.4 | Multi-seed cả 3 thuật toán | ĐÃ LÀM (vòng 2): LGB 3 seed, CatBoost 2 seed, XGB 2 seed |
| L1.5a | Pseudo-label test | ĐÃ LÀM (vòng 2, sàng lọc): mềm −0.00003 (2/4); cứng +0.00027 (4/4, dưới ngưỡng) → BỎ |
| L1.5b | Mô hình/đặc trưng riêng nhóm yếu | ĐÃ LÀM (gộp L2.4) |
| R1–R12 | Quy tắc vòng 1 | ĐÃ ÁP DỤNG |
| L2.1 | TE PRODUCT (α=10) làm cột meta-LR, lồng R14, có TE test | ĐÃ LÀM: trên mô hình vòng 1 +0.00155 (0.76946, 5/5); trên mô hình vòng 2 chỉ +0.00013 … +0.00002 → không dùng ở meta trong bản chọn (TE nằm trong LGB) |
| L2.2 | Tín hiệu sản phẩm vào tầng 1: (a) cột TE lồng fold cho GBDT; (b) PRODUCT categorical cho CatBoost; so cấu hình R16 | ĐÃ LÀM: (a) LGB sàng lọc +0.00212 riêng / +0.00049 trên PROF; đầy đủ lgbte_r2 0.77008 (+0.00030 vs lgb_r2, 5/5) → GIỮ; (b) CatBoost +0.00205 sàng lọc, cat_r2 0.76947 → GIỮ. Bảng 3 cấu hình ở mục Ensemble |
| L2.3 | Sửa dựng kỳ hạn (tiên nghiệm/phạt lãi, posterior/EM) | ĐÃ LÀM bởi A1 vòng 2 (EM có điều kiện: −0.00001, 2/4) → ghi nhận, T3 không lặp lại. Bản "phạt lãi > 35–40%" riêng: CHƯA LÀM (kỳ vọng ≈ 0 vì PROF/TE dùng khóa thô đã lấy tín hiệu) |
| L2.4 | Nhóm yếu ở tầng 1: TE (INCOME×EDU×tuổi) và GBDT riêng theo nhóm | ĐÃ LÀM: TE ở meta −0.00002 (1/5), làm cột LGB −0.00001 (2/4); GBDT riêng 4 nhóm (hưu trí/>60, Revolving, <30, thiếu 2 EXT) kém mô hình chung trong nhóm 0.005–0.011, trong stack −0.00010 … +0.00001 → BỎ |
| L2.5 / R17 | Meta = LR đơn giản | ĐÃ ÁP DỤNG |
| L2.6 / R18 | Không đưa mô hình tập con thông tin vào stack | ĐÃ ÁP DỤNG (bản 8 mô hình r1+r2 chỉ +0.00009 → loại; EBM corr 0.98 → loại) |
| L2.7 | uid khách hàng lặp lại | KHÔNG THỰC THI ĐƯỢC có ích (A2 đã đo) |
| L2.8 / R20 | Bảng AUC theo nhóm | ĐÃ LÀM (bảng trên) |
| L2.9 / R19 | Ablation TERM_HAT, `_RES` | ĐÃ LÀM: bỏ TERM_HAT +0.00013 (3/4), bỏ `_RES` +0.00005 (2/4), bỏ cả hai +0.00014 (3/4) → không ảnh hưởng, giữ nguyên |
| R13–R16, R21 | TE thử ở meta trước; lồng R14; một tín hiệu một cột; báo cả hai cấu hình; ≤ 1 tiến trình 4 luồng | ĐÃ ÁP DỤNG |
| A1 vòng 1 #1 | Tái dựng kỳ hạn/lãi suất; bản posterior/KDE | ĐÃ LÀM (vòng 1) + bản EM của A1 vòng 2 (≈ 0) |
| A1 vòng 1 #2, #3, #4 | KNN / điền khuyết EXT / ngày-hộ chiếu | ĐÃ LÀM (vòng 1) |
| A1 vòng 1 #5 | RealMLP / TabM (torch) | CHƯA LÀM (cần torch ~3 GB; A1 vòng 2 ghi TabM/ModernNCA giảm chất lượng trên CPU có giới hạn giờ) |
| A1 vòng 1 #6 | TE OOF cho tổ hợp danh mục | ĐÃ LÀM (vòng 2): PRODUCT có ích; ORG×OCC (A2: −0.00005), INCOME×EDU×tuổi (−0.00002) vô ích |
| A1 vòng 1 #7 | Null Importance | CHƯA LÀM (ablation L2.9 cho thấy bỏ cột gain thấp ≈ 0 → kỳ vọng ≈ 0) |
| A1 vòng 1 #8 | uid | KHÔNG THỰC THI ĐƯỢC có ích (L2.7) |
| A1 vòng 1 #9 | LGB DART / extra_trees / linear_tree | CHƯA LÀM |
| A1 vòng 1 #10 | TabICL / TabPFN | KHÔNG THỰC THI ĐƯỢC (cần GPU) |
| A1 vòng 2 #1 | Hồ sơ quần thể sản phẩm | ĐÃ LÀM: sàng lọc +0.00261 (4/4); đầy đủ LGB +0.0028, XGB +0.0029, Cat +0.0040 (cùng PCAT); stack +0.0029 |
| A1 vòng 2 #2 | Hồ sơ theo dự đoán mô hình (lồng fold) | ĐÃ LÀM (sàng lọc): −0.00007 (2/4) → BỎ |
| A1 vòng 2 #3 | EBM / GA2M | ĐÃ LÀM: 0.76715, corr 0.980, +0.00001 trong stack → BỎ |

Còn CHƯA LÀM sau vòng 2: A1 vòng 1 #5 (RealMLP/TabM), #7 (Null Importance), #9 (DART/extra_trees/linear_tree); phần "phạt lãi suất cao" của L2.3.

---

## VÒNG 3 (VÒNG CUỐI)

### Tác nhân 1 — Nghiên cứu phương pháp mới (`loop/round3_agent1.md`)
STATUS: NO_NEW_METHODS

#### VÒNG 3 — Tác nhân 1 (Nghiên cứu): không tìm được phương pháp mới có bằng chứng dương

Đã đọc: `rules.md`, `history.md` (vòng 0–2), `round1_agent1.md`, `round2_agent1.md`, `round2_agent2.md`, `fe_r2.py`, `run_r2.py`, `r2_screen.py`.
Mốc: stack vòng 2 = 0.77081. Script/log của tôi: `$S/loop/a1r3/` (`base.py` khung sàng lọc, `q0.py`, `e1.py`, `mk2.py`+`e2.py`, `e3.py`, `ofe.py`+`e6.py`, `e5.py`; log `log_e1/e2/e3/e5/e6.txt`, `log_ofe0.txt`; kết quả `scr.json`).

##### Kết luận một dòng
Tôi sàng lọc 9 hướng CHƯA có trong history/báo cáo cũ (14 biến thể). Không hướng nào vượt ngưỡng R2 (+0.0005 và ≥ 4/5): tốt nhất là +0.00025 (7/10 lần chấm) — nằm trong nhiễu sàng lọc ±0.0005 đã ghi ở vòng 2. Vì vậy báo `NO_NEW_METHODS` thay vì đề xuất cho có.

##### Cách đo
Khung y hệt sàng lọc R3 của vòng 2: BASE = IMP + RATE + TERM + PROF (129 cột), LGB lr 0.05 depth 4, fold 0/1 × seed 0/1, so ghép cặp.
BASE tái lập ĐÚNG số trong `r2_screen.json` ('+PROF'): [0.76695, 0.76692, 0.76896, 0.76888], TB 0.76793.
Hai ứng viên dương nhất được chấm thêm trên fold 2/3/4 × 2 seed (tổng 10 lần chấm ghép cặp).
Giới hạn: máy bị tranh CPU với Tác nhân 2 suốt phiên (load 11–19 trên 4 nhân) nên không chạy được 5 fold lr 0.01 đầy đủ; mọi số dưới đây là mức sàng lọc.

##### (2) Phương pháp mới đề xuất
Không có.

##### (3) Đã thử sàng lọc — tất cả ≈ 0 hoặc âm (đừng lặp lại)

| # | Hướng (vì sao chưa trùng) | Kết quả so BASE | Dương | Kết luận |
|---|---|---|---|---|
| 1 | **Phần dư "định giá" so với hồ sơ khách** (`mk2.py`): LGB hồi quy OOF target-free, train+test, dự đoán RATE_HAT, log(C/A), log thu nhập, thâm niên từ các cột khách hàng → cột `_PRED` + `_RES` (8 cột). Khác IMP (chỉ điền EXT) | fold 0/1: +0.00057 (4/4); fold 2/3/4: +0.00004 (3/6) → **gộp 10 lần: +0.00025, 7/10** | 7/10 | Dưới ngưỡng. Hai tập con riêng đều âm: chỉ RATE/C÷A −0.00023 (0/4), chỉ thu nhập/thâm niên −0.00009 (2/4) → mức +0.00057 ban đầu là nhiễu |
| 2 | **OpenFE** (Zhang et al., ICML 2023 — sinh đặc trưng tự động + feature boosting; `ofe.py`). Chọn đặc trưng CHỈ bằng nhãn phần train của fold 0 (80k dòng mẫu, init_score = logit OOF trong fold của BASE), 477 ứng viên qua vòng 2 | top 10: **−0.00048** (1/4); top 40: **−0.00032** (0/4) (fold 0 × 4 seed) | 0–1/4 | BỎ. Đặc trưng đứng đầu là tổ hợp vô nghĩa (vd. `DAYS_REGISTRATION + AMT_REQ_YEAR`) → không còn tương tác bậc 2 nào có ích |
| 3 | **Tự chưng cất / nhãn mềm** (`e3.py`): OOF inner 5-fold trong phần train, học lại với nhãn (1−α)·y + α·p, objective cross_entropy. Khác pseudo-label (dùng test) | α=0.3: +0.00006 (2/4); α=0.6: +0.00026 (3/4); α=0.9: −0.00145 (0/4); cross_entropy không chưng cất: +0.00011 (2/4) | — | BỎ (dưới ngưỡng, không đơn điệu theo α) |
| 4 | **Ràng buộc đơn điệu** trên 11 cột EXT (giảm), method `advanced` | −0.00001 | 2/4 | BỎ |
| 5 | **path_smooth = 50** (làm trơn giá trị lá theo nút cha) | fold 0/1: +0.00024 (3/4); fold 2/3/4: +0.00014 (4/6) → gộp +0.00018, 7/10 | 7/10 | Dưới ngưỡng; vô hại, có thể bật nhưng không tính là cải thiện |
| 6 | **max_bin = 1023** (để cây tách được từng cụm sản phẩm C/A) | +0.00010 | 2/4 | BỎ — tín hiệu sản phẩm đã do PROF/TE lấy hết |
| 7 | **GOSS** thay bagging | −0.00022 | 2/4 | BỎ |
| 8 | scale_pos_weight = 3 | −0.02197 | 0/4 | BỎ (hỏng early stopping/hiệu chuẩn ở lr 0.05; không đáng theo đuổi) |
| 9 | **Thứ trong tuần ẩn**: DAYS_* mod 7 (6 cột; ngày cấp giấy tờ rơi vào ngày làm việc ⇒ suy ra thứ nộp hồ sơ) | −0.00002 | 1/4 | BỎ |
| 10 | **Thứ tự dòng / SK_ID_CURR** (`q0.py`): ID = thứ tự file (0…171201 train, tiếp theo là test). Tự tương quan nhãn lag 1/2/5 ≈ −0.001; trung bình trượt nhãn lân cận (cửa sổ 50/500/5000) AUC 0.497–0.500; tương quan đặc trưng với ID ≤ 0.008 | không có tín hiệu | — | Dữ liệu đã xáo trộn; không khai thác được (và không trôi theo thời gian) |

Tìm kiếm web vòng này (giải pháp Home Credit chỉ dùng bảng application, mẹo GBDT/credit scoring 2025–2026): không cho kỹ thuật nào khả thi trên CPU mà chưa có trong sổ. Các bài báo mới đều là mô hình nền dạng bảng (cần GPU ở 171k dòng) hoặc lấy mẫu lại (ADASYN/SMOTE — tài liệu ghi lợi ích AUC không có ý nghĩa thống kê).

##### Mức ưu tiên các đề xuất CŨ còn chưa làm (không tính là mới)
| Mục | Ưu tiên | Lý do |
|---|---|---|
| A1 vòng 1 #9 — LGB DART / extra_trees / linear_tree | Trung bình-thấp | Rẻ, chỉ để đa dạng stack; EBM (corr 0.98) đã cho +0.00001 nên kỳ vọng 0 … +0.0002 |
| A1 vòng 1 #5 — RealMLP / TabM | Thấp | Cần torch ~3 GB, 4 nhân CPU bị tranh; MLP sklearn đã cho ≈ 0; kỳ vọng 0 … +0.0005 ở stack |
| A1 vòng 1 #7 — Null Importance | Rất thấp | Ablation L2.9 và thí nghiệm OpenFE (thêm cột → giảm) đều cho thấy thêm/bớt cột ≈ 0 |
| L2.3 "phạt lãi suất cao" | Rất thấp | max_bin 1023 và phần dư RATE đều ≈ 0 → tín hiệu sản phẩm đã bão hòa |

##### (4) Ước lượng trần AUC
- Ba phép thăm dò độc lập cùng cho thấy không còn thông tin: OpenFE (tìm kiếm tự động tương tác bậc 2 trên 39 cột lõi) âm; phần dư định giá ≈ 0; chưng cất/đơn điệu/làm trơn (giảm phương sai) ≈ 0.
- Trần thực tế của 33 cột application với phương pháp hợp lệ: **≈ 0.771 – 0.772** (hiện 0.77081; phần còn lại chỉ là +0.0002 … +0.0008 từ thêm seed, path_smooth và mô hình đa dạng).
- **0.80 không đạt được** bằng phương pháp hợp lệ với dữ liệu này: khoảng cách 0.029 gấp ~10 lần mức tăng của vòng tốt nhất, và vòng 3 không tìm thêm được nguồn tín hiệu nào. Mức 0.80 ở Kaggle gốc đến từ 6 bảng lịch sử (bureau, previous_application, installments…) không có ở đây.

### Tác nhân 2 — Rút bài học từ Vòng 2 (`loop/round3_agent2.md`)
#### VÒNG 3 — Tác nhân 2 (Rút kinh nghiệm từ vòng 2)

Nguồn: `rules.md`, `history.md` (vòng 0–2), `round2_agent1.md`, `round2_agent2.md`, `fe_r2.py`, `run_r2.py`, `r2_screen.py`, `ens_r2.py`, `ens_final_r2.py`, `log_r2_*.txt`, OOF `*_r2.npy`.
Phân tích của tôi: `$S/a2r3/` — `c1.py`/`c1b.py` (chẩn đoán OOF, không huấn luyện), `c2.py` (lạc quan early-stopping; LGB + PCAT), `c3.py` (TE tổ hợp ở meta), `c_cat.py`/`c_cat1.py` (biến thể CatBoost), `c4.py`/`c5.py` (harness thông tin còn lại; khóa sản phẩm × mức tiền vay); log cùng tên `.log`. Không sửa repo.
Mốc: stack LR (lgbte_r2 + cat_r2 + xgb_r2 + lrbin_r2) = **0.77081** (tôi dựng lại đúng 0.77081, từng fold khớp history.md).

Chú thích: [ĐO] = tôi đo trực tiếp. [SUY] = suy luận chưa kiểm chứng.
Lưu ý độ tin cậy: máy bị tranh CPU nặng (load 13–17) và khởi động lại một lần giữa chừng; mọi số dưới đây đều lấy từ lần chạy đã hoàn tất. Sàng lọc CatBoost chỉ fold 0/1, 1–2 seed (nhiễu lớn, xem 3.4).

---------------------------------------------------------------------------------------------------
##### 0. Kiểm tra tính hợp lệ (yêu cầu riêng của vòng này)

| Hạng mục | Kết luận | Bằng chứng |
|---|---|---|
| PROF (trung bình theo khóa TERM×RATE, train+test) | KHÔNG rò nhãn | Đọc `fe_r2.py::profile`: chỉ dùng EXT/tuổi/thâm niên/giới, không có TARGET; dòng test cũng nhận đúng cột đó → OOF và test cùng phân phối. Hợp lệ theo ràng buộc (dùng test không nhãn). |
| IMP (điền khuyết EXT) | KHÔNG rò nhãn | `fe_r1.py::impute`: hồi quy EXT_k từ cột khác, KFold trên train+test, không TARGET. |
| TE PRODUCT trong LGB | Lồng đúng R14 | `run_r2.py`: train a = inner 5-fold trong a; valid b và test = từ toàn a; nhãn b không dùng. |
| PCAT trong CatBoost | Hợp lệ | ordered target statistics; valid mã hóa bằng nhãn train. Mức "rare" dựa trên đếm train+test (không nhãn). |
| Stack | Lồng đúng | `ens_r2.py::stack`: meta-LR fit trên 4 fold, chấm fold còn lại; chuẩn hóa trong fold. |
| **Early stopping trên chính fold chấm** | **Có lạc quan nhỏ, có từ vòng 0** | [ĐO] LGB lr 0.05, 5 fold: AUC tại vòng ES tốt nhất 0.76899 so với AUC tại số vòng = trung bình best_iter của 4 fold KHÁC 0.76874 → **+0.00025** (5/5 dương; bản có TE: +0.00026). Mọi mô hình tầng 1 (LGB/XGB/Cat, mỗi seed) đều chọn số vòng trên fold chấm. |
| Gộp OOF giữa các fold | Không lạc quan | [ĐO] AUC gộp = AUC trung bình fold = AUC sau khi xếp hạng trong từng fold (lgbte 0.77008 / 0.77012 / 0.77012). Calibration từng fold: độ dốc 0.96–1.03, mean p 0.079–0.081 vs tỉ lệ 0.0807. |
| ID | Không rò | [ĐO] SK_ID_CURR đã sắp tăng trong file; AUC(y ~ trung bình y của ±1/5/50/500 láng giềng theo ID) = 0.500/0.498/0.495/0.501; tương quan phần dư với phần dư láng giềng ≤ 0.005 → không có tín hiệu thời gian/lô theo ID. |

Kết luận: không có rò rỉ nhãn. OOF 0.77081 lạc quan cỡ **0.0002–0.0003** do early stopping (ước lượng trung thực ≈ 0.7705–0.7706). Mức lệch này như nhau ở mọi vòng nên so sánh giữa các vòng vẫn đúng; nhưng nó cùng cỡ với các "cải thiện" +0.0002–0.0003 đang được cân nhắc → xem R22.

##### 1. Thất bại vòng 2 & nguyên nhân

| Thử nghiệm | Kết quả | Nguyên nhân |
|---|---|---|
| PRODUCT làm categorical cho LGB | −0.0029 (0/4) | [ĐO, `c2.py B`] Tôi tái lập −0.00300 (0/4). Khi siết tham số phân loại (cat_smooth 1000, cat_l2 300, min_data_per_group 1000, max_cat_threshold 4) thì hết hại: **+0.00016 (2/4)**; mức vừa (300/100/300/8): −0.00016. → Nguyên nhân là cách LGB xử lý biến phân loại: sắp ~2,000 mức theo thống kê gradient của CHÍNH dữ liệu train rồi cắt tham lam, không có cơ chế "ordered" → mỗi lần cắt là một phép target-encoding rò trong train. Siết đến mức hết hại thì cũng hết lợi → không có cấu hình LGB-categorical nào đáng dùng. |
| Thêm khóa phân loại thứ hai cho CatBoost (TERM×RATE, C/GOODS, AMT_ANNUITY) | −0.0007 … −0.0014 | Cùng tín hiệu sản phẩm (R15); thêm biến nhiều mức làm tăng số tổ hợp CTR mà CatBoost phải ước lượng → nhiễu. Lưu ý các số này đo 1 seed, nhiễu ±0.001 (3.4) — hướng đúng nhưng độ lớn không chắc. |
| EBM | 0.76715, +0.00001 trong stack | corr 0.980 với GBDT; cùng lớp hàm (cộng tính + tương tác cặp ⊂ cây nông). |
| GBDT riêng theo nhóm | kém 0.005–0.011 trong chính nhóm | Nhóm yếu là thiếu thông tin; mô hình chung dùng được cấu trúc học từ 82–90% dữ liệu còn lại. |
| Pseudo-label, RATEG, điền khuyết thứ hai, TE nhân khẩu học, KNN meta, PP | ≈ 0 | Không có thông tin mới (đều là hàm của cột đã có). [ĐO, `c4.py H`] LGB nông boosting tiếp từ logit stack r2 trên TOÀN BỘ 129 cột mô hình: −0.0010 (50 vòng), −0.0017 (100), −0.0036 (300), 0/5 → stack đã rút hết phần GBDT rút được từ các cột hiện có; chỉ CỘT MỚI mới giúp. |
| XGB trong stack | hệ số 0.036 | [ĐO] bỏ xgb_r2: 0.77085, **+0.00004 (5/5)**. xgb–lgb Spearman 0.9977 → XGB thuần là chi phí (2 seed × 5 fold). |
| TE PRODUCT làm cột LGB (lgbte) | +0.0003 ở 5 fold | [ĐO] Ở lr 0.05, 5 fold, 1 seed: PROF+TE 0.76885 vs PROF 0.76899 (−0.00014). Lợi ích +0.0003 của lgbte_r2 nằm trong nhiễu seed; giá trị thật của lgbte trong stack: bỏ nó −0.00024 (0/5), nhưng một phần vì nó là mô hình LGB duy nhất còn lại. |

Thất bại phương pháp luận:
- Sàng lọc CatBoost 1 seed × 2 fold: [ĐO] cùng cấu hình PROF+PCAT, seed 0 = 0.76694/0.76959, seed 1 = 0.76577/0.76877 → chênh giữa hai seed **0.0010**. Mọi kết luận CatBoost cỡ < 0.001 từ 1 seed là không phân giải được.
- Kết luận "tín hiệu sản phẩm bão hòa quanh +0.003" rút ra khi mỗi biến thể chỉ khác nhau khóa hoặc số cột; chưa thử làm MỊN khóa theo mức tiền vay (xem 3.2).

##### 2. Mẹo từ thành công vòng 2

1. **PROF hiệu quả vì nó là "hạng rủi ro mà ngân hàng gán cho sản phẩm", đo bằng chính quần thể mua sản phẩm đó.** Không dùng nhãn → không lệch inner/outer (thứ đã giết KNN vòng 1), dùng thêm 61.5k dòng test, và là cột số đơn điệu mà cây nông cắt được bằng 1–2 nhát (thay vì ~2,000 khoảng hẹp trên C/A).
   [ĐO] Phân rã: AUC của "trung bình stack theo PRODUCT" = 0.691; AUC của phần lệch trong sản phẩm = 0.700; stack 0.7708 → khoảng một nửa sức phân biệt là GIỮA các sản phẩm. Lợi ích PROF theo số dòng của sản phẩm (cat_r2 − cat_r1): 30–200 dòng **+0.0067**, 200–1000 +0.0040, > 1000 +0.0012, < 30 +0.0024 → lợi lớn nhất ở sản phẩm cỡ vừa (đủ dòng để trung bình ổn định, đủ hẹp để đặc trưng cho một phân khúc).
2. **PCAT hiệu quả cho CatBoost vì HAI cơ chế, và cơ chế thứ hai lớn hơn** [ĐO, `c_cat*.py`, lr 0.1, fold 0/1]:
   - ordered TS của riêng PRODUCT (`max_ctr_complexity=1`): 0.76689 (seed 0) / 0.76646 (seed 1) so với không PCAT 0.76621 → chỉ khoảng +0.0005;
   - cho phép TỔ HỢP (mặc định `max_ctr_complexity=4`): 0.76826 / 0.76727 → tắt tổ hợp mất **−0.00138 (seed 0), −0.00081 (seed 1)**; 3/4 lần chấm âm, 1 hòa.
   CatBoost tự tạo target-statistics cho PRODUCT × (biến phân loại khác và các nhát cắt số đã dùng trong cây) — tức "rủi ro của sản phẩm NÀY đối với loại khách NÀY", thứ mà một cột TE/PROF đơn không biểu diễn được.
3. **Cùng biến hại LGB** vì LGB không có ordered statistics (mục 1, dòng 1). Mẹo tổng quát: biến phân loại nhiều mức mang tín hiệu nhãn → CatBoost native; LGB/XGB chỉ nhận dạng không nhãn (PROF) hoặc TE lồng fold.
4. **CatBoost là nguồn đa dạng duy nhất đáng giá**: [ĐO] bỏ cat_r2 khỏi stack −0.00057 (lớn nhất), bỏ lgbte −0.00024, bỏ lrbin −0.00015, bỏ xgb +0.00004. Spearman cat–lgbte 0.982, lgb–xgb 0.998, lrbin–GBDT 0.92–0.93.
5. **Trung bình seed lợi cho CatBoost gấp đôi LGB**: [ĐO từ log] AUC fold 1 seed → trung bình seed: CatBoost 0.76910 → 0.76950 (**+0.00040** với 2 seed); LGB 0.76961 → 0.76981 (+0.00020 với 3 seed); XGB +0.00015. CatBoost còn nhiều phương sai seed chưa khử.
6. **Tham số CatBoost mặc định đang tốt** [ĐO, seed 0, so với PCAT 0.76826]: depth 4 −0.00056; l2_leaf_reg 20 −0.00050; rsm 0.3 −0.00117. (Đều trong nhiễu ±0.001 nhưng không cái nào dương → không có lý do đổi.)

##### 3. Phát hiện bất ngờ (đều [ĐO])

1. **Hồ sơ sản phẩm CHƯA bão hòa hẳn — làm mịn khóa theo mức tiền vay còn cho thêm.** Nhóm **PQ** = hồ sơ không nhãn (trung bình EXTF_MEAN, EXT_3, tuổi, thâm niên; train+test; < 30 dòng → NaN) theo khóa `TERM×RATE(0.5%) × ngũ phân vị AMT_CREDIT`, 4 cột, thêm vào PROF. LGB lr 0.05, **5 fold**, ghép cặp cùng seed (`c4.py S`, `c5.py`):

   | Seed | PROF | PROF + PQ | Chênh từng fold | TB | Dương |
   |---|---|---|---|---|---|
   | 0 | 0.76899 | 0.76947 | +0.00046 −0.00014 −0.00002 +0.00110 +0.00104 | +0.00049 | 3/5 |
   | 1 | 0.76870 | 0.76930 | +0.00007 +0.00111 +0.00120 +0.00031 +0.00032 | +0.00060 | 5/5 |
   | 2 | 0.76876 | 0.76939 | +0.00035 +0.00043 +0.00035 +0.00156 +0.00046 | +0.00063 | 5/5 |

   → **+0.00057, 13/15 dương** ở mô hình LGB đơn. CẢNH BÁO: harness trên stack r2 chỉ với 4 cột PQ = −0.0001 (ngang cột nhiễu −0.00015) → ở mức stack lợi ích có thể nhỏ hơn nhiều (kỳ vọng 0 … +0.0003). Chưa đo ở 5 fold lr 0.01, chưa đo cho CatBoost.
2. **Ở tầng meta, TE tổ hợp PRODUCT × ngũ phân vị AMT_CREDIT là cột tốt nhất** (`c3.py`, α = 10): trên stack không có cat và không có TE (0.76989): +0.00042 (5/5), so với TE PRODUCT thường +0.00023; trên stack đầy đủ bỏ xgb (0.77085): **+0.00013 (5/5) → 0.77098**. Các tổ hợp khác trên stack đầy đủ: × REGION_RATING +0.00009 (4/5), × EXT3_NA +0.00007 (5/5), × giới +0.00006 (5/5), × nhà ở +0.00005 (5/5); × học vấn / loại thu nhập / gia đình / nghề / tuổi / EXT / bảo hiểm / thâm niên ≈ 0. Tất cả dưới ngưỡng R2 → không dùng ở meta, nhưng cùng chỉ về một hướng với PQ: **mức tiền vay tách tiếp sản phẩm**.
   Làm cột cho LGB thì KHÔNG giúp: PROF + TE(P×Cq5) −0.00006 (2/5); PROF + TE(P) + TE(P×Cq5) +0.00020 (2/5) → dạng không nhãn (PQ) tốt hơn dạng TE, lặp lại đúng bài học PROF > TE của vòng 2.
3. **Stack đã rút hết thông tin từ cột hiện có** (harness toàn bộ cột: −0.0010 … −0.0036, 0/5). Mọi hướng "mô hình khác trên cùng cột" có kỳ vọng ≈ 0.
4. **Nhiễu seed của CatBoost ở sàng lọc ±0.001** (mục 1). So với LGB sàng lọc (±0.0005) thì gấp đôi.
5. **Bỏ XGB làm stack TĂNG** (+0.00004, 5/5) — nhỏ nhưng nhất quán; tiết kiệm ~1/3 thời gian tầng 1.
6. **CatBoost kém LGB rõ nhất ở các nhóm**: sản phẩm < 30 dòng (gộp thành `rare`: −0.0033), TERM_HAT 60 (−0.0036), tuổi > 60 (−0.0023), học dở đại học (−0.0043); hơn LGB ở TERM_HAT 10/12/24 (+0.0005 … +0.0014), tuổi 30–40 (+0.0008), sản phẩm > 1000 dòng (+0.0006). Tức PCAT mạnh ở sản phẩm phổ biến, yếu ở sản phẩm hiếm — hợp lý với ordered TS.
7. **Revolving vẫn không có gì**: cat_r2 − cat_r1 chỉ +0.0006 (Cash +0.0043); TE theo hạn mức (AMT_CREDIT) của Revolving ở meta −0.00001. Nhóm này (9.5%) không có khóa sản phẩm dùng được trong 33 cột.

##### 4. Cách ghép kỹ thuật & tối ưu logic cho vòng 3 — quy tắc cho Tác nhân 3

Giữ R1–R21. Bổ sung:

- **R22 (báo cáo AUC không early-stopping).** Với cấu hình cuối, ngoài OOF thông thường, báo thêm OOF "số vòng cố định" (mỗi fold dùng số vòng = trung bình best_iter của 4 fold kia) ít nhất cho LGB. Cải thiện < 0.0003 giữa hai cấu hình chỉ được coi là thật nếu dương cả ở bản số vòng cố định.
- **R23 (sàng lọc CatBoost).** Tối thiểu 2 seed × fold 0/1 (4 lần chấm ghép cặp cùng seed), ngưỡng nhận +0.001; 1 seed không đủ.
- **R24 (dạng của tín hiệu nhóm).** Thứ tự ưu tiên khi có một khóa nhóm mới: (1) hồ sơ KHÔNG nhãn train+test cho mọi GBDT; (2) biến phân loại gốc cho CatBoost (chỉ MỘT khóa); (3) TE lồng fold — chỉ khi (1) không khả thi. Không bao giờ đưa khóa nhiều mức làm categorical cho LGB/XGB.
- **R25 (cấu hình ensemble).** Stack LR = {LGB (có TE hoặc không — chênh trong nhiễu), CatBoost + PCAT nhiều seed, lrbin}; KHÔNG XGB. Dồn thời gian XGB sang seed CatBoost.

Cái gì cộng dồn / không cộng dồn (cập nhật):
- CỘNG DỒN: IMP + RATE/TERM + PROF (ba nguồn khác nhau); PROF + PCAT trong CatBoost (+0.002 sàng lọc, nhờ tổ hợp); CatBoost + LGB trong stack (+0.0006); lrbin (+0.00015); seed CatBoost (+0.0004 cho 2 seed).
- KHÔNG CỘNG DỒN: PROF + TE + TE ở meta (một tín hiệu); PROF + PROFX/PROFR/PROFC; XGB + LGB; EBM + GBDT; khóa phân loại thứ hai trong CatBoost; mô hình vòng trước + mô hình vòng này.
- CHƯA RÕ (cần đo ở stack): PROF + PQ (LGB đơn +0.0006).

Thứ tự đề nghị cho vòng 3: (1) sàng lọc PQ theo R3 cho LGB và R23 cho CatBoost → nếu giữ, thêm vào bộ đặc trưng; (2) chạy lại LGB (3 seed) + CatBoost (5 seed) + lrbin trên bộ chốt, bỏ XGB; (3) stack LR, so với 0.77081 từng fold; (4) báo OOF số vòng cố định (R22); (5) bảng nhóm R20.

##### 5. Chẩn đoán điểm nghẽn & hướng ưu tiên

Điểm nghẽn vẫn là THÔNG TIN. Bằng chứng mới: harness toàn bộ cột âm; mọi biến thể mô hình ≈ 0; nhóm yếu không đổi (tuổi > 60: 0.734; TERM_HAT 48: 0.735; < 30 tuổi: 0.742; thiếu 2 EXT: 0.753; sản phẩm < 30 dòng: 0.718).
Dự báo trung thực: các hướng dưới đây cộng lại kỳ vọng **+0.0003 … +0.0008** → ≈ 0.7711–0.7716. **0.80 không đạt được bằng phương pháp hợp lệ với 33 cột này**; sau ba vòng chưa có bằng chứng nào ngược lại, và khoảng cách còn lại (0.029) gấp ~40 lần tổng kỳ vọng của vòng 3.

Hướng ưu tiên (kỳ vọng / chi phí):
1. **L3.3 — PQ**: hồ sơ không nhãn theo khóa TERM×RATE × ngũ phân vị AMT_CREDIT (4 cột), cho LGB và CatBoost. LGB đơn +0.0006 (13/15). Chi phí: vài giây + chạy lại tầng 1.
2. **L3.2 — CatBoost 5 seed** (PROF + PCAT [+ PQ]). Kỳ vọng +0.0002–0.0003 ở mô hình, +0.0001–0.0002 ở stack. Chi phí ~3 phút/seed/fold khi máy rảnh.
3. **L3.1 — bỏ XGB khỏi stack** (+0.00004, 5/5; tiết kiệm thời gian cho mục 2).
4. **L3.4 — báo OOF số vòng cố định** (R22) để con số cuối trung thực.
5. **L2.3b — phạt lãi suất cao khi dựng kỳ hạn**: mục tồn duy nhất của vòng 2; kỳ vọng ≈ 0 (bản EM của A1 đã ≈ 0) — làm để đóng checklist, 1 lần sàng lọc.

##### 6. CHECKLIST BÀI HỌC

| Mã | Hướng / bài học có thể thực thi | Trạng thái |
|---|---|---|
| L1.1 | Điền khuyết EXT bằng mô hình + RATE_n, chạy đủ LGB/Cat/XGB, rank-avg | ĐÃ LÀM (vòng 1: LGB 0.76696, XGB 0.76643, Cat 0.76545; rank-avg 0.76751) |
| L1.1b | Kỳ hạn bổ sung n=6,10,18,30,42; RATE theo loại hợp đồng | ĐÃ LÀM (vòng 1, trong TERM; +0.0021 sàng lọc) |
| L1.1c | RATE với AMT_GOODS_PRICE làm gốc; điền khuyết EXT bằng thuật toán thứ hai rồi trung bình | ĐÃ LÀM (vòng 2: RATEG −0.00004 (2/4); Ridge −0.00001; trung bình LGB+Ridge +0.00015 (2/4) → bỏ) |
| L1.2 | KNN target-mean lồng fold làm đặc trưng | ĐÃ LÀM (vòng 1: −0.0005…−0.0007, 1/4 → bỏ) |
| L1.2b | KNN làm cột ở tầng meta / mô hình riêng | ĐÃ LÀM (vòng 2: +0.00002 … 0 → bỏ) |
| L1.3 | Mô hình khác họ: LR-binning, MLP | ĐÃ LÀM (vòng 1: lrbin giữ, MLP bỏ; vòng 2: lrbin_r2 0.75851, +0.00014 trong stack → giữ) |
| L1.4 | Multi-seed cấu hình cuối cho cả 3 thuật toán | ĐÃ LÀM (vòng 2: LGB 3 seed, CatBoost 2 seed, XGB 2 seed) |
| L1.5a | Pseudo-label test | ĐÃ LÀM (vòng 2: mềm −0.00003 (2/4); cứng +0.00027 (4/4, dưới ngưỡng) → bỏ) |
| L1.5b | Mô hình/đặc trưng riêng cho nhóm yếu | ĐÃ LÀM (vòng 2, gộp L2.4) |
| R1–R12 | Quy tắc đánh giá/phân bổ/ghép vòng 1 | ĐÃ LÀM (vòng 1–2, áp dụng) |
| L2.1 | TE PRODUCT (α=10) làm cột meta-LR, lồng R14 | ĐÃ LÀM (vòng 2: trên mô hình vòng 1 +0.00155 (5/5); trên mô hình vòng 2 +0.00013 … +0.00002 → không dùng ở meta) |
| L2.2 | Tín hiệu sản phẩm vào tầng 1: (a) cột TE cho GBDT; (b) PRODUCT categorical cho CatBoost | ĐÃ LÀM (vòng 2: (a) lgbte_r2 0.77008, +0.00030 vs lgb_r2, giữ; (b) CatBoost +0.00205 sàng lọc, cat_r2 0.76947, giữ) |
| L2.3 | Sửa dựng kỳ hạn bằng posterior/EM | ĐÃ LÀM (vòng 2 bởi A1: EM −0.00001, 2/4 → bỏ) |
| L2.3b | Sửa dựng kỳ hạn bằng phạt/tiên nghiệm lãi > 35–40%/năm khi chọn TERM_HAT; sàng lọc R3 trên bộ PROF (dựng lại TERM + PROF theo khóa mới) | CHƯA LÀM |
| L2.4 | Nhóm yếu ở tầng 1: TE (INCOME×EDU×tuổi), GBDT riêng theo nhóm | ĐÃ LÀM (vòng 2: TE −0.00002 (1/5) / −0.00001 (2/4); GBDT riêng kém 0.005–0.011 trong nhóm, trong stack −0.00010 … +0.00001 → bỏ) |
| L2.5 | Meta = LR đơn giản (R17) | ĐÃ LÀM (vòng 2, áp dụng) |
| L2.6 | Không đưa mô hình tập con thông tin vào stack (R18) | ĐÃ LÀM (vòng 2: bản 8 mô hình +0.00009 → loại; EBM corr 0.98 → loại) |
| L2.7 | Ghép khách hàng lặp lại (uid) | KHÔNG THỰC THI ĐƯỢC có ích (chỉ 124 dòng trùng khi buộc EXT1 bằng nhau; nhóm trùng lẫn nhãn) |
| L2.8 | Bảng AUC theo nhóm mỗi vòng (R20) | ĐÃ LÀM (vòng 2, bảng trong history.md) — tiếp tục mỗi vòng |
| L2.9 | Ablation TERM_HAT, `_RES` (R19) | ĐÃ LÀM (vòng 2: +0.00013 (3/4), +0.00005 (2/4) → giữ nguyên) |
| R13–R21 | Quy tắc vòng 2 | ĐÃ LÀM (vòng 2, áp dụng) |
| L3.1 | Bỏ xgb_r2 khỏi stack cuối (stack = LGB + CatBoost + lrbin); không chạy lại XGB | CHƯA LÀM (A2 đo sơ bộ: 0.77085, +0.00004, 5/5) |
| L3.2 | CatBoost PROF + PCAT với 5 seed (thay vì 2), đưa vào stack | CHƯA LÀM (cơ sở: 2 seed đã cho +0.00040 so với 1 seed) |
| L3.3 | Nhóm PQ: hồ sơ không nhãn train+test (mean EXTF_MEAN, EXT_3, DAYS_BIRTH, DAYS_EMPLOYED; < 30 dòng → NaN) theo khóa TERM×RATE(0.5%) × ngũ phân vị AMT_CREDIT, thêm vào PROF; sàng lọc R3 (LGB) và R23 (CatBoost), nếu giữ thì chạy đủ 5 fold và stack. Mã dựng: `a2r3/c4.py` (biến `PQ`) | CHƯA LÀM (A2 đo sơ bộ LGB lr 0.05, 5 fold × 3 seed: +0.00057, 13/15; harness trên stack ≈ 0) |
| L3.4 | Báo OOF "số vòng cố định" (số vòng = TB best_iter của 4 fold khác) cho mô hình cuối, ít nhất LGB, và ghi vào history.md (R22) | CHƯA LÀM (A2 đo: lạc quan ES +0.00025 ở LGB lr 0.05) |
| L3.5 | TE lồng fold của PRODUCT × ngũ phân vị AMT_CREDIT: cột meta và cột LGB | ĐÃ LÀM (vòng 3 bởi A2: meta +0.00013 (5/5) trên stack đầy đủ, dưới ngưỡng; cột LGB −0.00006 (2/5) → không dùng; hướng này chuyển thành L3.3) |
| L3.6 | LGB + PRODUCT categorical với regularization phân loại mạnh | ĐÃ LÀM (vòng 3 bởi A2: −0.00300 mặc định; −0.00016 / +0.00016 (2/4) khi siết → không dùng) |
| L3.7 | Biến thể tham số CatBoost trên PROF + PCAT (max_ctr_complexity 1, depth 4, l2 20, rsm 0.3) | ĐÃ LÀM (vòng 3 bởi A2, sàng lọc: −0.00138/−0.00081, −0.00056, −0.00050, −0.00117 → giữ mặc định) |
| L3.8 | TE tổ hợp PRODUCT × {giới, học vấn, loại thu nhập, gia đình, vùng, EXT, tuổi, bảo hiểm, nghề, nhà ở, EXT3_NA, thâm niên} và TE hạn mức Revolving ở meta | ĐÃ LÀM (vòng 3 bởi A2: tất cả ≤ +0.00009 trên stack đầy đủ → không dùng) |
| L3.9 | Đặc trưng láng giềng theo SK_ID_CURR (thứ tự hồ sơ) | KHÔNG THỰC THI ĐƯỢC có ích (A2 đo: AUC 0.495–0.501, tương quan phần dư ≤ 0.005) |
| R22–R25 | Quy tắc vòng 3 (OOF số vòng cố định; sàng lọc CatBoost ≥ 2 seed; thứ tự dạng tín hiệu nhóm; ensemble không XGB) | CHƯA LÀM (áp dụng trong vòng 3) |

Tổng: **6 dòng CHƯA LÀM** — L2.3b, L3.1, L3.2, L3.3, L3.4, R22–R25.

### Tác nhân 3 — Thực hành (dựng lại từ mã và nhật ký)
> **Ghi chú nguồn:** Tác nhân 3 vòng 3 bị dừng vì hết hạn mức tuần của model (13/11 ~08:00 UTC) **trước khi tự viết báo cáo**. Toàn bộ mục này do phiên chính dựng lại từ mã (`fe_r3.py`, `run_r3.py`, `div_r3.py`, `tabm_r3.py`, `ens_final_r3.py`, `queue1/2/3_r3.sh`) và nhật ký (`log_r3_*.txt`) mà Tác nhân 3 để lại; bước ghi submission (`ens_final_r3.py write`) do phiên chính chạy.

##### Đầu vào
- Tác nhân 1 vòng 3: `STATUS: NO_NEW_METHODS` → đây là vòng cuối theo điều kiện dừng 2.
- Danh sách bắt buộc: (A) checklist Tác nhân 2 — L2.3b, L3.1, L3.2, L3.3, L3.4, R22–R25; (B) việc cũ của Tác nhân 1 — LGB DART / extra_trees / linear_tree, Null Importance, RealMLP/TabM; (C) tùy chọn path_smooth = 50.

##### Mã đã viết
- `fe_r3.py`: `load_r3()` = đặc trưng vòng 2 + nhóm **PQ** (4 cột: hồ sơ không nhãn theo khóa TERM × RATE(0.5%) × ngũ phân vị AMT_CREDIT, tính trên train+test, nhóm < 30 dòng → NaN) + `term_alt()` dựng lại kỳ hạn có phạt lãi suất cao (L2.3b). Không cột nào dùng TARGET.
- `run_r3.py`: huấn luyện tầng 1 với **số vòng cố định** và lưu đường cong AUC để báo **cả hai** con số (L3.4/R22): (ES) dự đoán tại số vòng early stopping mô phỏng (patience 300 trên fold chấm, như vòng 0–2) và (FIX) dự đoán tại số vòng = trung bình số vòng ES của 4 fold khác → không dùng nhãn fold chấm. Cờ: TE, PCAT, PS50, ET, LT, DART, LR, NR, NI.
- `div_r3.py` (lrbin), `tabm_r3.py` (TabM-mini + piecewise-linear embeddings, torch CPU; early stopping trên 10% tách từ phần train của fold), `ens_final_r3.py` (so sánh rank-avg / stacking LR lồng fold và ghi submission).
- Hàng đợi chạy lại được: `queue1_r3.sh`, `queue2_r3.sh`, `queue3_r3.sh` (bỏ qua phần đã có; checkpoint theo fold/seed trong `r3m/`).

##### Sàng lọc (LightGBM lr 0.05, nhiều seed × fold)
`log_r3_screen1.txt` (4 lần chấm):

| Cấu hình | AUC TB | so BASE | dương |
|---|---|---|---|
| BASE (IMP+RATE+TERM+PROF, 129 cột) | 0.76804 | — | — |
| PQ | 0.76842 | +0.00038 | 3/4 |
| T_soft35 (phạt lãi suất mềm, L2.3b) | 0.76836 | +0.00032 | 3/4 |
| T_hard40 (phạt cứng, L2.3b) | 0.76817 | +0.00013 | 4/4 |
| T_soft35 + PQA | 0.76856 | +0.00052 | 3/4 |
| PS50 (path_smooth 50) | 0.76818 | +0.00014 | 3/4 |
| PQ + PS50 | 0.76869 | +0.00065 | 4/4 |
| ET (extra_trees) | 0.76743 | −0.00061 | 1/4 |
| LT (linear_tree) | 0.76730 | −0.00074 | 1/4 |

`log_r3_screen2.txt` (6 lần chấm khác, seed 2–4):

| Cấu hình | AUC TB | so BASE | dương |
|---|---|---|---|
| BASE | 0.76938 | — | — |
| PQ | 0.77004 | +0.00066 | 5/6 |
| T_soft35 | 0.76902 | −0.00036 | 1/6 |
| T_soft35 + PQA | 0.77021 | +0.00083 | 5/6 |
| PQ + PS50 | 0.77004 | +0.00066 | 6/6 |
| PS50 | 0.76950 | +0.00012 | 4/6 |

→ PQ giữ (dương ổn định). Phạt lãi suất (L2.3b) đảo dấu giữa hai đợt (+0.00032 → −0.00036) → loại; phần cộng thêm của T_soft35+PQA nằm ở PQ. PS50 chỉ +0.00012–0.00014 → **không đưa vào mô hình cuối** (lgbte_r3 dùng spec `+PROF+PQ~TE`, không có PS50).

`log_r3_screen3.txt`:

| Cấu hình | AUC TB | so BASE | dương |
|---|---|---|---|
| NI0 (Null Importance ngưỡng 0, giữ 88–92/129 cột) | 0.76737 | −0.00067 | 1/4 |
| NI1 (ngưỡng 1, giữ 33–36/129 cột) | 0.76200 | −0.00604 | 0/4 |
| DART (lr 0.1) | 0.76677 | −0.00127 | 0/4 |

→ Null Importance loại (bỏ cả những biến có ích như AMT_CREDIT, AMT_ANNUITY, DAYS_BIRTH).

Sàng lọc CatBoost có/không PQ (2 fold × 2 seed, lr 0.1, `log_r3_catscrA/B.txt`): A (PROF+PCAT) ES 0.76694/0.76577/0.76959/0.76877; B (PROF+PQ+PCAT) 0.76577/0.76784/0.76765/0.76882 → TB −0.00025, 2/4 → **CatBoost không dùng PQ**.

##### Huấn luyện đủ 5 fold

| Mô hình | Spec | Seed | OOF ES | OOF FIX | Folds (ES) |
|---|---|---|---|---|---|
| lgbte_r3 | LGB `+PROF+PQ~TE`, lr 0.01 | 0,1,2 | 0.77048 | 0.77032 | 0.7677 / 0.77138 / 0.77611 / 0.7671 / 0.77032 |
| catn_r3 | Cat `+PROF~PCAT`, lr 0.03 | 2,3,4 | 0.76968 | 0.76939 | 0.76819 / 0.77007 / 0.77414 / 0.76613 / 0.77011 |
| cat_r3 = 0.4·cat_r2 + 0.6·catn_r3 (**5 seed**, L3.2) | — | 0–4 | 0.76975 | — | 0.76838 / 0.77027 / 0.77402 / 0.76618 / 0.77008 |
| lrbin_r3 | LR trên đặc trưng phân khoảng (2074 cột) | — | 0.75917 | — | 0.75817 / 0.75886 / 0.76277 / 0.75541 / 0.76078 |
| dart_r3 | LGB DART lr 0.1, 1200 vòng | 0 | 0.76842 | 0.76842 | 0.76667 / 0.76892 / 0.77357 / 0.76527 / 0.76784 |
| et_r3 | LGB extra_trees lr 0.03 | 0 | 0.76960 | 0.76932 | 0.76642 / 0.77088 / 0.77413 / 0.76694 / 0.76982 |
| lt_r3 | LGB linear_tree lr 0.03 | 0 | 0.76963 | 0.76921 | 0.76639 / 0.76989 / 0.77506 / 0.76716 / 0.76973 |
| tabm_r3 | TabM seed 0 | 0 | 0.76654 | — | 0.76501 / 0.76843 / 0.77264 / 0.7636 / 0.76991 |
| tabmB_r3 | TabM seed 1 | 1 | 0.76785 | — | 0.76761 / 0.76844 / 0.77108 / 0.76519 / 0.76874 |
| tabmC_r3 | TabM seed 2 | 2 | 0.76575 | — | 0.76545 / 0.76779 / 0.7713 / 0.76403 / 0.76794 |
| tabm3_r3 = TB 3 seed TabM | — | 0,1,2 | 0.76887 | — | 0.7679 / 0.76923 / 0.77358 / 0.7661 / 0.77 |

Sự cố khi chạy:
- Cài torch từ `download.pytorch.org` bị proxy chặn (403) → cài torch 2.15.0 CPU từ PyPI vào `pylib_torch` (thành công), gói `tabm` + `rtdl_num_embeddings` vào `pylib_tabm`.
- Lần chạy TabM đầu lỗi `compute_bins`: có cột chỉ một giá trị trong fold → đã sửa và chạy lại.
- Các lệnh dài chia theo fold/seed với timeout; một số lần bị ngắt được chạy tiếp từ checkpoint.

##### Kết hợp (`log_r3_ens2.txt`, stacking LR lồng fold; mốc = stack vòng 2 0.77081)

| Bộ | OOF | so mốc r2 | dương |
|---|---|---|---|
| r2 bỏ XGB (L3.1) | 0.77085 | +0.00004 | 5/5 |
| rank_core3 (lgbte_r3, cat_r3, lrbin_r3) | 0.77050 | −0.00031 | 0/5 |
| **stack_core3** | 0.77124 | +0.00044 | 5/5 |
| stack_core3 + xgb_r2 | 0.77126 | (so core3 +0.00002, 3/5) | — |
| stack_core3 + lgbte_r2, cat_r2 | 0.77124 | (so core3 −0.00001, 2/5) | — |
| stack_core3 + tabm_r3 (1 seed) | 0.77130 | (so core3 +0.00012, 4/5) | — |
| stack_core3 + dart_r3 | — | (so core3 +0.00001, 2/5) | — |
| stack_core3 + et_r3 | — | (so core3 +0.00016, 3/5) | — |
| stack_core3 + lt_r3 | — | (so core3 +0.00014, 4/5) | — |
| stack_core3 + tabm3_r3 | — | (so core3 +0.00024, 4/5) | — |
| **stack_core3 + alldiv (dart, et, lt, tabm3)** | **0.77149** | +0.00070 (so core3 +0.00026) | **5/5** |
| rank_core3 + alldiv | 0.77144 | +0.00061 | 5/5 |
| stack ES (lgbte_r3, catn_r3, lrbin_r3) so với stack FIX tương ứng | — | FIX thấp hơn ES ≈ 0.00021 (0/5) | — |

(Bản `log_r3_ens1.txt` trước đó dùng TabM 1 seed: stack_core3 + alldiv = 0.77143.)

Hệ số stack có TabM 3 seed (core3 + tabm3): lgbte_r3_fix 0.339, cat_r3_fix 0.307, lrbin_r3 0.118, tabm3_r3 0.318. Tương quan hạng (Spearman) với stack_core3: lgbte 0.9955, cat 0.993, lrbin 0.9438, dart 0.9834, et 0.9864, lt 0.986, tabm 0.9708, tabm3 0.9792.

##### Kết quả cuối
- Chọn **stacking LR lồng fold trên lgbte_r3 + cat_r3 + lrbin_r3 + dart_r3 + et_r3 + lt_r3 + tabm3_r3**.
- OOF AUC **0.77149** (vòng 2: 0.77081; +0.00070, 5/5 fold). Folds: 0.76923 / 0.77219 / 0.77622 / 0.76827 / 0.77190.
- OOF "số vòng cố định" (L3.4): tầng 1 FIX thấp hơn ES 0.00016–0.00042 mỗi mô hình; stack FIX thấp hơn ES ≈ 0.0002 → OOF trung thực ước ≈ 0.7713.
- `loop/sub_round3.csv`: 61.500 dòng, đúng thứ tự sample_submission, không NaN; Spearman với sub_round2 = 0.9977. Đã chép sang `com/submission.csv`.

##### Bảng trạng thái từng mã

| Mã | Nội dung | Trạng thái |
|---|---|---|
| L2.3b | Phạt lãi suất cao khi dựng kỳ hạn | ĐÃ LÀM (vòng 3): T_soft35 +0.00032 (3/4) rồi −0.00036 (1/6); T_hard40 +0.00013 → loại |
| L3.1 | Bỏ XGB khỏi stack | ĐÃ LÀM: r2 +0.00004 (5/5); ở r3 thêm xgb_r2 +0.00002 → không dùng XGB |
| L3.2 | CatBoost 5 seed | ĐÃ LÀM: cat_r3 0.76975 vs cat_r2 0.76947 → giữ |
| L3.3 | Nhóm PQ vào tầng 1 | ĐÃ LÀM: LGB +0.00066 (5/6, 4/4 cùng PS50); lgbte_r3 0.77048 vs lgbte_r2 0.77008 → giữ cho LGB; CatBoost −0.00025 (2/4) → không dùng |
| L3.4 | OOF số vòng cố định | ĐÃ LÀM: bảng trên |
| R22 | Báo OOF không early stopping | ĐÃ ÁP DỤNG |
| R23 | Sàng lọc CatBoost ≥ 2 seed | ĐÃ ÁP DỤNG (catscrA/B 2 seed × 2 fold) |
| R24 | Thứ tự dạng tín hiệu nhóm | ĐÃ ÁP DỤNG (PQ đo trên nền PROF) |
| R25 | Ensemble không XGB | ĐÃ ÁP DỤNG |
| A1 #9 | LGB DART / extra_trees / linear_tree | ĐÃ LÀM: đơn lẻ kém lgbte (0.76842 / 0.76960 / 0.76963) nhưng đa dạng hóa stack → giữ trong stack |
| A1 #7 | Null Importance | ĐÃ LÀM: −0.00067 / −0.00604 → loại |
| A1 #5 | RealMLP / TabM | ĐÃ LÀM (TabM; RealMLP không làm riêng vì cùng vai trò mạng nơ-ron đa dạng hóa): TabM 3 seed 0.76887, +0.00024 trong stack → giữ |
| A1 R3 | path_smooth = 50 | ĐÃ LÀM: +0.00012–0.00014 riêng → dưới ngưỡng, không dùng |

Không còn mục CHƯA LÀM → **điều kiện dừng 2 thỏa; vòng lặp kết thúc.**

---

# PHẦN III — Kết luận chung

| Vòng | AUC (OOF 5 fold) |
|---|---|
| 0 | 0.76273 |
| 1 | 0.76790 |
| 2 | 0.77081 |
| 3 | **0.77149** (ước lượng trung thực ≈ 0.7713) |

- Mục tiêu 0.80 chưa đạt. Mọi tác nhân qua các vòng đều ước lượng trần hợp lệ với 33 cột hồ sơ vay khoảng 0.771–0.772; để vượt mức đó cần thêm dữ liệu (lịch sử tín dụng, các khoản vay trước) vốn không có trong bài.
- Vòng lặp dừng theo điều kiện 2 (Tác nhân 1 báo NO_NEW_METHODS và mọi mục checklist đã thực hiện).

## Những phần KHÔNG có bản gốc / còn thiếu
1. **Vòng 0** không có tác nhân; nội dung là kết quả do phiên chính ghi vào sổ chung.
2. **Tác nhân 3 vòng 3** không tự viết báo cáo (bị dừng vì hết hạn mức tuần); mục của nó được phiên chính dựng lại từ mã và nhật ký để lại.
3. **RealMLP** (A1 #5) chưa được chạy riêng; chỉ TabM được chạy làm đại diện mạng nơ-ron.
4. Mã nguồn, nhật ký và dự đoán OOF/test chỉ nằm trong thư mục tạm của phiên (không đưa vào repo theo yêu cầu "không tạo file nào khác") và sẽ mất khi phiên kết thúc.
