# Mã thí nghiệm — bài thi Home Credit (bản lưu trữ nguyên trạng)

Mã gốc của vòng 0 và vòng lặp 3 tác nhân (vòng 1–3), chép nguyên trạng từ thư mục tạm của phiên làm việc để không bị mất. Chi tiết từng vòng và từng tác nhân xem `LICH_SU_LAM_VIEC.md` ở gốc repo.

## Cấu trúc

| Thư mục | Nội dung | Ai viết |
|---|---|---|
| `vong0/` | Đặc trưng ban đầu (`fe.py`), huấn luyện (`run.py`), Optuna (`opt.py`), stacking | Phiên chính |
| `vong1/` | Điền khuyết EXT, lãi suất ngầm, kỳ hạn vay; huấn luyện + kết hợp vòng 1 | Tác nhân 3 vòng 1 |
| `vong2/` | Hồ sơ sản phẩm vay (PROF), TE PRODUCT, PRODUCT cho CatBoost; kết hợp vòng 2 | Tác nhân 3 vòng 2 |
| `vong3/` | Nhóm PQ, CatBoost 5 seed, DART/ET/LT, TabM; kết hợp cuối (`ens_final_r3.py`) | Tác nhân 3 vòng 3 |
| `tac_nhan_1/vong1–3/` | Thí nghiệm sàng lọc phương pháp mới | Tác nhân 1 |
| `tac_nhan_2/vong1–3/` | Phân tích, chẩn đoán, thí nghiệm rút bài học | Tác nhân 2 |

Chuỗi tạo `com/submission.csv` hiện tại (AUC OOF 0.77149): `vong3/fe_r3.py` → `vong3/run_r3.py`, `vong3/div_r3.py`, `vong3/tabm_r3.py` (theo thứ tự trong `vong3/queue*_r3.sh`) → `vong3/ens_final_r3.py write 'stack_core3+alldiv'`. Mỗi vòng dùng lại đặc trưng của vòng trước (`fe_r3` → `fe_r2` → `fe_r1` → `fe`), và kết hợp vòng 3 dùng cả dự đoán vòng 2.

## Lưu ý: CHƯA chạy được ngay trên máy khác / Colab
- Đường dẫn cố định tới thư mục tạm của phiên cũ (`S='/tmp/claude-0/...'`) và tới `/home/user/GCI_github/com/`.
- Cần các file trung gian (đặc trưng đã tính, dự đoán OOF/test `.npy`, mô hình) không được đưa vào repo (~8 GB).
- Thư viện: pandas, numpy, scikit-learn, lightgbm, xgboost, catboost, optuna; TabM cần torch (CPU), `tabm`, `rtdl_num_embeddings`; Tác nhân 1 vòng 3 dùng `openfe`.

Bản nộp chạy được trên Colab sẽ được làm riêng dưới dạng notebook.
