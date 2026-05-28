import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# ============================================================
# 1. SINH BẢNG DỮ LIỆU GỐC (full_data)
# ============================================================
np.random.seed(42)
n = 100

full_data = pd.DataFrame({
    'Độ_tuổi':               np.random.randint(18, 80, n),
    'Giới_tính':             np.random.choice([0, 1], n),          # 0=Nữ, 1=Nam
    'Thu_nhập':              np.round(np.random.uniform(3, 50, n), 2),   # triệu VNĐ
    'BMI':                   np.round(np.random.uniform(16, 40, n), 1),
    'HuyetAp_TamThu':        np.random.randint(90, 180, n),
    'DuongHuyet':            np.round(np.random.uniform(3.5, 12.0, n), 2),
    'Cholesterol':           np.round(np.random.uniform(3.0, 7.5, n), 2),
    'ThoiGian_VanDong_Daily':np.round(np.random.uniform(0, 120, n), 1),  # phút
    'ChiSo_Stress':          np.random.randint(1, 11, n),                 # 1-10
})

print("=" * 60)
print("BẢNG DỮ LIỆU GỐC (full_data) - 5 dòng đầu")
print("=" * 60)
print(full_data.head())
print(f"\nShape: {full_data.shape}")
print(f"Missing values: {full_data.isnull().sum().sum()}")

# ============================================================
# 2. SINH BẢNG MISSING DATA (~15% tổng số ô)
# ============================================================
missing_data = full_data.copy().astype(float)

total_cells = n * len(full_data.columns)
n_missing   = int(np.round(total_cells * 0.15))

rows_idx = np.random.randint(0, n,                        n_missing)
cols_idx = np.random.randint(0, len(full_data.columns),   n_missing)
for r, c in zip(rows_idx, cols_idx):
    missing_data.iloc[r, c] = np.nan

actual_pct = missing_data.isnull().sum().sum() / total_cells * 100
print("\n" + "=" * 60)
print("BẢNG DỮ LIỆU KHUYẾT (missing_data)")
print("=" * 60)
print(f"Số ô bị khuyết: {missing_data.isnull().sum().sum()} / {total_cells} "
      f"({actual_pct:.1f}%)")
print("\nSố giá trị thiếu theo cột:")
print(missing_data.isnull().sum())

# ============================================================
# 3. PHÂN RÃ MA TRẬN (Matrix Factorization) – dự đoán missing
# ============================================================
# Chuẩn hoá: Z-score trên full_data rồi áp dụng mask
col_means = full_data.mean()
col_stds  = full_data.std().replace(0, 1)

M_full    = ((full_data  - col_means) / col_stds).values          # (100, 9)
M_missing = ((missing_data - col_means) / col_stds).values        # (100, 9)

observed_mask = ~np.isnan(M_missing)
M_filled = np.where(observed_mask, M_missing, 0.0)   # khởi tạo NaN = 0

# Thuật toán SGD Matrix Factorization
def matrix_factorization_sgd(M, mask, k=5, lr=0.005, reg=0.02,
                              n_epochs=500, tol=1e-5, verbose=True):
    """
    Phân rã M ≈ U @ V^T
    M    : ma trận đầy đủ (ban đầu NaN đã được điền 0)
    mask : True ở ô quan sát được
    k    : số nhân tố ẩn (latent factors)
    """
    nrows, ncols = M.shape
    np.random.seed(0)
    U = np.random.normal(0, 0.1, (nrows, k))
    V = np.random.normal(0, 0.1, (ncols, k))

    prev_loss = np.inf
    for epoch in range(n_epochs):
        # Tính sai số chỉ trên ô quan sát
        pred  = U @ V.T
        error = (M - pred) * mask

        # Gradient descent
        U += lr * (error @ V      - reg * U)
        V += lr * (error.T @ U    - reg * V)

        loss = np.sum(error ** 2) + reg * (np.sum(U**2) + np.sum(V**2))
        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:4d} | Loss = {loss:.6f}")

        if abs(prev_loss - loss) < tol:
            if verbose:
                print(f"  Hội tụ tại epoch {epoch+1}")
            break
        prev_loss = loss

    return U @ V.T

print("\n" + "=" * 60)
print("PHÂN RÃ MA TRẬN (SGD, k=5 latent factors)")
print("=" * 60)
M_pred_norm = matrix_factorization_sgd(M_filled, observed_mask,
                                        k=5, lr=0.005, reg=0.02,
                                        n_epochs=500, verbose=True)

# Giải chuẩn hoá
M_pred = M_pred_norm * col_stds.values + col_means.values
predicted_df = pd.DataFrame(M_pred, columns=full_data.columns)

# ============================================================
# 4. TÍNH RMSE – chỉ trên các ô bị khuyết
# ============================================================
missing_mask = ~observed_mask    # True ở ô thiếu

print("\n" + "=" * 60)
print("KẾT QUẢ RMSE (chỉ tính trên ô bị khuyết)")
print("=" * 60)

rmse_per_col = {}
for j, col in enumerate(full_data.columns):
    mask_col = missing_mask[:, j]
    if mask_col.sum() == 0:
        rmse_per_col[col] = np.nan
        continue
    y_true = full_data[col].values[mask_col]
    y_pred = M_pred[:, j][mask_col]
    rmse_per_col[col] = np.sqrt(mean_squared_error(y_true, y_pred))

rmse_df = pd.DataFrame.from_dict(rmse_per_col, orient='index',
                                  columns=['RMSE'])
print(rmse_df.to_string())

# RMSE tổng hợp (tất cả ô thiếu)
y_true_all = full_data.values[missing_mask]
y_pred_all = M_pred[missing_mask]
overall_rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
print(f"\n{'─'*40}")
print(f"RMSE tổng hợp (overall): {overall_rmse:.4f}")

# ============================================================
# 5. VÍ DỤ SO SÁNH: giá trị thực vs dự đoán (10 ô đầu tiên)
# ============================================================
print("\n" + "=" * 60)
print("SO SÁNH GIÁ TRỊ THỰC vs DỰ ĐOÁN (10 ô thiếu đầu tiên)")
print("=" * 60)
rows_m, cols_m = np.where(missing_mask)
compare = []
for idx in range(min(10, len(rows_m))):
    r, c = rows_m[idx], cols_m[idx]
    col_name = full_data.columns[c]
    compare.append({
        'Dòng': r,
        'Cột': col_name,
        'Thực_tế': round(full_data.iloc[r, c], 3),
        'Dự_đoán': round(M_pred[r, c], 3),
        'Sai_số': round(abs(full_data.iloc[r, c] - M_pred[r, c]), 3)
    })
compare_df = pd.DataFrame(compare)
print(compare_df.to_string(index=False))

print("\n✅ Hoàn thành!")
