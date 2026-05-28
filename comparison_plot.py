import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_comparison_8cells(full_data: pd.DataFrame,
                           M_pred: np.ndarray,
                           missing_mask: np.ndarray,
                           out_path: str,
                           n_show: int = 8) -> None:
    """
    Vẽ biểu đồ so sánh Thực tế vs Dự đoán cho n_show ô thiếu
    được chọn từ CÁC CỘT KHÁC NHAU (mỗi cột xuất hiện tối đa 1 lần).
    """
    rows_m, cols_m = np.where(missing_mask)

    # ── Chọn ô đại diện: mỗi cột chỉ lấy 1 ô đầu tiên ──────────────────
    seen_cols = set()
    selected  = []
    for r, c in zip(rows_m, cols_m):
        if c not in seen_cols:
            seen_cols.add(c)
            selected.append((r, c))
        if len(selected) == n_show:
            break

    # Nếu vẫn chưa đủ n_show thì lấy thêm (không ràng buộc cột)
    if len(selected) < n_show:
        for r, c in zip(rows_m, cols_m):
            if (r, c) not in selected:
                selected.append((r, c))
            if len(selected) == n_show:
                break

    records = []
    for r, c in selected:
        col_name  = full_data.columns[c]
        actual    = full_data.iloc[r, c]
        predicted = M_pred[r, c]
        records.append({
            "Nhãn":    f"[{r},{col_name[:7]}]",
            "Cột_full": col_name,
            "Thực_tế":  actual,
            "Dự_đoán":  predicted,
            "Sai_số":   abs(actual - predicted),
        })

    df_cmp = pd.DataFrame(records)

    # ── Layout ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1.2]})
    x  = np.arange(n_show)
    w  = 0.34

    C_ACT  = "#2980b9"
    C_PRED = "#e67e22"
    C_ERR  = "#c0392b"

    # ── Subplot 1: grouped bar ───────────────────────────────────────────
    ax1 = axes[0]
    b_act  = ax1.bar(x - w/2, df_cmp["Thực_tế"],
                     width=w, color=C_ACT, alpha=0.88,
                     label="Thực tế", edgecolor="white", zorder=3)
    b_pred = ax1.bar(x + w/2, df_cmp["Dự_đoán"],
                     width=w, color=C_PRED, alpha=0.88,
                     label="Dự đoán (MF)", edgecolor="white", zorder=3)

    for bar in b_act:
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() * 1.015,
                 f"{bar.get_height():.2f}",
                 ha="center", va="bottom", fontsize=7.5,
                 color=C_ACT, fontweight="bold")
    for bar in b_pred:
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() * 1.015,
                 f"{bar.get_height():.2f}",
                 ha="center", va="bottom", fontsize=7.5,
                 color=C_PRED, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(df_cmp["Nhãn"], fontsize=9)
    ax1.set_ylabel("Giá trị", fontsize=10)
    ax1.set_title(
        "So sánh Giá trị Thực tế vs Dự đoán\n"
        f"({n_show} ô thiếu – mỗi ô từ một đặc trưng riêng biệt – Matrix Factorization)",
        fontsize=12, fontweight="bold", pad=12
    )
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── Subplot 2: absolute error ─────────────────────────────────────────
    ax2 = axes[1]
    err_bars = ax2.bar(x, df_cmp["Sai_số"],
                       color=C_ERR, alpha=0.75, width=0.48,
                       edgecolor="white", zorder=3)

    max_err = df_cmp["Sai_số"].max()
    for bar, (_, row) in zip(err_bars, df_cmp.iterrows()):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max_err * 0.025,
                 f"{row['Sai_số']:.3f}",
                 ha="center", va="bottom", fontsize=8.5,
                 color=C_ERR, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(df_cmp["Nhãn"], fontsize=9)
    ax2.set_ylabel("|Sai số|", fontsize=10)
    ax2.set_title("Sai số tuyệt đối |Thực tế − Dự đoán| theo từng đặc trưng",
                  fontsize=10, pad=6)
    ax2.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax2.spines[["top", "right"]].set_visible(False)

    # ── Ghi chú đầy đủ tên cột ────────────────────────────────────────────
    note = "Nhãn: [dòng, tên_cột_gốc]\n" + \
           "   ".join([f"[{r},{c}]  → {row['Cột_full']}"
                       for (r, c), (_, row) in zip(selected, df_cmp.iterrows())])
    fig.text(0.01, -0.02, note,
             fontsize=7.5, color="#555555", verticalalignment="top")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Biểu đồ so sánh (8 ô, riêng biệt) lưu tại: {out_path}")
    return df_cmp


# ── Chạy thử ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.metrics import mean_squared_error

    np.random.seed(42)
    n = 100
    full_data = pd.DataFrame({
        'Độ_tuổi':                np.random.randint(18, 80, n),
        'Giới_tính':              np.random.choice([0, 1], n),
        'Thu_nhập':               np.round(np.random.uniform(3, 50, n), 2),
        'BMI':                    np.round(np.random.uniform(16, 40, n), 1),
        'HuyetAp_TamThu':         np.random.randint(90, 180, n),
        'DuongHuyet':             np.round(np.random.uniform(3.5, 12.0, n), 2),
        'Cholesterol':            np.round(np.random.uniform(3.0, 7.5, n), 2),
        'ThoiGian_VanDong_Daily': np.round(np.random.uniform(0, 120, n), 1),
        'ChiSo_Stress':           np.random.randint(1, 11, n),
    })

    missing_data = full_data.copy().astype(float)
    total_cells  = n * len(full_data.columns)
    n_missing    = int(np.round(total_cells * 0.15))
    rows_idx = np.random.randint(0, n, n_missing)
    cols_idx = np.random.randint(0, len(full_data.columns), n_missing)
    for r, c in zip(rows_idx, cols_idx):
        missing_data.iloc[r, c] = np.nan

    col_means = full_data.mean()
    col_stds  = full_data.std().replace(0, 1)
    M_full    = ((full_data    - col_means) / col_stds).values
    M_missing = ((missing_data - col_means) / col_stds).values
    observed_mask = ~np.isnan(M_missing)
    M_filled = np.where(observed_mask, M_missing, 0.0)

    def mf_sgd(M, mask, k=5, lr=0.005, reg=0.02, n_epochs=500, tol=1e-5):
        nrows, ncols = M.shape
        np.random.seed(0)
        U = np.random.normal(0, 0.1, (nrows, k))
        V = np.random.normal(0, 0.1, (ncols, k))
        prev = np.inf
        for ep in range(n_epochs):
            pred  = U @ V.T
            err   = (M - pred) * mask
            U    += lr * (err @ V   - reg * U)
            V    += lr * (err.T @ U - reg * V)
            loss  = np.sum(err**2) + reg*(np.sum(U**2)+np.sum(V**2))
            if abs(prev - loss) < tol: break
            prev = loss
        return U @ V.T

    M_pred_norm = mf_sgd(M_filled, observed_mask)
    M_pred = M_pred_norm * col_stds.values + col_means.values
    missing_mask = ~observed_mask

    df_result = plot_comparison_8cells(
        full_data, M_pred, missing_mask,
        out_path="comparison_8cells.png",
        n_show=8
    )
    print("\nBảng so sánh chi tiết:")
    print(df_result[["Nhãn","Cột_full","Thực_tế","Dự_đoán","Sai_số"]].to_string(index=False))