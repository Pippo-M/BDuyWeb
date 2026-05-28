import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_rmse(rmse_df: pd.DataFrame,
              overall_norm: float,
              out_path: str) -> None:
    """
    Vẽ biểu đồ cột dọc (vertical bar) thể hiện RMSE của từng cột
    sau khi thực hiện Matrix Factorization Imputation.

    Tham số
    -------
    rmse_df      : DataFrame có index là tên cột, cột 'RMSE' là giá trị sai số
    overall_norm : RMSE tổng thể đã chuẩn hoá về thang 0–1
    out_path     : Đường dẫn lưu ảnh đầu ra (ví dụ: 'rmse_chart.png')
    """
    n_cols = len(rmse_df)
    colors = plt.cm.tab10(np.linspace(0, 1, n_cols))

    fig, ax = plt.subplots(figsize=(11, 6))

    x_pos = np.arange(n_cols)
    bars  = ax.bar(x_pos, rmse_df["RMSE"],
                   color=colors, width=0.55,
                   edgecolor="white", linewidth=0.8,
                   zorder=3)

    # ── Nhãn giá trị phía trên mỗi cột ──────────────────────────────────
    for bar, v in zip(bars, rmse_df["RMSE"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(rmse_df["RMSE"]) * 0.015,
            f"{v:.4f}",
            ha="center", va="bottom",
            fontsize=8.5, color="#333333"
        )

    # ── Đường tham chiếu RMSE trung bình ─────────────────────────────────
    mean_rmse = rmse_df["RMSE"].mean()
    ax.axhline(mean_rmse, color="#e74c3c", linewidth=1.4,
               linestyle="--", zorder=4,
               label=f"RMSE trung bình: {mean_rmse:.4f}")

    # ── Tuỳ chỉnh trục & nhãn ────────────────────────────────────────────
    ax.set_xticks(x_pos)
    ax.set_xticklabels(rmse_df.index, rotation=25,
                       ha="right", fontsize=9)
    ax.set_ylabel("RMSE (thang giá trị gốc)", fontsize=10)
    ax.set_xlabel("Đặc trưng (Feature)", fontsize=10)
    ax.set_title(
        "RMSE sau Matrix Factorization Imputation\n"
        f"(RMSE tổng thể chuẩn hóa 0–1: {overall_norm:.4f})",
        fontsize=12, fontweight="bold", pad=14
    )

    # ── Lưới nền nhẹ ─────────────────────────────────────────────────────
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", which="major", linestyle="--",
            alpha=0.4, zorder=0)
    ax.grid(axis="y", which="minor", linestyle=":",
            alpha=0.2, zorder=0)

    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Biểu đồ RMSE (vertical bar) lưu tại: {out_path}")
