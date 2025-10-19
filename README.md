

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f as f_dist, shapiro, levene, t as t_dist
try:
    from scipy.stats import studentized_range
    HAVE_SR = True
except Exception:
    HAVE_SR = False


# 題目資料

raw = {
    "A": {1: [9, 10], 2: [10, 12], 3: [16, 14]},
    "B": {1: [13, 15], 2: [10, 11], 3: [12, 15]},
    "C": {1: [8,  9],  2: [11, 9],  3: [12, 10]},
    "D": {1: [10, 13], 2: [14, 12], 3: [20, 17]},
}

# tidy（長格式）資料：每列一筆觀測
rows = []
for shoe, by_sock in raw.items():
    for sock, reps in by_sock.items():
        for r, y in enumerate(reps, 1):
            rows.append({"shoe": shoe, "sock": str(sock), "rep": r, "time": y})
df = pd.DataFrame(rows)


# 平衡檢查 & 基本數量

cell_n = df.groupby(["shoe", "sock"]).size().unstack()
assert (cell_n.values == cell_n.values[0, 0]).all(), "不是平衡設計；此手算法僅適用平衡資料"
a = df["shoe"].nunique()          # 鞋水準數
b = df["sock"].nunique()          # 襪水準數
n = int(cell_n.values[0, 0])      # 每格重複數
N = a * b * n                     # 總樣本數

# 輸出路徑
try:
    OUTDIR = (Path(__file__).parent / "outputs").resolve()
except NameError:
    OUTDIR = Path.cwd() / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)


# 一因子 ANOVA（鞋）

grand = df["time"].mean()
A_mean = df.groupby("shoe")["time"].mean()
groups_A = [g["time"].to_numpy() for _, g in df.groupby("shoe")]

n_i = np.array([len(g) for g in groups_A], dtype=float)      # 每鞋的樣本數（= b*n）
SS_A = float((n_i * (A_mean.values - grand) ** 2).sum())
SS_T = float(((df["time"] - grand) ** 2).sum())
SS_E_oneway = float(sum(((g - g.mean()) ** 2).sum() for g in groups_A))

df_A  = a - 1
df_E1 = N - a
MS_A  = SS_A / df_A
MS_E1 = SS_E_oneway / df_E1
F_A   = MS_A / MS_E1
p_A   = float(f_dist.sf(F_A, df_A, df_E1))

aov1 = pd.DataFrame({
    "SS": [SS_A, SS_E_oneway],
    "df": [df_A, df_E1],
    "MS": [MS_A, MS_E1],
    "F":  [F_A,  np.nan],
    "p":  [p_A,  np.nan],
}, index=["C(shoe)", "Residual"])


# 兩因子 ANOVA（鞋 × 襪）手算（平衡）

B_mean   = df.groupby("sock")["time"].mean()
cell_mean= df.groupby(["shoe", "sock"])["time"].mean()

SSA = b * n * float(((A_mean - grand) ** 2).sum())
SSB = a * n * float(((B_mean - grand) ** 2).sum())

cm = cell_mean.reset_index().rename(columns={"time": "cell"})
cm["Amean"] = cm["shoe"].map(A_mean)
cm["Bmean"] = cm["sock"].map(B_mean)
SSAB = n * float(((cm["cell"] - cm["Amean"] - cm["Bmean"] + grand) ** 2).sum())

SSE  = SS_T - SSA - SSB - SSAB

dfA  = a - 1
dfB  = b - 1
dfAB = (a - 1) * (b - 1)
dfE  = a * b * (n - 1)

MSA, MSB, MSAB, MSE = SSA/dfA, SSB/dfB, SSAB/dfAB, SSE/dfE
FA, FB, FAB = MSA/MSE, MSB/MSE, MSAB/MSE
pA2  = float(f_dist.sf(FA,  dfA, dfE))
pB2  = float(f_dist.sf(FB,  dfB, dfE))
pAB2 = float(f_dist.sf(FAB, dfAB, dfE))

aov2 = pd.DataFrame({
    "SS": [SSA, SSB, SSAB, SSE],
    "df": [dfA, dfB, dfAB, dfE],
    "MS": [MSA, MSB, MSAB, MSE],
    "F":  [FA,  FB,  FAB,  np.nan],
    "p":  [pA2, pB2, pAB2, np.nan],
}, index=["C(shoe)", "C(sock)", "C(shoe):C(sock)", "Residual"])



#主效應 & 交互作用

mean_by_shoe = A_mean.sort_index()
mean_by_sock = B_mean.sort_index()
mean_matrix  = cell_mean.unstack()  # index=sock, columns=shoe

# 主效應：鞋
plt.figure()
mean_by_shoe.plot(marker="o")
plt.title("Main Effect: Shoe")
plt.xlabel("Shoe"); plt.ylabel("Mean time over 100 (min)")
plt.tight_layout(); plt.savefig(OUTDIR / "main_effect_shoe.png", dpi=150); plt.close()

# 主效應：襪
plt.figure()
mean_by_sock.plot(marker="o")
plt.title("Main Effect: Sock")
plt.xlabel("Sock"); plt.ylabel("Mean time over 100 (min)")
plt.tight_layout(); plt.savefig(OUTDIR / "main_effect_sock.png", dpi=150); plt.close()

# 交互作用圖
plt.figure()
for sock in mean_matrix.columns:                
    y = mean_matrix[sock].values
    x = mean_matrix.index.astype(str)           
    plt.plot(x, y, marker='o', label=f'Sock {sock}')
plt.title('Interaction: y vs Shoe (lines = Sock)')
plt.xlabel('Shoe'); plt.ylabel('Mean time over 100 (min)')
plt.legend(title='Sock'); plt.tight_layout()
plt.savefig(OUTDIR / "interaction_plot.png", dpi=150); plt.close()

# 最佳組合 & 穩健性

best_combo = cell_mean.idxmin()
best_value = float(cell_mean.min())
variance_across_socks = cell_mean.unstack().var(axis=1).sort_values()


# 摘要報告

with open(OUTDIR / "summary_report_3-2_noregression.txt", "w", encoding="utf-8") as f:
    f.write(f"Design: a={a} shoes × b={b} socks × n={n} reps; N={N}\n\n")
    f.write("[One-way ANOVA: Shoe]\n")
    f.write(aov1.round(4).to_string() + "\n\n")
    f.write("[Two-way ANOVA: Shoe × Sock]\n")
    f.write(aov2.round(4).to_string() + "\n\n")
    f.write("[Means by shoe]\n")
    f.write(A_mean.round(3).to_string() + "\n\n")
    f.write("[Means by sock]\n")
    f.write(B_mean.round(3).to_string() + "\n\n")
    f.write("[Mean matrix (sock × shoe)]\n")
    f.write(mean_matrix.round(3).to_string() + "\n\n")
    f.write(f"[Best combo] Shoe {best_combo[0]} × Sock {best_combo[1]} = {best_value:.2f}\n\n")
    f.write("[Per-shoe variance across socks (lower=more robust)]\n")
    f.write(variance_across_socks.round(4).to_string() + "\n\n")
    
print("Saved to:", OUTDIR)
print(" - main_effect_shoe.png")
print(" - main_effect_sock.png")
print(" - interaction_plot.png")
print(" - summary_report_3-2_noregression.txt")

