# -*- coding: utf-8 -*-
"""
06_make_maps.py —— 全部结果图（图例 colorbar + 比例尺 + 指北针）
================================================================
产出 results/fig01~fig09（PNG 300dpi，中文标签 Microsoft YaHei）：

  fig01 研究区概况（真值 D50 + 采样点）      fig06 Pauli RGB + VV/VH 三联
  fig02 GPR 空间反演图（D50 均值）           fig07 特征重要性 + GPR 核参数
  fig03 GPR 不确定性分布图（std）            fig08 GPR vs XGB 反演对比
  fig04 散点拟合图（GPR/XGB 并排）           fig09 潮汐淹没频率图（模拟真值）
  fig05 时序平均相干系数图

所有空间图统一配置：色带 colorbar（含单位）+ 左下比例尺(km) + 右上指北针(N)。
图中标注数字全部来自 results/metrics.json（单一事实来源）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from common import (DATA, RESULTS, REPORT, LON0, LON1, LAT0, LAT1, MOUTH_LON,
                    MOUTH_LAT, load_json, save_json)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 100

KM_PER_DEG_LON = 111.32 * np.cos(np.deg2rad(37.75))   # ≈88.0 km/°
EXTENT = [LON0, LON1, LAT0, LAT1]


# ---------------------------------------------------------------------------
# 制图小工具（度坐标下绘制）
# ---------------------------------------------------------------------------

def scalebar(ax, km=5.0, loc=(0.055, 0.075)):
    """左下比例尺：km -> 经度换算后画双须横线+文字。"""
    x0, y0 = LON0 + loc[0] * (LON1 - LON0), LAT0 + loc[1] * (LAT1 - LAT0)
    dx = km / KM_PER_DEG_LON
    dy = 0.012 * (LAT1 - LAT0)
    ax.plot([x0, x0 + dx], [y0, y0], "k-", lw=2.4, solid_capstyle="butt", zorder=30)
    ax.plot([x0, x0], [y0 - dy, y0 + dy], "k-", lw=2.4, zorder=30)
    ax.plot([x0 + dx, x0 + dx], [y0 - dy, y0 + dy], "k-", lw=2.4, zorder=30)
    ax.text(x0 + dx / 2, y0 + dy * 1.2, f"{km:g} km", ha="center", va="bottom",
            fontsize=9, zorder=30,
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.0))


def north(ax):
    """右上指北针：N + 箭头。"""
    ax.annotate("N", xy=(0.958, 0.955), xytext=(0.958, 0.875),
                xycoords="axes fraction", ha="center", va="top", fontsize=13,
                fontweight="bold", zorder=30,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6,
                                mutation_scale=20))


def base_map(ax, water, land):
    """底图：海域浅蓝 / 非滩面陆地浅灰。"""
    ax.imshow(np.where(water, 1.0, np.nan), extent=EXTENT, cmap=matplotlib.colors.ListedColormap(["#aacfefff"]),
              vmin=0, vmax=1, aspect="auto", interpolation="nearest", zorder=1)
    ax.imshow(np.where(land & ~water, 1.0, np.nan), extent=EXTENT,
              cmap=matplotlib.colors.ListedColormap(["#d9d4c9ff"]),
              vmin=0, vmax=1, aspect="auto", interpolation="nearest", zorder=1)


def finish(ax, title):
    """统一收尾：经纬刻度、标题。"""
    ax.set_xlim(LON0, LON1); ax.set_ylim(LAT0, LAT1)
    ax.set_xticks(np.arange(LON0, LON1 + 1e-9, 0.1))
    ax.set_yticks(np.arange(LAT0, LAT1 + 1e-9, 0.1))
    ax.tick_params(labelsize=8)
    ax.set_xlabel("经度 (°E)", fontsize=9)
    ax.set_ylabel("纬度 (°N)", fontsize=9)
    ax.set_title(title, fontsize=11)


def spatial_panel(ax, field, cmap, label, title, water, land, vmin=None,
                  vmax=None, pts=None):
    """一张标准空间图：底图 + 数据 + colorbar + 比例尺 + 指北针。"""
    base_map(ax, water, land)
    dat = np.where(field_mask_ok, field, np.nan)      # flat 掩膜外的数据不显示
    im = ax.imshow(dat, extent=EXTENT, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="auto", interpolation="nearest", zorder=2)
    cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02, aspect=24)
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)
    if pts is not None:
        ax.scatter(pts[:, 0], pts[:, 1], s=14, c="black", marker="o",
                   edgecolors="white", linewidths=0.5, zorder=10, label="采样点")
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    scalebar(ax); north(ax)
    finish(ax, title)
    return im


# ---------------------------------------------------------------------------
# 数据装载
# ---------------------------------------------------------------------------

truth = dict(np.load(DATA / "truth_grain.npz"))
feats = dict(np.load(DATA / "features.npz"))
preds = dict(np.load(DATA / "pred_maps.npz"))
samples = np.loadtxt(DATA / "dataset.csv", delimiter=",", skiprows=1, usecols=(0, 1))
metrics = load_json(RESULTS / "metrics.json")

water = truth["water"]; flat = truth["flat"]
land = ~(water | flat)                       # 高潮滩上带/盐沼/内陆
field_mask_ok = flat                          # 数据仅显示于潮间带滩面
md = metrics["d50"]

# ---------------------------------------------------------------------------
# fig01 研究区概况：真值 D50 + 采样点
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 6.6))
spatial_panel(ax, truth["d50"], "YlOrBr", "真值中值粒径 D50 (μm)",
              "研究区概况：模拟真值 D50 分布与采样点位置", water, land,
              vmin=0, vmax=120, pts=samples)
ax.plot(MOUTH_LON, MOUTH_LAT, marker="^", ms=9, color="#1f4e79", zorder=11)
ax.annotate("黄河入海口（模拟河口）", (MOUTH_LON, MOUTH_LAT), xytext=(6, 8),
            textcoords="offset points", fontsize=8, color="#1f4e79", zorder=11)
fig.tight_layout(); fig.savefig(RESULTS / "fig01_study_area.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------------------
# fig02 GPR 空间反演图
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 6.6))
spatial_panel(ax, preds["gpr_mean_d50"], "YlOrBr", "反演 D50 (μm)",
              f"GPR 空间反演图：D50（test R²={md['gpr']['r2_test']}, "
              f"RMSE={md['gpr']['rmse_test']} μm）", water, land,
              vmin=0, vmax=120, pts=samples)
fig.tight_layout(); fig.savefig(RESULTS / "fig02_gpr_inversion.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------------------
# fig03 GPR 不确定性分布图
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 6.6))
spatial_panel(ax, preds["gpr_std_d50"], "magma", "预测标准差 σ (μm)",
              "GPR 预测不确定性分布图（逐像元后验标准差）", water, land,
              vmin=0, vmax=float(np.nanmax(preds["gpr_std_d50"][flat])), pts=samples)
fig.tight_layout(); fig.savefig(RESULTS / "fig03_gpr_uncertainty.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------------------
# fig04 散点拟合图（GPR / XGB 并排，真实测试集预测）
# ---------------------------------------------------------------------------
tp = np.genfromtxt(DATA / "test_predictions.csv", delimiter=",", names=True,
                   encoding="utf-8")
fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6), sharex=True, sharey=True)
lim = [0, 120]
for axi in axes:
    axi.plot(lim, lim, "k--", lw=1.1, label="1:1 线")
    axi.set_xlim(lim); axi.set_ylim(lim)
    axi.set_xlabel("实测 D50 (μm)")
    axi.grid(alpha=0.25)
# 左：GPR（颜色 = 预测 std）
r2, rmse = md["gpr"]["r2_test"], md["gpr"]["rmse_test"]
sc = axes[0].scatter(tp["y_true"], tp["gpr_pred"], c=tp["gpr_std"], cmap="viridis",
                     s=26, edgecolors="white", linewidths=0.4)
cb = plt.colorbar(sc, ax=axes[0], shrink=0.85)
cb.set_label("GPR 预测 std (μm)", fontsize=9)
axes[0].set_title(f"GPR（高斯过程回归）\n$R^2$={r2}, RMSE={rmse} μm（test, n={len(tp)}）",
                  fontsize=10)
axes[0].set_ylabel("预测 D50 (μm)")
# 右：XGBoost
r2x, rmsex = md["xgb"]["r2_test"], md["xgb"]["rmse_test"]
axes[1].scatter(tp["y_true"], tp["xgb_pred"], s=26, c="#2ca02c",
                edgecolors="white", linewidths=0.4)
axes[1].set_title(f"XGBoost（对照）\n$R^2$={r2x}, RMSE={rmsex} μm（test, n={len(tp)}）",
                  fontsize=10)
axes[1].set_ylabel("预测 D50 (μm)")
for axi in axes:
    axi.legend(fontsize=8, loc="upper left")
fig.suptitle("散点拟合：留出测试集实测 vs 预测（数字取自 metrics.json）", fontsize=11)
fig.tight_layout(); fig.savefig(RESULTS / "fig04_scatter.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------------------
# fig05 时序平均相干系数图
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 6.6))
spatial_panel(ax, feats["coherence"], "RdYlGn", "时序平均相干系数 γ",
              "时序相干系数图（VV 通道，15 景 105 景对时间平均）", water, land,
              vmin=0, vmax=1, pts=samples)
fig.tight_layout(); fig.savefig(RESULTS / "fig05_coherence.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------------------
# fig06 Pauli RGB + VV + VH 三联
# ---------------------------------------------------------------------------
def stretch(x, lo=2, hi=98):
    a, b = np.percentile(x, [lo, hi])
    return np.clip((x - a) / max(b - a, 1e-9), 0, 1)

rgb = np.stack([stretch(feats["pauli1_db"]), stretch(feats["pauli2_db"]),
                stretch(feats["pauli3_db"])], axis=-1)
fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.6))
for axi, dat, ttl, cmap, lab in [
        (axes[0], np.where(flat[..., None], rgb, np.nan), "Pauli RGB（R=|k1|²面散射, "
         "G=|k2|²二次, B=|k3|²体散射）", None, None),
        (axes[1], feats["vv_db"], "σ0 VV (dB)", "gray", "dB"),
        (axes[2], feats["vh_db"], "σ0 VH (dB)", "gray", "dB")]:
    base_map(axi, water, land)
    if cmap is None:
        axi.imshow(dat, extent=EXTENT, aspect="auto", interpolation="nearest", zorder=2)
    else:
        v = np.nanpercentile(dat[flat], [2, 98])
        im = axi.imshow(np.where(flat, dat, np.nan), extent=EXTENT, cmap=cmap,
                        vmin=v[0], vmax=v[1], aspect="auto",
                        interpolation="nearest", zorder=2)
        cb = plt.colorbar(im, ax=axi, shrink=0.8, pad=0.02)
        cb.set_label(lab, fontsize=9)
    scalebar(axi); north(axi); finish(axi, ttl)
fig.tight_layout(); fig.savefig(RESULTS / "fig06_pauli_rgb.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------------------
# fig07 特征重要性 + GPR 核参数
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))
imp = md["xgb"]["importance_gain"]
names = list(imp.keys()); vals = [imp[k] for k in names]
order = np.argsort(vals)
axes[0].barh(np.array(names)[order], np.array(vals)[order], color="#2ca02c")
axes[0].set_xlabel("XGBoost importance (gain)")
axes[0].set_title("XGBoost 特征重要性（对照模型）", fontsize=10)
axes[0].grid(alpha=0.25, axis="x")
axes[1].axis("off")
ker = md["gpr"]["kernel_fitted"]
txt = ("GPR 拟合核（似然口径见 metrics.json）：\n\n"
       f"{ker}\n\n"
       "· ConstantKernel：信号方差 —— 刻画 D50 总体起伏幅度\n"
       "· RBF(length_scale)：特征空间相关长度 —— 越小越局部\n"
       "· WhiteKernel：观测噪声水平 —— 承接采样观测误差\n\n"
       f"样本 n={metrics['n_samples']}（train {metrics['n_train']} / "
       f"test {metrics['n_test']}），5 折 CV 种子 {metrics['seed']}")
axes[1].text(0.02, 0.95, txt, va="top", fontsize=10)   # 默认中文字体（monospace 无中文字形）
axes[1].set_title("GPR 核结构与样本划分", fontsize=10)
fig.tight_layout(); fig.savefig(RESULTS / "fig07_importance.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------------------
# fig08 GPR vs XGB 空间反演对比
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2))
for axi, dat, ttl in [
        (axes[0], preds["gpr_mean_d50"],
         f"GPR 反演（R²={md['gpr']['r2_test']}, RMSE={md['gpr']['rmse_test']} μm）"),
        (axes[1], preds["xgb_mean_d50"],
         f"XGBoost 反演（R²={md['xgb']['r2_test']}, RMSE={md['xgb']['rmse_test']} μm）")]:
    im = spatial_panel(axi, dat, "YlOrBr", "反演 D50 (μm)", ttl, water, land,
                       vmin=0, vmax=120, pts=samples)
fig.tight_layout(); fig.savefig(RESULTS / "fig08_model_compare.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------------------
# fig09 潮汐淹没频率（模拟真值）
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.6, 6.6))
spatial_panel(ax, truth["f_wet"], "Blues", "淹没频率 f_wet（0~1）",
              "潮汐淹没频率图（模拟真值，相干失相干的物理驱动）", water, land,
              vmin=0, vmax=0.5, pts=samples)
fig.tight_layout(); fig.savefig(RESULTS / "fig09_inundation.png", dpi=300); plt.close(fig)

# ---------------------------------------------------------------------------
# 图件索引
# ---------------------------------------------------------------------------
idx_lines = [
    "# 结果图件索引（results/）\n",
    "| 图 | 文件 | 说明 |",
    "|---|---|---|",
    "| 图1 | fig01_study_area.png | 研究区概况：模拟真值 D50 与 150 个采样点 |",
    "| 图2 | fig02_gpr_inversion.png | GPR 空间反演图（D50） |",
    "| 图3 | fig03_gpr_uncertainty.png | GPR 不确定性分布图（后验 std） |",
    "| 图4 | fig04_scatter.png | 散点拟合：GPR vs XGBoost（标注 metrics.json 的 test R²/RMSE） |",
    "| 图5 | fig05_coherence.png | 时序平均相干系数图（VV 通道） |",
    "| 图6 | fig06_pauli_rgb.png | Pauli RGB 合成 + σ0 VV + σ0 VH 三联 |",
    "| 图7 | fig07_importance.png | XGBoost 特征重要性 + GPR 核结构 |",
    "| 图8 | fig08_model_compare.png | GPR vs XGBoost 空间反演对比 |",
    "| 图9 | fig09_inundation.png | 潮汐淹没频率（模拟真值，失相干驱动场） |",
    "",
    "所有空间图均含 colorbar 图例（带单位）、左下比例尺（km）、右上指北针。",
    "图4 散点为 05 号脚本保存的留出测试集真实预测（data/test_predictions.csv）。",
]
(RESULTS / "figures_index.md").write_text("\n".join(idx_lines), encoding="utf-8")
print("== 06 完成：fig01~fig09（300dpi）+ figures_index.md ==")
