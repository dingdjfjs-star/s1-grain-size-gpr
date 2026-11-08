# -*- coding: utf-8 -*-
"""
common.py —— 项目级公共配置与工具函数
================================================
Sentinel-1 时序相干系数与极化特征反演潮滩沉积物粒度（GPR + 不确定性）
研究区：黄河三角洲现代入海口 — 莱州湾西岸潮滩

所有脚本共用本模块：
  * 全局常量（研究区 bbox、网格、种子、时间序列定义）
  * 路径约定（data/ results/ report/）
  * numpy 存取小工具
  * 制图小工具（比例尺 / 指北针 / 带 label 的 colorbar）—— cartopy 不可用，
    全部用 matplotlib 手绘实现，保证离线可复现

作者：夜跑值班自动化（2026-11-04）
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# 1. 全局常量
# ----------------------------------------------------------------------------

SEED = 20261104          # 全项目统一随机种子（保证逐位可复现）

# --- 研究区（WGS84 经纬度）：黄河三角洲现代入海口 + 莱州湾西岸潮滩 ---
LON0, LON1 = 118.95, 119.45     # 经度范围（°E）
LAT0, LAT1 = 37.55, 37.95       # 纬度范围（°N）
MOUTH_LON, MOUTH_LAT = 119.26, 37.78   # 入海口（合成海岸线上的河口位置，对应清8汊走向）

# --- SAR 网格：512x512，约 87 m/像元（相当于 GRD 8x8 多视后的时序分析网格）---
NX, NY = 512, 512               # NX=列(经度向)，NY=行(纬度向)
DLON = (LON1 - LON0) / NX       # 0.0009765625 °/像元 ≈ 85.7 m
DLAT = (LAT1 - LAT0) / NY       # 0.00078125   °/像元 ≈ 86.8 m

# --- Sentinel-1 时序设定（IW GRD，VV+VH 双极化，升轨，12 天重访）---
N_SCENES = 15                   # 时序景数（约覆盖 2024-04-01 起半年内 15 个重访周期）
REPEAT_DAYS = 12                # 重访周期（天）
T0_STR = "2024-04-01"           # 首景日期
ORBIT = "ASC"                   # 升轨
POL_MODE = "VV+VH"              # 双极化

# --- 潮汐设定（莱州湾不规则半日潮，取简化正态分布刻画潮位统计）---
TIDE_MEAN = 0.0                 # 平均海面（m）
TIDE_STD = 0.55                 # 潮位标准差（m），半潮差约 ±0.8 m
TIDE_RANGE = 1.6                # 潮差（m，仅用于文档口径）

# --- 潮滩高程分带阈值（相对平均海面，m）---
ELEV_WATER_MAX = 0.02           # 低于此视为水域（潮下带）
ELEV_MARSH_MIN = 1.70           # 高于此视为高潮滩上带/盐沼（很少淹没）

# --- 波段/雷达常量 ---
C_BAND_FREQ = 5.405e9           # Sentinel-1 C 波段中心频率（Hz）
LAMBDA = 299792458.0 / C_BAND_FREQ   # 波长 ≈ 5.547 cm

# ----------------------------------------------------------------------------
# 2. 路径约定
# ----------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]   # 项目根目录
DATA = ROOT / "data"
SIM_L1 = DATA / "sim_l1"                     # 模拟器输出的等效 L1 GRD/SLC 包
S1_PROC = DATA / "s1_proc"                   # 预处理校正后产品
RESULTS = ROOT / "results"
REPORT = ROOT / "report"
TEST_DIR = ROOT / "test"

METRICS_JSON = RESULTS / "metrics.json"
DATASET_CSV = DATA / "dataset.csv"
SAMPLES_CSV = DATA / "samples.csv"

for _p in (DATA, SIM_L1, S1_PROC, RESULTS, REPORT):
    _p.mkdir(parents=True, exist_ok=True)

FEATURES = ["vv_db", "vh_db", "vv_vh_ratio", "pauli1_db", "pauli2_db",
            "pauli3_db", "coherence", "incidence", "twi"]
TARGETS = ["d50", "sand", "silt", "clay"]


# ----------------------------------------------------------------------------
# 3. 小工具
# ----------------------------------------------------------------------------

def save_json(path: Path, obj) -> None:
    """写 UTF-8 JSON（ensure_ascii=False 保留中文，缩进 2 便于 diff）。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    """读 UTF-8 JSON。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_db(x: np.ndarray, floor: float = 1e-7) -> np.ndarray:
    """线性功率 -> dB（加下限防 log(0)）。"""
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), floor))


def bilinear_sample(field: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """对规则网格场做双线性采样（rows/cols 为浮点行列号，超界取边界值）。"""
    h, w = field.shape
    r = np.clip(rows, 0.0, h - 1.001)
    c = np.clip(cols, 0.0, w - 1.001)
    r0 = r.astype(int); c0 = c.astype(int)
    fr = r - r0; fc = c - c0
    return (field[r0, c0] * (1 - fr) * (1 - fc) + field[r0 + 1, c0] * fr * (1 - fc)
            + field[r0, c0 + 1] * (1 - fr) * fc + field[r0 + 1, c0 + 1] * fr * fc)


def lonlat_to_rc(lon: np.ndarray, lat: np.ndarray):
    """经纬度 -> 网格浮点 (row, col)。row 向南增加（与数组行序一致）。"""
    col = (np.asarray(lon) - LON0) / DLON
    row = (LAT1 - np.asarray(lat)) / DLAT
    return row, col


def grid_lonlat():
    """返回网格每个像元的经度、纬度二维数组（用于输出 GeoTIFF/点表）。"""
    lon1d = LON0 + (np.arange(NX) + 0.5) * DLON
    lat1d = LAT1 - (np.arange(NY) + 0.5) * DLAT
    return np.meshgrid(lon1d, lat1d)


def scene_dates():
    """返回 15 景成像日期字符串列表（等间隔 12 天）。"""
    import datetime as dt
    t0 = dt.date.fromisoformat(T0_STR)
    return [(t0 + dt.timedelta(days=i * REPEAT_DAYS)).isoformat() for i in range(N_SCENES)]


# ----------------------------------------------------------------------------
# 4. 制图小工具（cartopy 不可用，手绘比例尺/指北针）
# ----------------------------------------------------------------------------

def add_scalebar(ax, extent_km, length_km=None, loc="lower left", pad=0.06):
    """在 ax 上加公里比例尺：横线 + 两端竖须 + 文字。extent_km=(xmin, xmax) 以 km 计。"""
    span = extent_km[1] - extent_km[0]
    if length_km is None:
        # 自动取总跨度的 1/4，再吸附到 {1,2,5,10,20,25,50} km 漂亮值
        raw = span / 4.0
        for v in (1, 2, 5, 10, 20, 25, 50):
            if v >= raw:
                length_km = v
                break
        else:
            length_km = 50
    x0 = extent_km[0] + pad * span
    y0 = ax.get_ylim()[0] + pad * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x0, x0 + length_km], [y0, y0], color="black", lw=2.2,
            solid_capstyle="butt", zorder=20)
    for xd in (x0, x0 + length_km):
        ax.plot([xd, xd], [y0 - 0.012 * span / 4, y0 + 0.012 * span / 4],
                color="black", lw=2.2, zorder=20)
    ax.text(x0 + length_km / 2, y0 + 0.02 * span, f"{length_km} km",
            ha="center", va="bottom", fontsize=9, zorder=20,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))


def add_north_arrow(ax, x=0.955, y=0.905):
    """在 ax（轴坐标）右上角画指北针：N 字 + 箭头三角。"""
    ax.annotate("N", xy=(x, y + 0.075), xytext=(x, y), xycoords="axes fraction",
                ha="center", va="bottom", fontsize=13, fontweight="bold", zorder=20,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6,
                                mutation_scale=22))


def add_colorbar(fig, im, ax, label, shrink=0.86, pad=0.02):
    """给 im 挂竖直 colorbar 并写单位标签（作为图例）。"""
    cbar = fig.colorbar(im, ax=ax, shrink=shrink, pad=pad, aspect=26)
    cbar.set_label(label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    return cbar


# —— 运行自检（python scripts/common.py 可直接看配置摘要）——
if __name__ == "__main__":
    print(f"研究区 bbox: [{LON0},{LAT0}] ~ [{LON1},{LAT1}]")
    print(f"网格 {NX}x{NY}，像元约 {DLAT * 111.32:.1f} m；景数 {N_SCENES}，重访 {REPEAT_DAYS} d")
    print("景日期:", scene_dates()[:3], "...")
    print(f"波长 λ = {LAMBDA * 100:.3f} cm；根目录 {ROOT}")
