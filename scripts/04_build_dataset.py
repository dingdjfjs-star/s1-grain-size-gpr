# -*- coding: utf-8 -*-
"""
04_build_dataset.py —— 采样数据集构建与 SAR 特征匹配
====================================================
模拟"公开潮滩沉积物粒度采样数据集"：
  * 采样点：3 条垂直岸线剖面（约 17 点/条）+ 99 个滩面随机点，共 150 点；
  * 点值 = 真值场双线性采样 + 观测噪声（D50 σ=4 μm，占比 σ=2% 后归一），
    代表现场取样粒度分析的观测误差；
  * 报告中明确本表为演示用合成采样集；真实接入格式即本 CSV 表头
    （lon,lat,sand,silt,clay,d50,source，国家地球系统科学数据中心/论文附表
    均可整理成此格式）。

特征匹配：每点取 3×3 窗口均值（抑制点-像元配准误差与残余斑点），
输出 data/samples.csv（采样表）与 data/dataset.csv（特征表）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from common import (DATA, SAMPLES_CSV, DATASET_CSV, SEED, FEATURES, TARGETS,
                    LON0, LAT1, DLON, DLAT, lonlat_to_rc, bilinear_sample)

N_PER_TRANSECT = 17     # 每条剖面采样数
N_RANDOM = 99           # 随机点数
NOISE_D50 = 4.0         # D50 观测噪声 σ (μm)
NOISE_FRAC = 0.02       # 占比观测噪声 σ


def pick_points(truth: dict, rng) -> np.ndarray:
    """生成 150 个采样点（行,列 浮点坐标）：剖面 + 随机。"""
    flat = truth["flat"]
    pts = []
    # --- 3 条垂直岸线的剖面：沿纬向取三条，经向横穿滩面 ---
    for lat_t in (37.62, 37.75, 37.88):
        row_f = (LAT1 - lat_t) / DLAT
        band = flat[int(round(row_f)), :]
        cols = np.where(band)[0]
        if len(cols) < N_PER_TRANSECT:
            continue
        for c in np.linspace(cols.min() + 2, cols.max() - 2, N_PER_TRANSECT):
            pts.append((row_f + rng.normal(0, 2.0), c + rng.normal(0, 2.0)))
    # --- 滩面随机点 ---
    idx = np.argwhere(flat)
    sel = idx[rng.choice(len(idx), size=N_RANDOM, replace=False)]
    for r, c in sel:
        pts.append((r + 0.5 + rng.normal(0, 3.0), c + 0.5 + rng.normal(0, 3.0)))
    return np.array(pts)


def window_mean(field: np.ndarray, win: int = 3) -> np.ndarray:
    """3×3 窗口均值（边界用边缘填充）。"""
    return uniform_filter(field.astype(np.float64), win, mode="nearest")


def main() -> None:
    rng = np.random.default_rng(SEED + 2)
    truth = dict(np.load(DATA / "truth_grain.npz"))
    feats = dict(np.load(DATA / "features.npz"))

    pts = pick_points(truth, rng)
    rows, cols = pts[:, 0], pts[:, 1]
    lon = LON0 + (cols + 0.0) * DLON
    lat = LAT1 - rows * DLAT

    # --- 真值 + 观测噪声 ---
    d50 = bilinear_sample(truth["d50"], rows, cols) + rng.normal(0, NOISE_D50, len(pts))
    sand = bilinear_sample(truth["sand"], rows, cols) + rng.normal(0, NOISE_FRAC, len(pts))
    silt = bilinear_sample(truth["silt"], rows, cols) + rng.normal(0, NOISE_FRAC, len(pts))
    clay = bilinear_sample(truth["clay"], rows, cols) + rng.normal(0, NOISE_FRAC, len(pts))
    frac = np.stack([sand, silt, clay])
    frac = np.clip(frac, 0.01, 0.97)
    frac = frac / frac.sum(axis=0)
    sand, silt, clay = frac

    samples = pd.DataFrame({"lon": lon.round(6), "lat": lat.round(6),
                            "sand": sand.round(4), "silt": silt.round(4),
                            "clay": clay.round(4), "d50": d50.round(2),
                            "source": "synthetic-demo-v1(锚定黄河三角洲公开研究格局)"})
    samples.to_csv(SAMPLES_CSV, index=False, encoding="utf-8")

    # --- 特征匹配（3×3 窗均值后双线性采样）---
    data = {"lon": lon.round(6), "lat": lat.round(6)}
    for f in FEATURES:
        data[f] = bilinear_sample(window_mean(feats[f]), rows, cols).round(5)
    for t in TARGETS:
        data[t] = {"d50": d50, "sand": sand, "silt": silt, "clay": clay}[t].round(4)
    ds = pd.DataFrame(data)
    ds.to_csv(DATASET_CSV, index=False, encoding="utf-8")

    assert len(ds) == len(samples) and not ds[FEATURES].isna().any().any()
    print(f"== 04 完成：采样点 {len(ds)}（剖面 {3 * N_PER_TRANSECT} + 随机 {N_RANDOM}）")
    print("   特征-目标 Pearson 相关（d50）:")
    for f in FEATURES:
        r = np.corrcoef(ds[f], ds["d50"])[0, 1]
        print(f"     {f:<12s} r = {r:+.3f}")
    print(f"   d50 范围 {ds['d50'].min():.0f}~{ds['d50'].max():.0f} μm；"
          f"CSV -> samples.csv / dataset.csv")


if __name__ == "__main__":
    main()
