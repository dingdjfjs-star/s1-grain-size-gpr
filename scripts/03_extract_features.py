# -*- coding: utf-8 -*-
"""
03_extract_features.py —— 特征提取：极化 / 时序相干 / 入射角 / 地形 TWI
========================================================================
从 02 号校正产品与等效 SLC 包提取 9 维特征栅格：

  极化特征（时序平均，dB）：
    vv_db, vh_db          校正后 σ⁰VV/VH
    vv_vh_ratio           VV−VH (dB)
    pauli1/2/3_db         Pauli 三分量 |k1|²,|k2|²,|k3|²（等效 SLC 复数据）
  时序相干系数：
    coherence             VV/VH 通道景对相干的时间平均（5×5 窗估计）
    coh_vv, coh_vh        分通道相干图
  几何/地形：
    incidence             时序平均入射角（°）
    twi                   地形湿度指数 ln(a/tanβ)，D8 汇流 + 真实 SRTM DEM

说明：
  * 相干估计用复数据（m_vv/m_vh）；真实 GRD 为强度数据无法直接估时序相干，
    真实轨应改用 SLC 对（见 README/报告误差来源）—— 模拟轨按 SLC 流程执行。
  * Pauli 分量直接取模拟复 Pauli 基分量 |k|²（裸滩面以 k1 面散射主导）。
输出：data/features.npz（全部特征栅格）+ 控制台类别 median 自检表
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from common import (DATA, S1_PROC, N_SCENES, NX, NY, save_json)

COH_WIN = 5      # 相干估计窗口（多视复数据上的 boxcar）
PAIR_IDX = [(i, j) for i in range(N_SCENES) for j in range(i + 1, N_SCENES)]


# ---------------------------------------------------------------------------
# 时序相干
# ---------------------------------------------------------------------------

def coherence_map(m_i: np.ndarray, m_j: np.ndarray) -> np.ndarray:
    """景对 (i,j) 的相干系数图：5×5 窗复相干估计（可分离 boxcar 实现）。"""
    prod = m_i * np.conj(m_j)
    num = (uniform_filter(prod.real, COH_WIN) ** 2
           + uniform_filter(prod.imag, COH_WIN) ** 2)
    den = uniform_filter(np.abs(m_i) ** 2, COH_WIN) * \
        uniform_filter(np.abs(m_j) ** 2, COH_WIN)
    return (np.sqrt(num) / np.sqrt(np.maximum(den, 1e-12))).astype(np.float32)


def temporal_coherence(channel: str):
    """全部景对相干的时间平均图与景间标准差图。"""
    scenes = [np.load(S1_PROC.parent / "sim_l1" / f"slc_scene_{i:02d}.npz")[channel]
              for i in range(1, N_SCENES + 1)]
    acc = np.zeros((NY, NX), np.float64)
    acc2 = np.zeros_like(acc)
    for (i, j) in PAIR_IDX:
        g = coherence_map(scenes[i], scenes[j])
        acc += g
        acc2 += g.astype(np.float64) ** 2
    n = len(PAIR_IDX)
    mean = acc / n
    std = np.sqrt(np.maximum(acc2 / n - mean ** 2, 0.0))
    return mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------------------
# 地形湿度指数 TWI（D8 单流向）
# ---------------------------------------------------------------------------

def d8_flow_accum(dem: np.ndarray) -> np.ndarray:
    """D8 单流向汇流累计像元数（按高程降序依次把累计量汇入最陡下坡邻元）。"""
    h, w = dem.shape
    # 8 邻域偏移（行,列）与距离
    offs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
            (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]
    acc = np.ones(h * w, np.float64)
    receiver = np.full(h * w, -1, np.int64)
    idx = np.arange(h * w)
    rr, cc = np.unravel_index(idx, (h, w))
    best_slope = np.zeros(h * w)
    for dr, dc, dist in offs:
        r2, c2 = rr + dr, cc + dc
        ok = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
        slope = np.full(h * w, -np.inf)
        slope[ok] = (dem[rr[ok], cc[ok]] - dem[r2[ok], c2[ok]]) / dist
        upd = ok & (slope > best_slope)          # 最陡下坡方向
        best_slope[upd] = slope[upd]
        receiver[upd] = r2[upd] * w + c2[upd]    # 接收像元一维索引
    # 平坦/海域（无下坡）保留为汇：acc 仅含自身，TWI 由坡度下限控制
    order = np.argsort(-dem.ravel())             # 高到低处理
    for k in order:
        rc = receiver[k]
        if rc >= 0:
            acc[rc] += acc[k]
    return acc.reshape(h, w)


def compute_twi(dem: np.ndarray):
    """TWI = ln(a / tanβ)；a=(acc+1)·像元面积，β 为坡度角（tanβ 下限 1e-3）。"""
    cs = 86.0                                    # 像元尺寸 m
    dzdy, dzdx = np.gradient(dem, cs, cs)
    tanb = np.maximum(np.hypot(dzdx, dzdy), 1e-3)
    acc = d8_flow_accum(dem)
    a = (acc + 1.0) * cs                         # 单位汇水宽度近似
    return np.log(a / tanb).astype(np.float32)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    print("== 03 特征提取 ==")
    # --- 极化特征：时序平均 ---
    vv_acc = np.zeros((NY, NX)); vh_acc = np.zeros((NY, NX))
    p1_acc = np.zeros((NY, NX)); p2_acc = np.zeros((NY, NX)); p3_acc = np.zeros((NY, NX))
    theta_acc = np.zeros((NY, NX))
    for i in range(1, N_SCENES + 1):
        proc = np.load(S1_PROC / f"proc_scene_{i:02d}.npz")
        slc = np.load(DATA / "sim_l1" / f"slc_scene_{i:02d}.npz")
        vv_acc += proc["sig0_vv"]
        vh_acc += proc["sig0_vh"]
        p1_acc += np.abs(slc["k1"]) ** 2
        p2_acc += np.abs(slc["k2"]) ** 2
        p3_acc += np.abs(slc["k3"]) ** 2
        theta_acc += proc["theta"]
    n = N_SCENES
    vv_db = 10 * np.log10(np.maximum(vv_acc / n, 1e-9))
    vh_db = 10 * np.log10(np.maximum(vh_acc / n, 1e-9))
    floor = 1e-8
    pauli1 = 10 * np.log10(np.maximum(p1_acc / n, floor))
    pauli2 = 10 * np.log10(np.maximum(p2_acc / n, floor))
    pauli3 = 10 * np.log10(np.maximum(p3_acc / n, floor))

    # --- 时序相干 ---
    # 特征口径取 VV 主极化通道：平滑泥滩 VH 近 NESZ，相干被热噪声严重压低，
    # 文献中潮滩时序相干亦多以主极化通道为准；VH 相干保留作参考图。
    coh_vv, std_vv = temporal_coherence("m_vv")
    coh_vh, _ = temporal_coherence("m_vh")
    coherence = coh_vv

    # --- 入射角 / TWI ---
    incidence = (theta_acc / n).astype(np.float32)
    dem = np.load(S1_PROC / "proc_scene_01.npz")["dem"].astype(np.float64)
    twi = compute_twi(dem)

    np.savez_compressed(DATA / "features.npz",
                        vv_db=vv_db, vh_db=vh_db, vv_vh_ratio=vv_db - vh_db,
                        pauli1_db=pauli1, pauli2_db=pauli2, pauli3_db=pauli3,
                        coherence=coherence, coh_vv=coh_vv, coh_vh=coh_vh,
                        incidence=incidence, twi=twi)

    # --- 类别 median 自检（借用真值场仅作诊断打印，不进入特征）---
    tr = np.load(DATA / "truth_grain.npz")
    flat = tr["flat"]; fw = tr["f_wet"]
    hi, lo = flat & (fw < 0.10), flat & (fw > 0.35)
    chk = dict(coh_high_flat=round(float(np.median(coherence[hi])), 3),
               coh_low_flat=round(float(np.median(coherence[lo])), 3),
               vv_high=round(float(np.median(vv_db[flat & (tr["d50"] > 60)])), 2),
               vv_low=round(float(np.median(vv_db[flat & (tr["d50"] < 25)])), 2),
               twi_range=[round(float(twi.min()), 2), round(float(twi.max()), 2)],
               incidence_range=[round(float(incidence.min()), 2),
                                round(float(incidence.max()), 2)])
    save_json(DATA / "features_check.json", chk)
    print(f"  相干：高滩 {chk['coh_high_flat']} vs 低滩 {chk['coh_low_flat']}"
          f"（期望低滩显著更低）")
    print(f"  σ⁰VV：粗滩 {chk['vv_high']} dB vs 细滩 {chk['vv_low']} dB")
    print(f"  TWI {chk['twi_range']}；入射角 {chk['incidence_range']}°")
    print("== 03 完成：data/features.npz ==")


if __name__ == "__main__":
    main()
