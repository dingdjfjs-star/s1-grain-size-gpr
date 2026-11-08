# -*- coding: utf-8 -*-
"""
02_preprocess_grd.py —— Sentinel-1 GRD 预处理链（对模拟 L1 包真实执行）
========================================================================
对 data/sim_l1/ 的 15 景等效 L1 GRD 包按业务化顺序执行：

  ① 轨道校正（占位实现）…… 模拟数据无轨道误差，本步读取/校验轨道与几何
     元数据并重算入射角场；真实 SAFE 接入时替换为 POD 精轨文件插值（见 README）。
  ② 热噪声去除 ……………… I_sig = (DN/gain)² − NESZ_lin(距向)/calib(距向)，
     与 S1 IPF 的 L1 → L1.5 去噪口径一致（噪声在功率域扣除）。
  ③ 辐射定标 …………………… σ⁰ = I_sig · calib(距向)，输出线性 σ⁰VV/VH。
  ④ Lee Sigma 斑点滤波（自实现，不调库）…… 7×7 窗口，按斑点统计模型
     I = L·R·Gamma(L,1/L) 以 sigma 范围选窗内同类像元后取均值（Lee 1986 /
     S1 IPF 思想）；ENL 由暗端均质区稳健估计；与 5×5 增强 Lee 对比择优。
  ⑤ 地形校正（简化版）…… 用真实 SRTM DEM（tools/fetch_dem.py 抓取，约
     700 m 抽样上采样）计算局部入射角做 cosθ_l/cosθ 面积改正；坡度 <2° 时
     改正量微小，完整 Range-Doppler 几何纠正的差异在报告中说明。

输出：data/s1_proc/proc_scene_NN.npz（校正后 σ⁰VV/VH、θ、DEM、局部入射角）
      data/s1_proc/processing_log.json（每景 ENL/噪声扣除量/改正量等质检数字）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from common import (DATA, SIM_L1, S1_PROC, N_SCENES, NX, NY, SEED, save_json,
                    to_db)

WIN = 7            # Lee Sigma 滤波窗口
Z_SIGMA = 1.645    # sigma 范围双侧 90%（对应 SNAP Lee Sigma 的常用口径）
MIN_PIX = 5        # 窗内入选像元数下限，不足则退化为窗口均值
LOOK_AZ_DEG = 95.0  # 升轨右视近似方位（由 01 元数据 look_az_deg 传入口径）


# ---------------------------------------------------------------------------
# 预处理各步
# ---------------------------------------------------------------------------

def step1_orbit(meta: dict) -> dict:
    """① 轨道校正（占位）：校验元数据几何自洽，返回几何记录。

    真实 SAFE 接入：读 RESUMED 精密轨道（POD）状态矢量，按成像时刻插值
    卫星位置/速度，重算零多普勒距离多普勒定位 —— 模拟数据几何由构造保证，
    此处只做一致性校验并留痕。
    """
    ok = (meta.get("orbit") == "ASC" and 0.0 < meta.get("look_az_deg", 0) < 360.0)
    return {"orbit": meta.get("orbit"), "look_az_deg": meta.get("look_az_deg"),
            "orbit_check_ok": bool(ok)}


def step2_thermal_noise(dn: np.ndarray, gain: float, nesz_db: np.ndarray,
                        calib: np.ndarray):
    """② 热噪声去除：功率域扣除 NESZ（含跨轨条带），返回信号强度。"""
    i_meas = (dn.astype(np.float64) / gain) ** 2
    i_noise = (10.0 ** (nesz_db.astype(np.float64) / 10.0)) / calib  # (NX,)
    i_sig = np.maximum(i_meas - i_noise[None, :], 1e-9)
    return i_sig, i_noise


def step3_calibration(i_sig: np.ndarray, calib: np.ndarray):
    """③ 辐射定标：σ⁰ = I_sig · calib（跨轨 LUT），返回线性 σ⁰。"""
    return i_sig * calib[None, :]


def estimate_enl(sig0_lin: np.ndarray) -> float:
    """稳健 ENL 估计：暗端 30% 像元（水面/平滑泥滩，最均质）的 (mean/std)²。"""
    win = sliding_window_view(sig0_lin, (WIN, WIN))
    m = win.mean(axis=(-2, -1))
    s = win.std(axis=(-2, -1)) + 1e-9
    enl_map = (m / s) ** 2
    thr = np.quantile(sig0_lin, 0.30)
    dark = m < thr
    return float(np.clip(np.median(enl_map[dark]) if dark.any() else 2.0, 1.0, 8.0))


def lee_sigma_filter(img: np.ndarray, enl: float) -> np.ndarray:
    """④ 自实现 Lee Sigma 滤波（sigma 范围选点 + 均值），返回滤波后图像。

    斑点模型：I = L·R·Gamma(L,1/L)，R 为真实反射率。
    对每像元窗口：均值 m、标准差 s；sigma 范围取 [m−z·s, m+z·s]
    （z=1.645，双侧 90%），仅对窗内落在范围中的像元取均值；
    入选不足 MIN_PIX 时退化为窗口均值（等效增强 Lee）。
    全程 sliding_window_view 向量化，无逐像元循环。
    """
    win = sliding_window_view(img, (WIN, WIN)).astype(np.float64)   # (H',W',7,7)
    m = win.mean(axis=(-2, -1))
    s = win.std(axis=(-2, -1))
    lo, hi = m - Z_SIGMA * s, m + Z_SIGMA * s
    in_range = (win >= lo[..., None, None]) & (win <= hi[..., None, None])
    cnt = in_range.sum(axis=(-2, -1))
    sel_sum = (win * in_range).sum(axis=(-2, -1))
    mean_sel = np.where(cnt >= MIN_PIX, sel_sum / np.maximum(cnt, 1), m)
    # 边界回填（sliding 窗口尺寸损失）
    pad = WIN // 2
    out = np.pad(mean_sel, pad, mode="edge")
    return out[:img.shape[0], :img.shape[1]].astype(np.float32)


def lee_enhanced_5x5(img: np.ndarray) -> np.ndarray:
    """④' 对照：5×5 增强 Lee（Kuan 权重型），用于与 Lee Sigma 对比择优。

    w = 1 − Cu²/C²（Cu=1/sqrt(ENL) 为纯斑点变异系数），
    输出 = m + w·(center − m)：均质区趋近窗口均值，异质区保留原像元。
    """
    win = sliding_window_view(img, (5, 5)).astype(np.float64)
    m = win.mean(axis=(-2, -1))
    var = win.var(axis=(-2, -1))
    enl = max(estimate_enl(img), 1.0)
    cu2 = 1.0 / enl
    c2 = var / np.maximum(m ** 2, 1e-9)
    w = np.clip(1.0 - cu2 / np.maximum(c2, 1e-9), 0.0, 1.0)
    center = win[..., 2, 2]
    out = m + w * (center - m)
    pad = 2
    return np.pad(out, pad, mode="edge")[:img.shape[0], :img.shape[1]].astype(np.float32)


def load_dem() -> np.ndarray:
    """加载 DEM：真实 SRTM（64×64 约700m 抽样）双线性上采样到网格；
    失败回退模拟器地形（合成 DEM）。返回 (512,512) 米制高程。"""
    f = DATA / "dem_real.npy"
    if f.exists():
        g = np.load(f).astype(np.float64)
        return zoom(g, (NY / g.shape[0], NX / g.shape[1]), order=1)
    truth = np.load(DATA / "truth_grain.npz")
    return truth["elev"].astype(np.float64)      # 合成兜底（报告注明）


def local_incidence(dem: np.ndarray, theta: np.ndarray):
    """简化地形改正：由 DEM 坡度/坡向 + 视向方位计算局部入射角 θ_l（°）。

    cosθ_l = cosθ·cosβ + sinθ·sinβ·cos(φ−α)；返回 θ_l 与改正因子 cosθ_l/cosθ。
    """
    cs = 86.0                                    # 像元尺寸(m)
    dzdy, dzdx = np.gradient(dem, cs, cs)        # y 向南增 -> 坡向定义见下
    beta = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    alpha = np.degrees(np.arctan2(-dzdy, dzdx))  # 坡向（东为 0，逆时针）
    phi = np.radians(LOOK_AZ_DEG - 90.0)         # 视向（近似正东）
    th = np.radians(theta)[None, :]
    b = np.radians(beta)
    cos_tl = (np.cos(th) * np.cos(b)
              + np.sin(th) * np.sin(b) * np.cos(phi - np.radians(alpha)))
    theta_l = np.degrees(np.arccos(np.clip(cos_tl, -1.0, 1.0)))
    return theta_l, np.clip(cos_tl, 0.05, 1.0) / np.cos(th)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    dem = load_dem()
    print("== 02 预处理链：DEM 就绪，范围 "
          f"{dem.min():.2f}~{dem.max():.2f} m ==")
    logs = []
    for i in range(1, N_SCENES + 1):
        z = np.load(SIM_L1 / f"grd_scene_{i:02d}.npz", allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        dn_vv, dn_vh = z["dn_vv"], z["dn_vh"]
        gain, calib, nesz_vv, nesz_vh = float(z["gain"]), z["calib"], z["nesz_vv"], z["nesz_vh"]
        theta = z["theta"].astype(np.float64)

        geo = step1_orbit(meta)
        # --- VV 通道 ---
        i_vv, i_noise_vv = step2_thermal_noise(dn_vv, gain, nesz_vv, calib)
        sig0_vv = step3_calibration(i_vv, calib)
        # --- VH 通道 ---
        i_vh, i_noise_vh = step2_thermal_noise(dn_vh, gain, nesz_vh, calib)
        sig0_vh = step3_calibration(i_vh, calib)

        enl_vv_before = estimate_enl(sig0_vv)
        # --- ④ Lee Sigma vs 5×5 增强 Lee：按滤波前后 ENL 提升择优 ---
        ls = lee_sigma_filter(sig0_vv, enl_vv_before)
        enl_ls = estimate_enl(ls)
        el = lee_enhanced_5x5(sig0_vv)
        enl_el = estimate_enl(el)
        use_sigma = enl_ls >= enl_el
        sig0_vv_f = ls if use_sigma else el
        sig0_vh_f = lee_sigma_filter(sig0_vh, enl_vv_before) if use_sigma \
            else lee_enhanced_5x5(sig0_vh)

        # --- ⑤ 简化地形改正 ---
        theta_l, corr = local_incidence(dem, theta)
        sig0_vv_f = sig0_vv_f * corr.astype(np.float32)
        sig0_vh_f = sig0_vh_f * corr.astype(np.float32)

        np.savez_compressed(S1_PROC / f"proc_scene_{i:02d}.npz",
                            sig0_vv=sig0_vv_f, sig0_vh=sig0_vh_f,
                            theta=theta.astype(np.float32),
                            theta_l=theta_l.astype(np.float32),
                            corr=corr.astype(np.float32),
                            dem=dem.astype(np.float32))
        rec = dict(scene=i, date=meta["date"], orbit_check_ok=geo["orbit_check_ok"],
                   noise_removed_db_vv=round(float(np.median(
                       to_db(sig0_vv + (10.0 ** (nesz_vv / 10.0))[None, :])
                       - to_db(sig0_vv))), 2),
                   enl_before=round(enl_vv_before, 2), enl_after=round(enl_ls, 2),
                   enl_enh5=round(enl_el, 2), filter_chosen="lee_sigma" if use_sigma else "enh_lee5",
                   theta_l_max_dev_deg=round(float(np.abs(theta_l - theta[None, :]).max()), 4),
                   terrain_corr_max=round(float(np.abs(corr - 1).max()), 6))
        logs.append(rec)
        print(f"  景{i:02d}: ENL {rec['enl_before']}->{rec['enl_after']} "
              f"(5x5对照 {rec['enl_enh5']}, 取 {rec['filter_chosen']})；"
              f"热噪扣除 {rec['noise_removed_db_vv']} dB；地形改正 |Δ|≤{rec['terrain_corr_max']}")

    save_json(S1_PROC / "processing_log.json",
              {"win": WIN, "z_sigma": Z_SIGMA, "min_pix": MIN_PIX,
               "dem_source": "SRTM30m via opentopodata (700m 抽样上采样)"
                             if (DATA / "dem_real.npy").exists() else "合成 DEM 兜底",
               "scenes": logs})
    print("== 02 完成：s1_proc/*.npz + processing_log.json ==")


if __name__ == "__main__":
    main()
