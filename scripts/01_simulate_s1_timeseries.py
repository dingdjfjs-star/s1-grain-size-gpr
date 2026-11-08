# -*- coding: utf-8 -*-
"""
01_simulate_s1_timeseries.py —— Sentinel-1 GRD 时序模拟器（主轨数据源，含真值）
================================================================================
为什么需要模拟器：
  Sentinel-1 GRD 下载需 CDSE 账号（本机未配置凭据），本项目采用"双轨制"：
  主轨用物理机制驱动的模拟数据端到端跑通全部流程（真值粒度场已知，可评估反演
  精度）；真实数据轨由 tools/ 下脚本提供场景清单与 DEM，接入方法见 README。

模拟什么（物理上自洽、与真实 S1 时序研究口径一致）：
  1. 真值场：合成 DEM（沿海平原+潮滩分带）-> 潮位统计 -> 淹没频率 f_wet；
     粒度分区（河口叶瓣偏粗、低能滩偏细、高滩粗化低滩细化）-> D50 与砂/粉砂/
     黏土占比；微地貌粗糙度 s 随 D50 增大。
  2. SAR 正演：15 景 IW GRD 等效数据（VV+VH，升轨，12 天重访）。
     每景先抽潮位 -> 每像元得到湿/干状态与土壤湿度 -> 经验模型给出 σ⁰VV/VH
     （粗糙度↑、湿度↑则亮；入射角有跨轨依赖；水面低后向散射且随风变化）->
     按 Pauli 基生成复散射（k1 奇次面散射 / k2 二次分量 / k3 体散射去极化）。
  3. 时序失相干（本研究物理核心）：淹没会改变地表（水平化+沉积物再搬运），
     干燥像元保留相干纹理。机制：每景复场 = w_i·持久纹理 C_p + sqrt(1-w_i²)
     ·新息噪声，w 干=0.766 / 湿=0.180，多视后景对相干期望约
     干-干 0.85 / 干-湿 0.30 / 湿-湿 0.12。
  4. 辐射量化：强度叠加热噪声（NESZ 随距向变化，远距条带可见）-> 仿 GRD
     定标逆变换为 uint16 DN + 定标/噪声 LUT（02 号脚本将把它们真实地去除）。

输出：
  data/truth_grain.npz        真值场（D50/占比/高程/淹没频率/粗糙度/掩膜）
  data/truth_d50.tif          真值 D50 GeoTIFF（GIS 可直接打开）
  data/sim_l1/grd_scene_NN.npz   等效 L1 GRD 包（DN + LUT + 几何 + 元数据）
  data/sim_l1/slc_scene_NN.npz   等效 SLC 包（多视复 Pauli 分量，供相干估计）
  data/sim_summary.json       每景潮位/淹没比/类median检验表（供自检）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from common import (DATA, SIM_L1, LON0, LON1, LAT0, LAT1, MOUTH_LON, MOUTH_LAT,
                    NX, NY, DLON, DLAT, SEED, N_SCENES, TIDE_STD,
                    ELEV_WATER_MAX, ELEV_MARSH_MIN, LAMBDA, grid_lonlat,
                    lonlat_to_rc, scene_dates, save_json, to_db)

GAIN = 600.0          # DN 量化增益：DN = sqrt(I_measured) * GAIN（K = 1/GAIN）
W_DRY = 0.766         # 干燥像元持久纹理权重（多视后景对相干期望约 0.85）
W_WET = 0.180         # 淹没像元纹理权重（多视后景对相干期望约 0.12）
N_SUBLOOK = 4         # 子视数：模拟 GRD 检测前的多视平均


# ---------------------------------------------------------------------------
# 1. 真值场
# ---------------------------------------------------------------------------

def build_truth():
    """构造研究区真值场（DEM/淹没频率/粒度/粗糙度），返回 dict。"""
    rng = np.random.default_rng(SEED)
    lon, lat = grid_lonlat()
    # 归一化横断坐标：0=西端（陆），1=东端（海）
    x = (lon - LON0) / (LON1 - LON0)

    # 1a. 合成 DEM：沿海平原向东缓倾入莱州湾，tanh 剖面使潮间带宽度约 3.5 km
    km_lon = 111.32 * np.cos(np.deg2rad(37.75))                  # °经 -> km
    km_lat = 110.6                                               # °纬 -> km
    d_mouth = np.hypot((lon - MOUTH_LON) * km_lon,
                       (lat - MOUTH_LAT) * km_lat)               # 离河口距离(km)
    coast_shift = gaussian_filter(rng.normal(0, 1, (NY, NX)), 22) * 1.5   # 岸线蜿蜒(±1.3 km)
    mouth_bulge = 0.30 * np.exp(-(d_mouth / 2.2) ** 2)          # 河口叶瓣向海凸出
    undul = gaussian_filter(rng.normal(0, 1, (NY, NX)), 10) * 5.0           # 微起伏(±0.2 m)
    elev = (0.85 - 2.20 * np.tanh((x - 0.62 + coast_shift) / 0.10)
            + mouth_bulge + undul)

    # 1b. 潮位统计 -> 淹没频率：潮位近似 N(0, TIDE_STD)
    from scipy.special import erfc
    f_wet = 0.5 * erfc(elev / (TIDE_STD * np.sqrt(2.0)))        # P(潮位 > 高程)

    # 1c. 掩膜：潮下带水域 / 潮间带滩面 / 高潮滩上带
    water = elev < ELEV_WATER_MAX
    marsh = elev > ELEV_MARSH_MIN
    flat = (~water) & (~marsh)

    # 1d. 粒度分区真值（锚定黄河三角洲公开研究的定性格局）：
    #     河口叶瓣/拦门砂粗（砂质），莱州湾沿岸低能滩细（粉砂-黏土），
    #     高滩暴露粗化、低滩细化；叠加平滑小尺度变异。
    d50 = (12.0
           + 78.0 * np.exp(-(d_mouth / 6.5) ** 2)               # 河口源项
           + 16.0 * (elev / 1.7)                                 # 高滩粗化
           + gaussian_filter(rng.normal(0, 1, (NY, NX)), 7) * 120.0)  # 小尺度变异(±5 μm)
    d50 = np.clip(d50, 8.0, 132.0)
    # 砂占比由 D50 经 logistic 映射（63 μm 为砂/粉砂界限）
    sand = 0.93 / (1.0 + np.exp(-(d50 - 58.0) / 9.0)) + 0.01
    clay = (0.05 + 0.22 * (1.0 - sand / 0.94)
            * gaussian_filter(rng.normal(1, 0.25, (NY, NX)), 9).clip(0.2, 1.8))
    clay = np.clip(clay, 0.02, 0.42)
    silt = np.clip(1.0 - sand - clay, 0.05, 0.93)
    tot = sand + silt + clay
    sand, silt, clay = sand / tot, silt / tot, clay / tot        # 严格归一

    # 1e. 微地貌粗糙度 rms 高度（cm）：随 D50 线性增大（波痕/裂隙发育）
    s_cm = 0.30 + 0.016 * d50

    np.savez_compressed(DATA / "truth_grain.npz", lon=lon, lat=lat, elev=elev,
                        f_wet=f_wet, d50=d50, sand=sand, silt=silt, clay=clay,
                        s_cm=s_cm, water=water, flat=flat, marsh=marsh)
    _write_truth_tif(d50)
    return dict(lon=lon, lat=lat, elev=elev, f_wet=f_wet, d50=d50, sand=sand,
                silt=silt, clay=clay, s_cm=s_cm, water=water, flat=flat,
                marsh=marsh, d_mouth=d_mouth)


def _write_truth_tif(d50):
    """真值 D50 存 GeoTIFF（EPSG:4326，便于 GIS 软件检查）。"""
    try:
        import rasterio
        from rasterio.transform import from_bounds
        with rasterio.open(DATA / "truth_d50.tif", "w", driver="GTiff",
                           height=NY, width=NX, count=1, dtype="float32",
                           crs="EPSG:4326",
                           transform=from_bounds(LON0, LAT0, LON1, LAT1, NX, NY)) as dst:
            dst.write(d50.astype("float32"), 1)
    except Exception as exc:                                    # noqa: BLE001
        print(f"  truth GeoTIFF 跳过: {exc}")


# ---------------------------------------------------------------------------
# 2. SAR 正演
# ---------------------------------------------------------------------------

def theta_profile():
    """跨轨入射角（°）：IW 升轨 31.5 -> 41.0（近距->远距，仅随列变化）。"""
    return 31.5 + 9.5 * np.arange(NX) / (NX - 1)


def sigma0_truth(truth, tide_i, wind, theta):
    """给定本景潮位与风场，计算每像元真值后向散射 σ⁰VV/VH（线性功率）。

    经验正演（简化，文档口径见报告第 5 章）：
      σ⁰VV_dB = -14 + 14·log10(s/0.45cm) - 0.22·(θ-36.5°) + 24·(mv-0.25) + 平滑扰动
      σ⁰VH_dB = σ⁰VV_dB - (9.0 + 4.5·(1-粗糙度归一))      （去极化随粗糙度增大而减小）
      水面：   σ⁰VV_dB = -21.5 + 风场项 - 0.15·(θ-36.5°)，VH = VV - 8
    """
    elev, s_cm = truth["elev"], truth["s_cm"]
    wet = elev < tide_i                                          # 本景被淹没
    rough_norm = np.clip((np.log10(s_cm) - np.log10(0.3))
                         / (np.log10(2.5) - np.log10(0.3)), 0.0, 1.0)
    mv = np.where(wet, 0.46, np.clip(0.14 + 0.10 * (1 - elev / 1.7), 0.10, 0.42))
    mv = np.where(truth["water"], 0.0, mv)

    vv = (-14.0 + 14.0 * np.log10(np.maximum(s_cm, 1e-3) / 0.45)
          - 0.22 * (theta[None, :] - 36.5) + 24.0 * (mv - 0.25))
    vv = vv + gaussian_filter(rng_local.normal(0, 1, (NY, NX)), 6) * 0.4
    vh = vv - (9.0 + 4.5 * (1.0 - rough_norm))

    wv_vv = (-22.5 - 0.15 * (theta[None, :] - 36.5)
             + wind * 1.0 + gaussian_filter(rng_local.normal(0, 1, (NY, NX)), 8) * 0.3)
    vv = np.where(truth["water"], wv_vv, vv)
    vh = np.where(truth["water"], vv - 8.0, vh)
    return wet, 10.0 ** (vv / 10.0), 10.0 ** (vh / 10.0)         # 线性功率


# 全局 rng：01 号脚本内部顺序使用，保证可复现
rng_local = np.random.default_rng(SEED)


def complex_field(power, w_i, c_text, rng):
    """生成一个 Pauli 分量的多视复场（返回多视后的 complex64 与多视强度）。

    复场模型：S_ik = sqrt(power)·( w_i·C_p + sqrt(1-w_i²)·E_ik )
      C_p —— 持久纹理（跨景/跨子视共享，承载时序相干）
      E_ik —— 新息噪声（每子视独立）
    返回：mean_k S_ik（complex64，等效多视 SLC）、mean_k |S_ik|²（等效 GRD 强度）
    """
    amp = np.sqrt(power, dtype=np.float64)
    acc_m = np.zeros_like(c_text)
    acc_i = np.zeros(c_text.shape, dtype=np.float64)   # 强度累加器必须是实数
    for _ in range(N_SUBLOOK):
        e = rng.standard_normal(c_text.shape) + 1j * rng.standard_normal(c_text.shape)
        s = amp * (w_i * c_text + np.sqrt(np.maximum(1.0 - w_i ** 2, 0.0)) * e)
        acc_m += s
        acc_i += np.abs(s) ** 2
    return (acc_m / N_SUBLOOK).astype(np.complex64), (acc_i / N_SUBLOOK)


def simulate_scene(truth, tide_i, theta, rng, c_text):
    """模拟一景：返回 GRD 包 dict 与 SLC 包 dict。

    c_text —— (c1, c2, c3) 持久纹理复场，跨景共享（时序相干的物理载体）。
    """
    lon, lat = truth["lon"], truth["lat"]
    wind = float(rng.normal(0, 1))                               # 本景海面风况
    wet, p_vv, p_vh = sigma0_truth(truth, tide_i, wind, theta)

    # --- Pauli 分量功率：σ⁰HH=|k1+k2|²/2, σ⁰VV=|k1-k2|²/2, σ⁰HV=|k3|²/2 ---
    #     二次分量很小（裸滩面），其功率记 P2 = (2σ⁰VV)·(0.03+0.05·粗糙度)
    rough_norm = np.clip((np.log10(truth["s_cm"]) - np.log10(0.3))
                         / (np.log10(2.5) - np.log10(0.3)), 0.0, 1.0)
    p2 = 2.0 * p_vv * (0.03 + 0.05 * rough_norm)
    p1 = np.maximum(2.0 * p_vv - p2, 1e-9)
    p3 = 2.0 * p_vh

    c1, c2, c3 = c_text

    # 本景失相干权重：干燥/淹没（VH 与 VV 同权重，简化）
    wet_frac_px = float(wet.mean())
    w_field = np.where(wet, W_WET, W_DRY)

    k1_m, i1 = complex_field(p1, w_field, c1, rng)
    k2_m, i2 = complex_field(p2, w_field, c2, rng)
    k3_m, i3 = complex_field(p3, w_field, c3, rng)

    # --- 等效 GRD 强度（VV/VH 通道）---
    i_vv = (i1 + i2 - 2.0 * np.real(k1_m * np.conj(k2_m))) / 2.0   # mean|k1-k2|²/2
    i_vv = np.maximum(i_vv, 1e-9)
    i_vh = np.maximum(i3 / 2.0, 1e-9)

    # --- 定标/噪声 LUT（σ⁰ = I_sig·calib；I_meas = I_sig + NESZ/calib）---
    calib = np.cos(np.deg2rad(36.0)) / np.cos(np.deg2rad(theta))  # 随入射角增大
    nesz_vv_dB = -22.0 + 2.5 * (np.arange(NX) / (NX - 1)) ** 2    # 远距 NESZ 恶化
    nesz_vh_dB = nesz_vv_dB - 3.0
    i_noise_vv = (10.0 ** (nesz_vv_dB / 10.0)) / calib
    i_noise_vh = (10.0 ** (nesz_vh_dB / 10.0)) / calib

    # --- 量化为 uint16 DN（含热噪声，模拟 L1 GRD 原始记录）---
    dn_vv = np.clip(np.round(np.sqrt(i_vv + i_noise_vv) * GAIN), 0, 65535).astype(np.uint16)
    dn_vh = np.clip(np.round(np.sqrt(i_vh + i_noise_vh) * GAIN), 0, 65535).astype(np.uint16)

    # --- 等效 SLC：多视复场 + 热噪声（供 03 相干估计；真实轨应下载 SLC 对）---
    n_slc = np.sqrt((10.0 ** (nesz_vv_dB / 10.0)) / N_SUBLOOK)
    m_vv = ((k1_m - k2_m) / np.sqrt(2.0)
            + n_slc * (rng.standard_normal((NY, NX)) + 1j * rng.standard_normal((NY, NX))))
    m_vh = k3_m / np.sqrt(2.0) + n_slc * (
        rng.standard_normal((NY, NX)) + 1j * rng.standard_normal((NY, NX)))

    grd = dict(dn_vv=dn_vv, dn_vh=dn_vh, theta=theta, calib=calib.astype(np.float32),
               nesz_vv=nesz_vv_dB.astype(np.float32), nesz_vh=nesz_vh_dB.astype(np.float32),
               gain=GAIN)
    slc = dict(m_vv=m_vv.astype(np.complex64), m_vh=m_vh.astype(np.complex64),
               k1=k1_m, k2=k2_m, k3=k3_m)
    stats = dict(wind=wind, wet_frac=wet_frac_px)
    return grd, slc, stats, wet


# ---------------------------------------------------------------------------
# 3. 主流程
# ---------------------------------------------------------------------------

def main() -> None:
    print("== 01 模拟器：构造真值场 ==")
    truth = build_truth()
    fw = truth["f_wet"][truth["flat"]]
    print(f"  滩面像元 {truth['flat'].sum()}（水域 {truth['water'].sum()}）"
          f"；滩面淹没频率 {fw.min():.2f}~{fw.max():.2f}")
    print(f"  D50 滩面范围 {truth['d50'][truth['flat']].min():.0f}"
          f"~{truth['d50'][truth['flat']].max():.0f} μm")

    theta = theta_profile()
    dates = scene_dates()
    rng = np.random.default_rng(SEED + 1)
    # 持久纹理：每像元每 Pauli 分量一个复高斯纹理，跨全部 15 景共享
    #（它承载时序相干；各景新息噪声在 complex_field 内逐子视独立生成）
    c_text = tuple(rng.standard_normal((NY, NX)) + 1j * rng.standard_normal((NY, NX))
                   for _ in range(3))
    summary = []
    for i in range(N_SCENES):
        # 潮位：半日潮+大小潮周期在固定成像时刻的采样，跨 15 景覆盖干湿各样
        tide_i = 0.85 * np.sin(2 * np.pi * (i * 12 + 3.0) / 14.77) + rng.normal(0, 0.05)
        grd, slc, stats, wet = simulate_scene(truth, tide_i, theta, rng, c_text)
        meta = dict(scene=i + 1, date=dates[i], orbit="ASC", pol="VV+VH",
                    tide_level_m=float(tide_i), wet_frac=stats["wet_frac"],
                    wind_z=stats["wind"], look_az_deg=95.0,
                    note="模拟等效 L1 GRD 包；DN = sqrt(I_sig+I_noise)*gain")
        np.savez_compressed(SIM_L1 / f"grd_scene_{i+1:02d}.npz",
                            dn_vv=grd["dn_vv"], dn_vh=grd["dn_vh"],
                            theta=grd["theta"], calib=grd["calib"],
                            nesz_vv=grd["nesz_vv"], nesz_vh=grd["nesz_vh"],
                            gain=grd["gain"], meta=json.dumps(meta, ensure_ascii=False))
        np.savez_compressed(SIM_L1 / f"slc_scene_{i+1:02d}.npz",
                            m_vv=slc["m_vv"], m_vh=slc["m_vh"],
                            k1=slc["k1"], k2=slc["k2"], k3=slc["k3"])
        # 类别 median 检验（真值口径，供自检报告引用）
        dry_flat = truth["flat"] & (~wet)
        mud = dry_flat & (truth["d50"] < 25)
        sandy = dry_flat & (truth["d50"] > 60)
        s = dict(scene=i + 1, date=dates[i], tide=round(float(tide_i), 3),
                 wet_frac=round(stats["wet_frac"], 3),
                 vv_mud_dB=round(_med_db(mud, grd), 2) if mud.any() else None,
                 vv_sand_dB=round(_med_db(sandy, grd), 2) if sandy.any() else None,
                 vv_water_dB=round(_med_db(truth["water"], grd), 2))
        summary.append(s)
        print(f"  景{i+1:02d} {dates[i]} 潮位 {tide_i:+.2f} m 淹没率 {stats['wet_frac']:.2f}"
              f" | σ⁰VV(mud/sand/water) {s['vv_mud_dB']} / {s['vv_sand_dB']} / {s['vv_water_dB']} dB")

    save_json(DATA / "sim_summary.json",
              {"seed": SEED, "n_scenes": N_SCENES, "scenes": summary,
               "w_dry": W_DRY, "w_wet": W_WET, "n_sublook": N_SUBLOOK,
               "expected_coh": {"dry_dry": 0.85, "dry_wet": 0.30, "wet_wet": 0.12}})
    print("== 01 完成：truth_grain.npz / truth_d50.tif / sim_l1/*.npz / sim_summary.json ==")


def _med_db(mask, grd):
    """掩膜内 σ⁰VV 中位数（dB）—— 由 DN 直接反推强度（未定标，仅自检量级）。"""
    I = (grd["dn_vv"].astype(np.float64) / grd["gain"]) ** 2
    return float(np.median(10 * np.log10(I[mask]))) if mask.any() else float("nan")


if __name__ == "__main__":
    main()
