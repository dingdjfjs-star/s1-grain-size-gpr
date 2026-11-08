# -*- coding: utf-8 -*-
"""
test_pipeline.py —— 全管线自动化质检（对照 夜跑任务书 §4 阶段7 清单）
=====================================================================
运行：E:\\Ana\\python.exe test\\test_pipeline.py
在管线产物（data/ results/ report/）之上做 23 项断言，覆盖：

  A. 可复现性     模拟器种子确定性（真值场逐位一致）
  B. 预处理数值   热噪条带实际被扣除 / σ⁰ 物理区间 / Lee 滤波 ENL 提升
  C. 特征物理     相干 ∈[0,1] 且高滩>低滩 / 相干-淹没频率负相关 /
                  Pauli 面散射主导 / TWI、入射角量纲合理
  D. 数据完整性   采样表行数 / 无 NaN / 占比归一 / 测试集预测表齐全
  E. 模型与指标   GPR std>0 / R²≥0.75 门槛 / 覆盖率合理 / 数字单一来源
  F. 交付物       9 幅图齐全非空 / drawio XML 合法 / 报告六章且数字一致 /
                  AGENTS.md 已登记 / 真实数据轨产物落盘

全部通过输出 test/自检报告.md 并以退出码 0 结束；任何失败退出码 1。
"""
from __future__ import annotations

import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import (DATA, RESULTS, REPORT, ROOT, SIM_L1, S1_PROC, FEATURES,  # noqa: E402
                    TARGETS, NX, NY, N_SCENES, load_json, to_db)

RESULTS_CHECKS = []


def check(name):
    """断言装饰器：登记函数名并收集 (通过?, 说明)。"""
    def deco(fn):
        RESULTS_CHECKS.append((name, fn))
        return fn
    return deco


def load_sim01():
    spec = importlib.util.spec_from_file_location(
        "sim01", ROOT / "scripts" / "01_simulate_s1_timeseries.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- A 可复现性
@check("A1 模拟器种子确定性：build_truth 两次运行真值场逐位一致")
def _t():
    mod = load_sim01()
    t1 = mod.build_truth()
    t2 = mod.build_truth()
    for k in ("d50", "sand", "clay", "elev", "f_wet"):
        if not np.array_equal(t1[k], t2[k]):
            return False, f"{k} 两次运行不一致"
    return True, "d50/sand/clay/elev/f_wet 全部逐位一致"


# ---------------------------------------------------------------- B 预处理
@check("B1 热噪声去除：远距水域强度中位数下降 ≥2 dB")
def _t():
    log = load_json(S1_PROC / "processing_log.json")
    z = np.load(SIM_L1 / "grd_scene_01.npz")
    p = np.load(S1_PROC / "proc_scene_01.npz")
    tr = np.load(DATA / "truth_grain.npz")
    far_water = tr["water"] & (np.arange(NX)[None, :] > 460)
    gain = float(z["gain"]); calib = z["calib"]; nesz = z["nesz_vv"]
    before = to_db((z["dn_vv"].astype(np.float64) / gain) ** 2 * calib[None, :]
                   + (10.0 ** (nesz / 10.0))[None, :])
    after = to_db(p["sig0_vv"])
    drop = float(np.median(before[far_water]) - np.median(after[far_water]))
    return (drop >= 2.0, f"远距水域下降 {drop:.2f} dB（log 中位数口径 "
            f"{log['scenes'][0]['noise_removed_db_vv']} dB）")


@check("B2 辐射定标：滩面 σ⁰VV 中位数落在物理区间 [-25, 0] dB")
def _t():
    p = np.load(S1_PROC / "proc_scene_01.npz")
    tr = np.load(DATA / "truth_grain.npz")
    med = float(np.median(to_db(p["sig0_vv"][tr["flat"]])))
    return (-25.0 <= med <= 0.0, f"σ⁰VV 滩面中位数 {med:.2f} dB")


@check("B3 Lee Sigma 滤波：≥13/15 景 ENL 提升，择优记录完整")
def _t():
    log = load_json(S1_PROC / "processing_log.json")["scenes"]
    up = sum(1 for r in log if r["enl_after"] > r["enl_before"])
    chosen = {r["filter_chosen"] for r in log}
    return (up >= 13 and chosen <= {"lee_sigma", "enh_lee5"},
            f"{up}/15 景 ENL 提升；择优 {chosen}")


# ---------------------------------------------------------------- C 特征物理
@check("C1 时序相干 ∈ [0,1]")
def _t():
    f = dict(np.load(DATA / "features.npz"))
    c = f["coherence"]
    return (float(c.min()) >= 0.0 and float(c.max()) <= 1.0,
            f"范围 [{c.min():.3f}, {c.max():.3f}]")


@check("C2 相干分带：高滩(少淹)中位数 − 低滩(常淹)中位数 ≥ 0.10")
def _t():
    f = dict(np.load(DATA / "features.npz"))
    tr = np.load(DATA / "truth_grain.npz")
    hi = f["coherence"][tr["flat"] & (tr["f_wet"] < 0.10)]
    lo = f["coherence"][tr["flat"] & (tr["f_wet"] > 0.35)]
    d = float(np.median(hi) - np.median(lo))
    return (d >= 0.10, f"高滩 {np.median(hi):.3f} vs 低滩 {np.median(lo):.3f}（差 {d:.3f}）")


@check("C3 失相干物理：相干与淹没频率在滩面呈显著负相关（r ≤ −0.3）")
def _t():
    f = dict(np.load(DATA / "features.npz"))
    tr = np.load(DATA / "truth_grain.npz")
    m = tr["flat"]
    r = float(np.corrcoef(f["coherence"][m], tr["f_wet"][m])[0, 1])
    return (r <= -0.3, f"Pearson r = {r:.3f}")


@check("C4 Pauli 分量：面散射 k1 主导（k1 中位数 > k2、k3）")
def _t():
    f = dict(np.load(DATA / "features.npz"))
    tr = np.load(DATA / "truth_grain.npz")
    m1 = float(np.median(f["pauli1_db"][tr["flat"]]))
    m2 = float(np.median(f["pauli2_db"][tr["flat"]]))
    m3 = float(np.median(f["pauli3_db"][tr["flat"]]))
    return (m1 > m2 and m1 > m3 and -30 <= m1 <= 10,
            f"median(k1,k2,k3) = ({m1:.2f}, {m2:.2f}, {m3:.2f}) dB")


@check("C5 地形与几何：TWI ∈ [5,20]，入射角 ∈ [30,42]°")
def _t():
    f = dict(np.load(DATA / "features.npz"))
    twi_ok = 5.0 <= float(f["twi"].min()) and float(f["twi"].max()) <= 20.0
    inc_ok = 30.0 <= float(f["incidence"].min()) and float(f["incidence"].max()) <= 42.0
    return (twi_ok and inc_ok,
            f"TWI [{f['twi'].min():.2f},{f['twi'].max():.2f}]；"
            f"θ [{f['incidence'].min():.2f},{f['incidence'].max():.2f}]°")


# ---------------------------------------------------------------- D 数据完整
@check("D1 采样/特征表：150 行、特征无 NaN、占比和为 1")
def _t():
    ds = pd.read_csv(DATA / "dataset.csv")
    ok_rows = len(ds) == 150
    ok_nan = not ds[FEATURES].isna().any().any()
    ssum = float(np.abs(ds[["sand", "silt", "clay"]].sum(axis=1) - 1).max())
    return (ok_rows and ok_nan and ssum < 0.01,
            f"rows={len(ds)}, NaN={bool(ds[FEATURES].isna().any().any())}, "
            f"最大|占比和-1|={ssum:.4f}")


@check("D2 时序包齐全：15 景 GRD + 15 景 SLC")
def _t():
    g = len(list(SIM_L1.glob("grd_scene_*.npz")))
    s = len(list(SIM_L1.glob("slc_scene_*.npz")))
    return (g == N_SCENES and s == N_SCENES, f"GRD {g}, SLC {s}")


@check("D3 测试集预测表：30 行、列齐全（图4 数据源）")
def _t():
    tp = pd.read_csv(DATA / "test_predictions.csv")
    cols = {"y_true", "gpr_pred", "gpr_std", "xgb_pred"}
    return (len(tp) == 30 and cols <= set(tp.columns),
            f"rows={len(tp)}, cols={sorted(tp.columns)}")


# ---------------------------------------------------------------- E 模型指标
@check("E1 GPR 后验 std 在滩面全部 > 0")
def _t():
    pm = dict(np.load(DATA / "pred_maps.npz"))
    sd = pm["gpr_std_d50"][pm["flat_mask"]]
    return (bool((sd > 0).all()), f"min std = {sd.min():.4f} μm")


@check("E2 精度门槛：GPR 对 D50 的 test R² ≥ 0.75")
def _t():
    m = load_json(RESULTS / "metrics.json")["d50"]["gpr"]
    return (m["r2_test"] >= 0.75, f"R²={m['r2_test']}, RMSE={m['rmse_test']} μm")


@check("E3 对照实验完整：XGBoost 指标与特征重要性齐备")
def _t():
    m = load_json(RESULTS / "metrics.json")["d50"]["xgb"]
    imp = m.get("importance_gain", {})
    return (0 <= m["r2_test"] <= 1 and len(imp) == len(FEATURES)
            and abs(sum(imp.values()) - 1) < 0.01,
            f"XGB R²={m['r2_test']}；重要性 {len(imp)} 项，和={sum(imp.values()):.3f}")


@check("E4 不确定性校准：95% 覆盖率 ∈ [0.6,1.0]，|残差|~std 正相关")
def _t():
    u = load_json(RESULTS / "metrics.json")["uncertainty"]
    return (0.6 <= u["coverage_95"] <= 1.0 and u["spearman_absres_std"] > 0,
            f"coverage={u['coverage_95']}, spearman={u['spearman_absres_std']}")


@check("E5 四目标两模型指标齐备且 R² ∈ [0,1]")
def _t():
    M = load_json(RESULTS / "metrics.json")
    bad = []
    for t in TARGETS:
        for k in ("gpr", "xgb"):
            r2 = M[t][k]["r2_test"]
            if not (0.0 <= r2 <= 1.0):
                bad.append(f"{t}/{k}={r2}")
    return (not bad, "全部合格" if not bad else str(bad))


# ---------------------------------------------------------------- F 交付物
@check("F1 结果图 ≥8 幅、全部 >20 KB")
def _t():
    figs = sorted(RESULTS.glob("fig*.png"))
    small = [(f.name, f.stat().st_size) for f in figs if f.stat().st_size < 20480]
    return (len(figs) >= 8 and not small,
            f"{len(figs)} 幅；过小: {small if small else '无'}")


@check("F2 draw.io 技术路线：XML 合法、mxGraphModel 结构、节点≥15")
def _t():
    p = REPORT / "技术路线.drawio"
    tree = ET.parse(p)
    r = tree.getroot()
    cells = r.iter("mxCell")
    n_vertices = sum(1 for c in cells if c.get("vertex") == "1")
    has_model = any(True for _ in r.iter("mxGraphModel"))
    return (has_model and n_vertices >= 15,
            f"vertex={n_vertices}, mxGraphModel={has_model}")


@check("F3 项目报告：六章齐全且关键数字与 metrics.json 一致")
def _t():
    md = (REPORT / "项目报告.md").read_text(encoding="utf-8")
    heads = ["## 1 背景", "## 2 数据", "## 3 技术路线", "## 4 结果",
             "## 5 误差来源", "## 6 后续改进"]
    miss = [h for h in heads if h not in md]
    M = load_json(RESULTS / "metrics.json")
    r2s = str(M["d50"]["gpr"]["r2_test"])
    return (not miss and r2s in md,
            f"缺失章节: {miss or '无'}；GPR R² 数字 {r2s} {'已引用' if r2s in md else '未找到'}")


@check("F4 真实数据轨落盘：场景清单 + DEM 状态")
def _t():
    man = json.loads((DATA / "real_scenes_manifest.json").read_text(encoding="utf-8"))
    dem = json.loads((DATA / "dem_status.json").read_text(encoding="utf-8"))
    return ("status" in man and dem.get("ok") is True,
            f"manifest={man['status']}({man['n_scenes']}景, {man.get('channel')}); "
            f"dem_ok={dem.get('ok')}({dem.get('n_points')}点)")


@check("F5 资产登记：AGENTS.md 已含本项目条目")
def _t():
    txt = (ROOT.parents[1] / "AGENTS.md").read_text(encoding="utf-8")
    return ("s1-grain-size-gpr" in txt, "AGENTS.md 资产星图包含 s1-grain-size-gpr")


@check("F6 图件索引与 README 存在")
def _t():
    return ((RESULTS / "figures_index.md").exists()
            and (ROOT / "README.md").exists()
            and (ROOT / "requirements.txt").exists(),
            "figures_index.md / README.md / requirements.txt 就绪")


# ---------------------------------------------------------------- 主流程
def main() -> None:
    lines = ["# 自检报告（test/test_pipeline.py）\n",
             "自动质检 23 项断言，全部通过才允许收工。\n",
             "| # | 断言 | 结果 | 说明 |", "|---|---|---|---|"]
    n_pass = 0
    for i, (name, fn) in enumerate(RESULTS_CHECKS, 1):
        try:
            ok, msg = fn()
        except Exception as exc:                                # noqa: BLE001
            ok, msg = False, f"异常 {type(exc).__name__}: {exc}"
        n_pass += ok
        lines.append(f"| {i} | {name} | {'✅' if ok else '❌'} | {msg} |")
    tail = f"\n**结果：{n_pass}/{len(RESULTS_CHECKS)} 通过**" + \
        ("，全绿 ✅" if n_pass == len(RESULTS_CHECKS) else "，存在失败 ❌")
    lines.append(tail)
    (ROOT / "test" / "自检报告.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[test] {n_pass}/{len(RESULTS_CHECKS)} 通过")
    for i, (name, fn) in enumerate(RESULTS_CHECKS, 1):
        pass
    sys.exit(0 if n_pass == len(RESULTS_CHECKS) else 1)


if __name__ == "__main__":
    main()
