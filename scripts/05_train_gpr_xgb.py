# -*- coding: utf-8 -*-
"""
05_train_gpr_xgb.py —— GPR 粒度反演（均值+不确定性）与 XGBoost 对照
====================================================================
数据：data/dataset.csv（150 样本 × 9 特征），目标 D50（主）+ 砂/粉砂/黏土占比（副）。

模型：
  * GPR（主模型）：StandardScaler + ConstantKernel·RBF + WhiteKernel，
    normalize_y，n_restarts=4；预测输出均值与逐点后验标准差 std ——
    不确定性图是本研究核心产出；
  * XGBoost（对照）：hist 直方图算法，500 树；输出特征重要性（gain）。
评估：80/20 固定种子划分 + 训练集 5 折交叉验证；R² 与 RMSE；
不确定性质量：test 集 |残差| 与 GPR std 的 Spearman 相关、95% 覆盖率。
全栅格逐像元预测（分块）→ 空间反演图 / 不确定性图所需场。

输出：results/metrics.json（报告引用的唯一数字来源）
      data/pred_maps.npz（gpr_mean/gpr_std/xgb_mean 的 D50 空间场）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from common import (DATA, RESULTS, DATASET_CSV, METRICS_JSON, SEED, FEATURES,
                    TARGETS, NX, NY, save_json)


def make_gpr():
    """GPR 管线：标准化 + 常数·RBF + 白噪声核。"""
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e3)) + WhiteKernel(1e-2, (1e-6, 1e1))
    return make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                 n_restarts_optimizer=4, random_state=SEED))


def make_xgb():
    """XGBoost 对照模型。"""
    return XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4,
                        subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                        n_jobs=4, tree_method="hist", verbosity=0)


def eval_regressor(model, Xtr, ytr, Xte, yte):
    """拟合 + test R²/RMSE + 训练集 5 折 CV；返回 dict。"""
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    r2 = float(1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2))
    rmse = float(np.sqrt(np.mean((yte - pred) ** 2)))
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_r2 = cross_val_score(model, Xtr, ytr, cv=kf, scoring="r2")
    cv_rmse = -cross_val_score(model, Xtr, ytr, cv=kf,
                               scoring="neg_root_mean_squared_error")
    return dict(r2_test=round(r2, 4), rmse_test=round(rmse, 4),
                r2_cv_mean=round(float(cv_r2.mean()), 4),
                r2_cv_std=round(float(cv_r2.std()), 4),
                rmse_cv_mean=round(float(cv_rmse.mean()), 4),
                rmse_cv_std=round(float(cv_rmse.std()), 4)), pred


def predict_grid(model, feats_flat: np.ndarray, chunk: int = 40000):
    """全栅格分块预测，返回 (N,) 均值与（GPR 才有）std。"""
    from sklearn.pipeline import Pipeline
    is_gpr = isinstance(model, Pipeline) and isinstance(
        model.steps[-1][1], GaussianProcessRegressor)
    n = feats_flat.shape[0]
    out = np.empty(n, np.float64)
    std = np.empty(n, np.float64) if is_gpr else None
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        if is_gpr:
            mu, sd = model.predict(feats_flat[a:b], return_std=True)
            out[a:b] = mu
            std[a:b] = sd
        else:
            out[a:b] = model.predict(feats_flat[a:b])
    return out, std


def main() -> None:
    ds = pd.read_csv(DATASET_CSV)
    X = ds[FEATURES].to_numpy()
    flat_mask = np.load(DATA / "truth_grain.npz")["flat"]     # 仅用于限定制图范围
    metrics = {"seed": SEED, "n_samples": len(ds),
               "n_train": None, "n_test": None, "features": FEATURES,
               "units": {"d50": "μm", "sand/silt/clay": "%(0-1)"},
               "cv": "5-fold(shuffle,seed=%d)" % SEED}

    idx_tr, idx_te = train_test_split(np.arange(len(ds)), test_size=0.2,
                                      random_state=SEED)
    metrics["n_train"], metrics["n_test"] = len(idx_tr), len(idx_te)
    maps = {}

    for t in TARGETS:
        y = ds[t].to_numpy()
        Xtr, ytr = X[idx_tr], y[idx_tr]
        Xte, yte = X[idx_te], y[idx_te]
        metrics[t] = {}

        # --- GPR ---
        gpr = make_gpr()
        rec, pred_te = eval_regressor(gpr, Xtr, ytr, Xte, yte)
        gpr.fit(X, y)                              # 全样本重拟合用于制图
        kernel_str = str(gpr.steps[-1][1].kernel_)
        rec["kernel_fitted"] = kernel_str
        test_df = None
        if t == "d50":
            # 不确定性质量评估（主目标）+ 保存测试集预测供图 4 使用
            gpr_te = make_gpr()
            gpr_te.fit(Xtr, ytr)
            mu_te, sd_te = gpr_te.predict(Xte, return_std=True)
            res = np.abs(yte - mu_te)
            metrics["uncertainty"] = {
                "spearman_absres_std": round(float(spearmanr(res, sd_te).statistic), 4),
                "coverage_95": round(float(np.mean(res <= 1.96 * sd_te)), 4),
                "std_median_test": round(float(np.median(sd_te)), 4)}
            test_df = pd.DataFrame({"y_true": yte, "gpr_pred": mu_te,
                                    "gpr_std": sd_te})
        metrics[t]["gpr"] = rec

        # --- XGBoost ---
        xgb = make_xgb()
        rec2, pred_xgb = eval_regressor(xgb, Xtr, ytr, Xte, yte)
        rec2["importance_gain"] = {f: round(float(v), 4) for f, v in
                                   zip(FEATURES, xgb.feature_importances_)}
        metrics[t]["xgb"] = rec2
        if t == "d50":
            test_df["xgb_pred"] = pred_xgb
            test_df.to_csv(DATA / "test_predictions.csv", index=False)

    # ---- 全栅格制图（D50）：由 features.npz 展开，与 04 同口径 3×3 窗均值 ----
    from scipy.ndimage import uniform_filter
    feats = dict(np.load(DATA / "features.npz"))
    Xg = np.stack([uniform_filter(feats[f].astype(np.float64), 3, mode="nearest").ravel()
                   for f in FEATURES], axis=1)

    gpr = make_gpr(); gpr.fit(X, ds["d50"].to_numpy())
    mu_g, sd_g = predict_grid(gpr, Xg)
    xgb = make_xgb(); xgb.fit(X, ds["d50"].to_numpy())
    mu_x, _ = predict_grid(xgb, Xg)
    maps = dict(gpr_mean_d50=mu_g.reshape(NY, NX).astype(np.float32),
                gpr_std_d50=sd_g.reshape(NY, NX).astype(np.float32),
                xgb_mean_d50=mu_x.reshape(NY, NX).astype(np.float32),
                flat_mask=flat_mask)
    np.savez_compressed(DATA / "pred_maps.npz", **maps)

    save_json(METRICS_JSON, metrics)

    # --- 控制台摘要 ---
    m = metrics["d50"]
    print("== 05 完成 ==")
    print(f"  D50  GPR : R²(test)={m['gpr']['r2_test']}, RMSE={m['gpr']['rmse_test']} μm, "
          f"CV R²={m['gpr']['r2_cv_mean']}±{m['gpr']['r2_cv_std']}")
    print(f"  D50  XGB : R²(test)={m['xgb']['r2_test']}, RMSE={m['xgb']['rmse_test']} μm, "
          f"CV R²={m['xgb']['r2_cv_mean']}±{m['xgb']['r2_cv_std']}")
    print(f"  不确定性: |残差|~std Spearman={metrics['uncertainty']['spearman_absres_std']}, "
          f"95%覆盖率={metrics['uncertainty']['coverage_95']}")
    print(f"  占比各目标 R²(GPR/XGB): " +
          ", ".join(f"{t} {metrics[t]['gpr']['r2_test']}/{metrics[t]['xgb']['r2_test']}"
                    for t in ("sand", "silt", "clay")))
    print(f"  -> {METRICS_JSON.name} / pred_maps.npz")


if __name__ == "__main__":
    main()
