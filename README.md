# Sentinel-1 时序相干 + 极化特征 GPR 反演潮滩沉积物粒度（含预测不确定性）

黄河三角洲 — 莱州湾潮滩 | Sentinel-1 GRD 预处理 → VV/VH/Pauli/时序相干/入射角/TWI →
高斯过程回归（均值 + 逐点后验 std）∥ XGBoost 对照 | 中科院考研复试作品集项目

> **D50 反演：GPR test R²=0.9109 / RMSE=8.74 μm**（XGBoost 0.8743 / 10.38），
> 95% 置信覆盖率 93.3%；指标明细见 `results/metrics.json`，报告见 `report/项目报告.md`。

## 快速开始（三行命令，全流程约 5 分钟，固定种子 20261104）

```bat
cd /d E:\zcode\探索前沿\s1-grain-size-gpr
E:\Ana\python.exe -I -X utf8 scripts\01_simulate_s1_timeseries.py
E:\Ana\python.exe -I -X utf8 -W ignore scripts\02_preprocess_grd.py
E:\Ana\python.exe -I -X utf8 -W ignore scripts\03_extract_features.py
E:\Ana\python.exe -I -X utf8 -W ignore scripts\04_build_dataset.py
E:\Ana\python.exe -I -X utf8 -W ignore scripts\05_train_gpr_xgb.py
E:\Ana\python.exe -I -X utf8 -W ignore scripts\06_make_maps.py
E:\Ana\python.exe -I -X utf8 test\test_pipeline.py
```

跑完看 `results/`（9 幅图 + metrics.json + figures_index.md）与 `report/`（项目报告 + 技术路线.drawio）。

## 目录说明

| 路径 | 内容 |
|---|---|
| `scripts/01~06` | 编号管线：模拟器 → 预处理 → 特征 → 建表 → 建模 → 出图 |
| `scripts/common.py` | 全局配置/种子/制图小工具（比例尺、指北针、colorbar） |
| `tools/` | 真实数据轨：`fetch_real_catalog.py`（CDSE OData 匿名清单）、`fetch_dem.py`（SRTM 30m，断点缓存）、`make_drawio.py`（重生成技术路线图） |
| `data/` | 模拟 L1 包（`sim_l1/`）、校正产品（`s1_proc/`）、真值场、特征栅格、采样/特征/预测表、真实清单与 DEM |
| `results/` | 图件、metrics.json（报告数字唯一来源）、figures_index.md |
| `report/` | 项目报告（六章）、技术路线 drawio/md |
| `test/` | 23 项断言自检 + 自检报告 |

## 数据策略（双轨制）

- **主轨（模拟，本文结果来源）**：物理机制驱动的 S1 GRD 时序模拟器——真值粒度场已知（河口粗源 + 高滩粗化 + 低能细粒），潮位驱动的淹没失相干机制，热噪声/斑点/量化全仿真，预处理链对其真实执行。为什么：GRD 下载需账号，且粒度反演精度只有带真值才能严格评估。
- **真实数据轨（已通两步）**：
  1. **场景清单**：`tools/fetch_real_catalog.py` 走 CDSE OData `OData.CSC.Intersects` 匿名查询（STAC GET 备用），研究区 2024-04~09 窗内 **46 景**真实 IW GRD（升轨），落盘 `data/real_scenes_manifest.json`。
  2. **真实 DEM**：`tools/fetch_dem.py` 从 opentopodata 抓 SRTM 30m（64×64 采样格网约 700 m 间隔，限流 1 req/s，断点缓存），供 TWI 与地形校正使用。

## 真实数据接入一条龙（需要你注册一个免费账号）

1. 注册 [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu/) 账号；
2. 从 `data/real_scenes_manifest.json` 挑选场景（`download_api` 字段给出 OData 下载 URL），命令行带 token 下载 SAFE 包（例）：
   `curl -H "Authorization: Bearer <你的token>" -L -o S1A_....SAFE.zip "<download_api>"`
   （或用 `s1dl`/`sentinelsat` 等工具；网络不稳建议走工作区 `scripts/fetch.ps1` 断点续传）
3. 解压 SAFE 后运行预处理（`scripts/02_preprocess_grd.py` 的 5 步链与模拟包同接口；轨道校正步骤需替换为 POD 精轨插值——占位处已在代码注释标明）；
4. **相干与 Pauli 建议改用 SLC 对**：GRD 为强度数据无法估时序相干，Pauli 需 SLC + 伪全极化重建（PQP）；这是真实接入的两个方法学调整点（报告 §5 已注明）；
5. 粒度采样表整理为 `lon,lat,sand,silt,clay,d50,source` CSV 覆盖 `data/samples.csv`，从 04 号脚本起重跑。

## 已知局限

见 `report/项目报告.md` §5（失相干二态简化、噪声底压低暗像元相干、采样稀疏与尺度 mismatch、GRD 双极化 Pauli 局限、模拟-真实域差距）。

## 依赖

见 `requirements.txt`（全部已在 `E:\Ana` 环境核验：numpy/scipy/pandas/scikit-learn/rasterio/xarray/matplotlib/xgboost 3.4.1/geopandas；**不依赖 cartopy**，空间图用 matplotlib 手绘制图要素）。

## 引用方向

Lee (1986) Sigma 滤波；ESA S1 IPF 热噪/定标口径；Rasmussen & Williams (2006) GPR；
黄河三角洲沉积物分带与莱州湾潮滩水动力公开研究。
