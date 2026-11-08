# 结果图件索引（results/）

| 图 | 文件 | 说明 |
|---|---|---|
| 图1 | fig01_study_area.png | 研究区概况：模拟真值 D50 与 150 个采样点 |
| 图2 | fig02_gpr_inversion.png | GPR 空间反演图（D50） |
| 图3 | fig03_gpr_uncertainty.png | GPR 不确定性分布图（后验 std） |
| 图4 | fig04_scatter.png | 散点拟合：GPR vs XGBoost（标注 metrics.json 的 test R²/RMSE） |
| 图5 | fig05_coherence.png | 时序平均相干系数图（VV 通道） |
| 图6 | fig06_pauli_rgb.png | Pauli RGB 合成 + σ0 VV + σ0 VH 三联 |
| 图7 | fig07_importance.png | XGBoost 特征重要性 + GPR 核结构 |
| 图8 | fig08_model_compare.png | GPR vs XGBoost 空间反演对比 |
| 图9 | fig09_inundation.png | 潮汐淹没频率（模拟真值，失相干驱动场） |

所有空间图均含 colorbar 图例（带单位）、左下比例尺（km）、右上指北针。
图4 散点为 05 号脚本保存的留出测试集真实预测（data/test_predictions.csv）。