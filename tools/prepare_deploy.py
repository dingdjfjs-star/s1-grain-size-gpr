# -*- coding: utf-8 -*-
"""准备 GitHub Pages 部署拷贝：镜像 代码+文档+图件+指标 到 ../s1-grain-size-gpr-site/

排除项（不进仓库）：
- data/sim_l1/  data/s1_proc/  data/*.npz  data/*.tif   模拟 SAR 时序与中间栅格（~百 MB，
  scripts 01~03 固定种子可重新生成；README 已写明）
- .venv / __pycache__ / *.pyc / .git / .zcode
保留项：代码、README、index.html、requirements、report/（报告+drawio）、results/（图件+
metrics）、data/ 下的轻量表（采样/特征/真实场景清单/自检 JSON）。

用法：E:\\Ana\\python.exe tools\\prepare_deploy.py   （在项目根目录运行）
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT.parent / "s1-grain-size-gpr-site"

COPY_DIRS = [
    "scripts",
    "tools",
    "test",
    "report",        # 项目报告.md / 技术路线.drawio / 技术路线.md
    "results",       # 9 幅图 + metrics.json + figures_index.md
]
COPY_FILES = [
    "index.html",
    "README.md",
    "requirements.txt",
    # 轻量数据表（真实数据轨痕迹与自检数字，总量 < 1 MB）
    "data/samples.csv",
    "data/dataset.csv",
    "data/test_predictions.csv",
    "data/real_scenes_manifest.json",
    "data/sim_summary.json",
    "data/features_check.json",
    "data/dem_meta.json",
    "data/dem_status.json",
]
EXCLUDE_IN_RESULTS = {"*.npz"}      # results 下若有中间栅格不带走


def main() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True)

    for item in COPY_DIRS:
        src = ROOT / item
        dst = SITE / item
        if not src.exists():
            print(f"[warn] 缺少目录 {item}，跳过（先跑一遍管线 01~06）")
            continue
        shutil.copytree(src, dst,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                      *EXCLUDE_IN_RESULTS))
    for item in COPY_FILES:
        src = ROOT / item
        if not src.exists():
            print(f"[warn] 缺少文件 {item}，跳过")
            continue
        dst = SITE / item
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    (SITE / ".nojekyll").write_text("", encoding="utf-8")   # 纯静态站，跳过 Jekyll

    n = sum(1 for _ in SITE.rglob("*") if _.is_file())
    size = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file()) / 1e6
    print(f"部署拷贝完成：{SITE}")
    print(f"  文件 {n} 个，共 {size:.1f} MB")


if __name__ == "__main__":
    main()
