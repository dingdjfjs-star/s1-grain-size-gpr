# -*- coding: utf-8 -*-
"""
fetch_dem.py —— 真实数据轨②：opentopodata 抓取真实 SRTM 30m 高程
==================================================================
在研究区 bbox 上按约 700 m 间隔（64x64 采样格网）抓取真实 SRTM 高程，
存 data/dem_real.npy + data/dem_real.tif + data/dem_meta.json，
供 03_extract_features.py 计算 TWI 与 02 地形校正使用。

* opentopodata 限流：每请求 <=100 点、1 req/s —— 带断点缓存（dem_cache.csv）
  与逐请求重试；总耗时约 1 分钟。
* 抓取失败不抛异常：写 dem_status.json 后正常退出（03 自动回退合成 DEM）。
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import DATA, LON0, LAT0, LON1, LAT1  # noqa: E402

API = "https://api.opentopodata.org/v1/srtm30m"
NG = 64                     # 采样格网 64x64（间隔约 700 m，沿海平原地形足够）
PER_REQ = 50                # 每请求点数（上限 100，留余量）
CACHE = DATA / "dem_cache.csv"


def build_points():
    """生成 64x64 采样点（纬度向从北到南，保证 npy 行序与网格一致）。"""
    lats = np.linspace(LAT1, LAT0, NG)
    lons = np.linspace(LON0, LON1, NG)
    pts = [(round(la, 5), round(lo, 5)) for la in lats for lo in lons]
    return pts


def load_cache() -> dict:
    """读断点缓存 {(lat,lon): elev}。"""
    cache = {}
    if CACHE.exists():
        with open(CACHE, "r", encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if len(row) == 3 and row[0] != "lat":
                    cache[(float(row[0]), float(row[1]))] = float(row[2])
    return cache


def fetch_batch(pts) -> list:
    """抓一批点（opentopodata locations 用竖线分隔 lat,lon），带 3 次重试。"""
    q = "|".join(f"{la},{lo}" for la, lo in pts)
    url = f"{API}?locations={q}"
    last = None
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "s1-grain-size-gpr/1.0"})
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            out = []
            for p, r in zip(pts, data["results"]):
                z = r["elevation"]
                out.append(z if z is not None else 0.0)   # 海上空值记 0 m
            return out
        except Exception as exc:                            # noqa: BLE001
            last = exc
            time.sleep(2.5)
    raise RuntimeError(f"opentopodata 连续 3 次失败: {last}")


def main() -> None:
    pts = build_points()
    cache = load_cache()
    todo = [p for p in pts if p not in cache]
    print(f"[fetch_dem] 共 {len(pts)} 点，缓存命中 {len(pts) - len(todo)}，待抓 {len(todo)}")

    try:
        for i in range(0, len(todo), PER_REQ):
            batch = todo[i:i + PER_REQ]
            zs = fetch_batch(batch)
            with open(CACHE, "a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                if not CACHE.exists() or CACHE.stat().st_size == 0:
                    w.writerow(["lat", "lon", "elev_m"])
                for (la, lo), z in zip(batch, zs):
                    w.writerow([la, lo, z])
            cache.update(dict(zip(batch, zs)))
            time.sleep(1.15)          # 限流：<=1 req/s
            print(f"  ... {min(i + PER_REQ, len(todo))}/{len(todo)}")

        grid = np.array([cache[p] for p in pts], dtype=float).reshape(NG, NG)
        np.save(DATA / "dem_real.npy", grid)

        # 存 GeoTIFF（真实数据痕迹，供 GIS 软件直接打开）
        try:
            import rasterio
            from rasterio.transform import from_bounds
            profile = dict(driver="GTiff", height=NG, width=NG, count=1,
                           dtype="float32", crs="EPSG:4326")
            transform = from_bounds(LON0, LAT0, LON1, LAT1, NG, NG)
            with rasterio.open(DATA / "dem_real.tif", "w", **profile,
                               transform=transform) as dst:
                dst.write(grid.astype("float32"), 1)
        except Exception as exc:                            # noqa: BLE001
            print(f"  GeoTIFF 写出跳过: {exc}")

        meta = {"source": "opentopodata srtm30m", "n_grid": NG,
                "posting_deg": (LON1 - LON0) / (NG - 1),
                "note": "约 700 m 间隔抽样后上采样使用；海洋区 SRTM 记 0 m",
                "elev_min_m": float(grid.min()), "elev_max_m": float(grid.max())}
        (DATA / "dem_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        (DATA / "dem_status.json").write_text(
            json.dumps({"ok": True, "n_points": len(pts)}, ensure_ascii=False),
            encoding="utf-8")
        print(f"[fetch_dem] OK elev {grid.min():.1f}~{grid.max():.1f} m -> dem_real.npy/tif")
    except Exception as exc:                                    # noqa: BLE001
        (DATA / "dem_status.json").write_text(
            json.dumps({"ok": False, "reason": str(exc)}, ensure_ascii=False),
            encoding="utf-8")
        print(f"[fetch_dem] FAILED: {exc}（03 将回退合成 DEM）")


if __name__ == "__main__":
    main()
