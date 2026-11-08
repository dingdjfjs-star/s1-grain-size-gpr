# -*- coding: utf-8 -*-
"""
fetch_real_catalog.py —— 真实数据轨①：CDSE 真实 S1 GRD 场景清单
=================================================================
匿名查询 Copernicus Data Space Ecosystem（CDSE）目录，列出研究区
（黄河三角洲-莱州湾 bbox）内与模拟时序同长度的真实 Sentinel-1 GRD L1
产品清单（产品名/日期/轨道/极化/大小），落盘 data/real_scenes_manifest.json。

* 通道：主用 OData `OData.CSC.Intersects` 空间过滤（2026-11-04 实测匿名可用，
  tools/probe_catalog.py 为通道探针留档）；备用 STAC GET（当前返回 400）。
* 只做元数据查询，不下载 GRD 包（下载需 CDSE 账号 token，接入方法见 README）。
* 任何失败都不抛异常：写 status 字段后正常退出，保证不阻塞主轨管线。
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import DATA, LON0, LAT0, LON1, LAT1, N_SCENES, REPEAT_DAYS, T0_STR, scene_dates  # noqa: E402

ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
STAC = "https://catalogue.dataspace.copernicus.eu/stac/search"
POLY = (f"POLYGON(({LON0} {LAT0},{LON1} {LAT0},{LON1} {LAT1},"
        f"{LON0} {LAT1},{LON0} {LAT0}))")
END = (__import__("datetime").date.fromisoformat(T0_STR)
       + __import__("datetime").timedelta(days=N_SCENES * REPEAT_DAYS)).isoformat()


def _get_json(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "s1-grain-size-gpr/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def query_odata() -> list:
    """OData 主通道：按时间窗 + 空间相交过滤，翻页取全，返回产品 dict 列表。"""
    flt = ("Collection/Name eq 'SENTINEL-1' and contains(Name,'GRD') and "
           f"ContentDate/Start ge {T0_STR}T00:00:00.000Z and "
           f"ContentDate/Start lt {END}T00:00:00.000Z and "
           f"OData.CSC.Intersects(area=geography'SRID=4326;{POLY}')")
    items, skip = [], 0
    while True:
        q = urllib.parse.urlencode({"$filter": flt, "$orderby": "ContentDate/Start",
                                    "$top": 50, "$skip": skip,
                                    "$expand": "Attributes"})
        page = _get_json(f"{ODATA}?{q}")
        vals = page.get("value", [])
        items.extend(vals)
        if len(vals) < 50:
            break
        skip += 50
    return items


def parse_odata(items: list) -> list:
    """压缩 OData 产品记录为关键字段清单。"""
    out = []
    for it in items:
        attrs = {a.get("Name"): a.get("Value") for a in it.get("Attributes", [])}
        out.append({
            "id": it.get("Id"),
            "product_name": it.get("Name"),
            "datetime": (it.get("ContentDate") or {}).get("Start"),
            "orbit_direction": attrs.get("orbitDirection"),
            "relative_orbit": attrs.get("relativeOrbitNumber"),
            "polarization_mode": "VV+VH" if "_GRDH_" in it.get("Name", "") else "other",
            "size_mb": round((it.get("ContentLength") or 0) / 1e6, 1),
            "download_api": (f"{ODATA}({it.get('Id')})/$value  —— 需 CDSE 账号 token"),
        })
    out.sort(key=lambda d: d.get("datetime") or "")
    return out


def query_stac() -> list:
    """备用通道：STAC GET（当前 400，保留以便日后接口恢复）。"""
    q = urllib.parse.urlencode({
        "bbox": f"{LON0},{LAT0},{LON1},{LAT1}",
        "datetime": f"{T0_STR}T00:00:00Z/{END}T00:00:00Z",
        "collections": "sentinel-1-l1-grd", "limit": 200})
    raw = _get_json(f"{STAC}?{q}")
    out = []
    for ft in raw.get("features", []):
        p = ft.get("properties", {})
        out.append({"id": ft.get("id"), "product_name": p.get("title") or ft.get("id"),
                    "datetime": p.get("datetime"), "orbit_direction": p.get("sat:orbit_state"),
                    "polarization_mode": "VV+VH", "size_mb": None})
    return out


def main() -> None:
    out = {"bbox": [LON0, LAT0, LON1, LAT1], "datetime": f"{T0_STR} ~ {END}",
           "note": "仅元数据清单；下载需注册 CDSE 账号并携带 token（见 README 真实数据接入）",
           "sim_time_series_dates": scene_dates()}
    scenes, channel, reason = [], None, None
    for ch, fn in (("odata", query_odata), ("stac-get", query_stac)):
        try:
            scenes = parse_odata(fn()) if ch == "odata" else fn()
            if scenes:
                channel = ch
                break
            reason = f"{ch}: 查询成功但 0 条"
        except Exception as exc:                                # noqa: BLE001
            reason = f"{ch}: {type(exc).__name__}: {exc}"
    out["status"] = "ok" if channel else "failed"
    out["channel"] = channel
    out["reason"] = reason
    out["n_scenes"] = len(scenes)
    out["scenes"] = scenes

    path = DATA / "real_scenes_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[fetch_real_catalog] status={out['status']} channel={channel} "
          f"n_scenes={len(scenes)} -> {path}")


if __name__ == "__main__":
    main()
