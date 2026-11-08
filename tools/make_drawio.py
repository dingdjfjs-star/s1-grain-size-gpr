# -*- coding: utf-8 -*-
"""
make_drawio.py —— 生成 draw.io 可直接打开的技术路线图（report/技术路线.drawio）
================================================================================
程序化拼装 mxGraphModel XML：五层泳道（数据/预处理/特征/建模/输出），
节点矩形 + 正交连线；同时输出同内容 Markdown 版 report/技术路线.md。
重新生成：E:\\Ana\\python.exe tools\\make_drawio.py
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"

# 泳道与节点布局：五列，每列一个浅色背景带 + 标题 + 若干节点
BANDS = [
    ("数据层", "#dae8fc", "#6c8ebf", [
        ("d1", "Sentinel-1 L1 GRD 时序\n（模拟 15 景 IW，VV+VH，12 d 重访）\n含热噪声/量化 DN + 定标/噪声 LUT"),
        ("d2", "等效 SLC 复数据\n（Pauli 基 k1/k2/k3，持久纹理承载相干）"),
        ("d3", "真实数据轨\nCDSE OData 场景清单 46 景（匿名查询）\nSRTM 30m DEM（opentopodata 4096 点）"),
        ("d4", "沉积物粒度采样集 150 点\n（砂/粉砂/黏土占比、D50；观测噪声 σ=4 μm）"),
    ]),
    ("预处理层", "#d5e8d4", "#82b366", [
        ("p1", "① 轨道校正（元数据校验/占位，\n真实轨换 POD 精轨插值）"),
        ("p2", "② 热噪声去除\n（功率域扣 NESZ 跨轨 LUT）"),
        ("p3", "③ 辐射定标\n（DN → σ⁰VV / σ⁰VH）"),
        ("p4", "④ Lee Sigma 斑点滤波（自实现）\n7×7 sigma 范围选点；ENL 4.3→8+"),
        ("p5", "⑤ 简化地形校正\n（真实 DEM 局部入射角面积改正）"),
    ]),
    ("特征层", "#ffe6cc", "#d79b00", [
        ("f1", "极化特征\nσ⁰VV、σ⁰VH、VV/VH 比\nPauli 三分量 |k1|²|k2|²|k3|²"),
        ("f2", "时序相干系数 γ\nVV 通道，105 景对 5×5 窗估计\n高滩 0.80 / 低滩 0.47"),
        ("f3", "几何/地形特征\n入射角 31.5°–41°、TWI(D8)"),
        ("f4", "特征集 9 维 × 150 样本\n（3×3 窗均值匹配）"),
    ]),
    ("建模层", "#e1d5e7", "#9673a6", [
        ("m0", "数据划分 80/20（种子固定）\n+ 训练集 5 折交叉验证"),
        ("m1", "高斯过程回归 GPR（主模型）\nC·RBF + White 核\n输出：预测均值 μ + 后验 std"),
        ("m2", "XGBoost（对照）\n500 树 hist；特征重要性(gain)"),
    ]),
    ("输出层", "#f8cecc", "#b85450", [
        ("o1", "精度评估\nR² / RMSE（test + 5 折 CV）"),
        ("o2", "空间反演图（GPR/XGB）\n不确定性分布图（std）\n散点拟合图、相干图、Pauli RGB"),
        ("o3", "项目报告（六章）+ draw.io 技术路线\n中科院考研复试作品集材料"),
    ]),
]
# 主要连线（id, id）
EDGES = [
    ("d1", "p1"), ("p1", "p2"), ("p2", "p3"), ("p3", "p4"), ("p4", "p5"),
    ("d3", "p5"),                                    # 真实 DEM -> 地形校正
    ("p5", "f1"), ("d2", "f1"), ("d2", "f2"), ("d3", "f3"),
    ("f1", "f4"), ("f2", "f4"), ("f3", "f4"),
    ("d4", "f4"),                                    # 采样点匹配
    ("f4", "m0"), ("m0", "m1"), ("m0", "m2"),
    ("m1", "o1"), ("m2", "o1"), ("m1", "o2"), ("m2", "o2"), ("o2", "o3"), ("o1", "o3"),
]

BAND_X0, BAND_W, GAP = 40, 300, 22
HEAD_H = 40


def build_xml() -> str:
    """拼装 mxfile XML（明文，draw.io 可直接打开/导入）。"""
    cells = []
    for bi, (title, fill, stroke, nodes) in enumerate(BANDS):
        x = BAND_X0 + bi * (BAND_W + GAP)
        # 泳道背景带
        cells.append(
            f'<mxCell id="band{bi}" value="{title}" style="rounded=0;html=1;'
            f'fillColor=#f5f5f5;strokeColor=#666666;verticalAlign=top;fontSize=15;'
            f'fontStyle=1;opacity=60;" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="40" width="{BAND_W}" height="880" as="geometry"/>'
            f'</mxCell>')
        for ni, (nid, label) in enumerate(nodes):
            y = 100 + ni * 130
            cells.append(
                f'<mxCell id="{nid}" value="{label}" style="rounded=1;html=1;'
                f'whiteSpace=wrap;fillColor={fill};strokeColor={stroke};fontSize=11;'
                f'align=center;" vertex="1" parent="1">'
                f'<mxGeometry x="{x + 30}" y="{y}" width="{BAND_W - 60}" '
                f'height="110" as="geometry"/></mxCell>')
    for ei, (s, t) in enumerate(EDGES):
        cells.append(
            f'<mxCell id="e{ei}" style="edgeStyle=orthogonalEdgeStyle;rounded=0;'
            f'html=1;strokeWidth=1.5;" edge="1" parent="1" source="{s}" target="{t}">'
            f'<mxGeometry relative="1" as="geometry"/></mxCell>')
    body = "".join(cells)
    return (
        '<mxfile host="app.diagrams.net" type="device">'
        '<diagram id="s1-grain-techroute" name="技术路线">'
        '<mxGraphModel dx="1600" dy="900" grid="1" gridSize="10" page="1" '
        'pageWidth="1700" pageHeight="960" math="0" shadow="0">'
        f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root>'
        '</mxGraphModel></diagram></mxfile>')


MD = """# 技术路线（与 技术路线.drawio 同内容，Markdown 备份版）

**数据层** → **预处理层** → **特征层** → **建模层** → **输出层**

1. **数据层**
   - Sentinel-1 L1 GRD 时序（模拟 15 景 IW GRD，VV+VH，12 d 重访；DN + 定标/噪声 LUT）
   - 等效 SLC 复数据（Pauli 基 k1/k2/k3，持久纹理承载时序相干）
   - 真实数据轨：CDSE OData 匿名场景清单（46 景 IW GRD）+ SRTM 30 m DEM（4096 点）
   - 沉积物粒度采样集 150 点（砂/粉砂/黏土占比、D50，含观测噪声）
2. **预处理层**（对模拟 L1 包真实执行；接口对真实 SAFE 兼容）
   ① 轨道校正（占位/元数据校验） → ② 热噪声去除（NESZ 跨轨 LUT） →
   ③ 辐射定标（σ⁰VV/VH） → ④ Lee Sigma 斑点滤波（自实现，ENL 4.3→8+） →
   ⑤ 简化地形校正（真实 DEM 局部入射角面积改正）
3. **特征层**
   - 极化：σ⁰VV、σ⁰VH、VV/VH 比、Pauli 三分量 |k1|²、|k2|²、|k3|²
   - 时序：相干系数 γ（VV 通道，105 景对、5×5 窗估计、时间平均）
   - 地形/几何：入射角（31.5°–41°）、TWI（真实 SRTM + D8 汇流）
   - 汇成 9 维特征集 × 150 样本（3×3 窗均值匹配）
4. **建模层**：80/20 固定种子划分 + 5 折 CV；GPR（C·RBF+White，μ 与 std）∥ XGBoost（对照）
5. **输出层**：R²/RMSE（test+CV）；空间反演图、不确定性分布图、散点拟合图、
   相干图、Pauli RGB；六章项目报告 + draw.io 技术路线（作品集材料）
"""


def main() -> None:
    REPORT.mkdir(exist_ok=True)
    (REPORT / "技术路线.drawio").write_text(build_xml(), encoding="utf-8")
    (REPORT / "技术路线.md").write_text(MD, encoding="utf-8")
    print("== drawio/技术路线 生成完成：report/技术路线.drawio + 技术路线.md ==")


if __name__ == "__main__":
    main()
