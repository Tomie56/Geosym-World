# GeoSym World Geometry 与 GeoClaw

## 项目定位

本仓库实现两个相互衔接的层次：

1. `geosym_world_geometry`：基于 SymPy 精确坐标的 2D/3D 几何状态、构造、事实闭包、验证、合成和渲染引擎；
2. `geoclaw`：建立在几何引擎上的 Agent 数据结构与交互环境。

GeoClaw 的核心新意不是给几何题增加计算器、缩放或辅助线，而是：

> 将多模态几何解题表示为可观察、可交互、可由 solver 逐步验证的工具调用轨迹，并系统评估模型的工具选择、参数正确性、无效动作、证据强度和失败恢复能力。

模型不再只看静态图并一次性输出答案，而是在统一环境中执行：

```text
observe -> select tool -> ground arguments -> execute
        -> solver feedback -> recover or continue -> final
```

## 主要能力

### 精确几何引擎

- 可重放 construction trace；
- SymPy 支持的精确 2D 和小规模 3D 状态；
- 点、线、线段、射线、圆、弧、多边形、区域、平面和多面体；
- 长度、角度、面积、体积、平行、垂直、共线、相似等事实闭包；
- theorem-lite 几何关系检查；
- SVG 与 OpenCV PNG 双渲染；
- 参数化合成和旧数据适配。

### GeoClaw Agent 环境

六类统一 JSON action：

| Action | 作用 |
|---|---|
| `zoom` | 检查小标签、密集交点、阴影边界等局部区域。 |
| `calculate` | 执行受限符号计算，并记录依赖的几何事实。 |
| `measure` | 区分图像像素近似测量和 solver 精确测量。 |
| `construct` | 中点、垂线、平行线、圆、交点等事务式辅助构造。 |
| `verify` | 对 typed claim 返回 true、false 或 unknown。 |
| `final` | 提交最终答案并终止 episode。 |

构造失败时自动回滚，不污染 geometry state、FactStore 或 action trace。成功构造会重新渲染当前图像，形成可审计的几何 sketchpad。

## 核心数据结构

### `GeoClawTask`

任务包含：

- 问题、图像和 formal trace；
- 精确 target 和 answer；
- 可接受的必要工具集合；
- 可接受中间事实；
- oracle/reference trace 与参考成本；
- 关键视觉区域 `VisualRegion`；
- 难度结构 `DifficultyProfile`；
- 动作预算和数值容差。

难度不只用单一标签，而是显式记录：

- `level`：高中、大学、奥赛等；
- `category`：visual、numeric、construction、mixed；
- `visual_complexity`；
- `construction_depth`；
- `theorem_depth`；
- `symbolic_growth`。

### `ToolFeedback`

每步返回：

- `status`：是否合法、可执行；
- `utility`：progress、redundant、harmful、unknown；
- `result`；
- solver evidence；
- `evidence_strength`；
- 结构化错误；
- 剩余预算和 step image。

证据等级包括：

```text
symbolic_proved
exact_coordinate
solver_derived
multi_instance_verified
numeric_verified
unknown
counterexample_found
invalid
```

### Private Audit

模型可见 observation 与离线审计数据严格隔离。视觉测量的 oracle residual、计算表达式是否缺少几何支持等信息只写入 private trajectory audit，不暴露给 Agent。

## 计算表达式溯源

`calculate` 支持提交 `fact_ids`：

```json
{
  "action": "calculate",
  "arguments": {
    "expression": "4**2 + 3**2",
    "fact_ids": ["fact_00010", "fact_00011"]
  }
}
```

环境记录：

- 表达式是否可解析；
- 引用的 fact 是否存在；
- 是否有几何事实支持；
- supporting facts；
- 最终答案正确但计算缺少几何依据的 masking 风险。

因此评测的不是“计算器有没有算对”，而是模型是否从正确的几何关系得到正确表达式。

## 视觉区域与测量

任务可以标注关键归一化区域：

```json
{
  "id": "dense_intersection",
  "region": [0.2, 0.1, 0.8, 0.7],
  "purpose": "dense_intersection",
  "objects": ["K", "M", "circle_O"],
  "required": true
}
```

`zoom` 返回与标注区域的 IoU、覆盖率和 required-region coverage。

`measure.mode=image` 的长度/距离通过实际 `RenderScene` 和 viewport transform 获取像素值；`measure.mode=oracle` 返回 exact FactStore 值。离线 scorer 可比较 visual–solver residual，研究模型是否被非按比例绘制的图像误导。

## 指标

已有指标包括：

- Final Accuracy；
- Tool Selection Precision / Recall / F1；
- Schema / Argument Validity；
- Invalid Action Rate；
- Verified Step Ratio；
- Useful Action Rate；
- Tool Efficiency；
- Recovery Rate 和 Feedback Gain；
- Expression Validity；
- Geometry–Expression Support Rate；
- Calculation Masking Rate；
- Zoom Grounding IoU；
- Required Region Coverage；
- Visual Measurement Rate；
- Visual–Solver Residual；
- Measurement Reliance Error；
- Construction Validity / Utility；
- Evidence Strength 分布。

指标不依赖唯一标准证明轨迹，而依据必要工具集合、参数合法性、事实闭包增量和失败恢复事件。

## 快速开始

```bash
cd /data/Geosym127K
export PYTHONPATH=/data/Geosym127K
```

列出 pilot：

```bash
python -m geosym_world_geometry.tools.run_geoclaw_pilot --list-tasks
```

运行单题：

```bash
python -m geosym_world_geometry.tools.run_geoclaw_pilot \
  --task construct_midpoint \
  --output-dir geosym_world_geometry/outputs/geoclaw_demo
```

工具消融：

```bash
python -m geosym_world_geometry.tools.run_geoclaw_pilot \
  --task construct_midpoint \
  --allowed-tools construct,verify
```

批量运行并生成 manifest：

```bash
python -m geosym_world_geometry.tools.run_geoclaw_batch \
  --baseline oracle-final \
  --output-dir geosym_world_geometry/outputs/geoclaw_batch
```

完整说明见 `docs/GEOCLAW_QUICK_START.md`。

## 测试

```bash
cd /data/Geosym127K
PYTHONPATH=/data/Geosym127K pytest -q -p no:cacheprovider geosym_world_geometry/tests
```

## 当前边界

- 主实验仍使用 oracle formal geometry state；
- Parser 当前负责保守对象引用解析，不是 OCR 或 learned scene parser；
- typed verifier 不是完整自然语言证明器；
- 当前 pilot 主要用于验证数据结构与工具闭环；
- 尚未绑定具体模型 API、训练代码和前端；
- 单实例或多实例数值验证必须与严格符号证明区分。
