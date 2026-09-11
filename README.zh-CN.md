# Geosym-World

[English README](README.md)

Geosym-World 是面向几何数据、几何工具和几何智能体的开源基础库。项目把几何图形、约束、构造轨迹、推导事实、渲染结果和数据集记录放进统一的可验证状态模型中。

## 项目定位

Geosym-World 主要提供：

- 统一的 2D/3D Geometry State Contract；
- 基于 SymPy 精确值与数值值并存的几何内核；
- 构造、求解、事实闭包和轨迹验证；
- 从同一个场景生成 SVG/PNG 的渲染管线；
- 带退化检查、依赖审计、难度分层和失败分类的合成引擎；
- 图片与题目文本到结构化几何假设的 Parser v0；
- 数据集适配、组件 Benchmark 和诊断工具。

Parser 输出默认是带 `confidence` 和 `provenance` 的 hypothesis，必须经过 grounding、admission 和验证后才能成为持久事实。

## 目录

```text
src/geosym_world/schema/       Geometry State Contract
src/geosym_world/kernel/       2D/3D 几何内核
src/geosym_world/construction/ 构造执行
src/geosym_world/solver/       推导与事实闭包
src/geosym_world/verifier/     状态和轨迹验证
src/geosym_world/rendering/    SVG/PNG 渲染与标注
src/geosym_world/synthesis/    合成、增强和数据导出
src/geosym_world/parser/       结构化与视觉 Parser
src/geosym_world/benchmark/    组件评测与诊断
tests/                         回归测试
docs/                          设计和接口文档
artifacts/                     本地产物，默认不纳入 Git
```

## 环境与测试

项目统一使用 `geosym` Conda 环境：

```bash
cd /data/Geosym127K/Geosym-World
/data/miniconda3/envs/geosym/bin/python -m pytest -q
```

安装开发依赖：

```bash
/data/miniconda3/envs/geosym/bin/python -m pip install -e '.[test]'
```

## 合成数据

列出可用图形族：

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.synthesize_records --list-families
```

生成小规模审阅集：

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.synthesize_records \
  --output artifacts/generated/review/records.jsonl \
  --max-records 6 --per-family 1 --no-mutation \
  --format png --workers 4 --scene-payload metadata --seed 23
```

生成严格计数的数据集：

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.synthesize_records \
  --preset large-500 --format png --workers 4 --require-exact-count
```

标注级别可以独立控制。`--annotation-level points` 只绘制点名；`--annotation-level givens` 绘制公开数值和关系标注。`--relation-markers` 统一控制直角、平行、等长和角平分等符号。

## Parser

Parser 的输入只包括图片和题目文本，不读取 formal state 或 scene metadata。合成数据的真值只用于监督、评测和诊断。

典型流程为：

```text
图片/文本 -> OCR 与点名 -> primitive -> topology
          -> relation hypothesis -> grounding -> admission
```

Parser 结果、重建状态和评测产物写入 `artifacts/parser/`。生成图片、记录和 manifest 写入 `artifacts/generated/`。这些运行产物默认被 Git 忽略。

## 开发原则

- 几何对象使用稳定 ID，点名顺序和对象引用必须可重放；
- 观测、hypothesis、admitted fact 和 derived fact 分层保存；
- 构造和推导结果保留 provenance 及依赖事实；
- 合成任务优先由目标和依赖图驱动，避免无关操作；
- Geosym-World 不依赖 GeoClaw；GeoClaw 作为上层 Agent 系统依赖本项目。

真实开放世界题图解析、广泛 3D 解析、完整定理证明和 CAD 级渲染仍属于逐步建设中的能力。
