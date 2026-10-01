# Direct agent reconstruction — 20260930 v1

本目录包含 20 张题图的手工数学模型、模型驱动重绘、原图/重绘逐图对照和总览。模型只根据输入 JSON 的 `parser_input.image` 与 `parser_input.question` 建立；本次实验没有读取正式 Parser、formal state、ground truth、旧 redraw 或历史解析结果。

- `01_*/model.json`：逐图数学模型，含坐标、对象、关系、题面数量、目标、检查和不确定性。
- `01_*/redraw.png`：由 `render_instructions` 绘制。
- `01_*/comparison.png`：原图与重绘并排。
- `contact.png`：20 张并排总览（两列，每行一张样本）。
- `quality_review.json`：逐图人工质量判断。
- `model_index.json`、`inputs.json`：本次样本索引与输入图像哈希。

`render_models.py` 和 `build_models.py` 位于工作区 `/data/Geosym127K/tmp/direct_agent_20260930_v1/`，仅为一次性实验脚本。
