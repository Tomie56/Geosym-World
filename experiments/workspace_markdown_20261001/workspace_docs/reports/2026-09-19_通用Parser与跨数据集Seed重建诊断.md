# 通用 Parser 与跨数据集 Seed 重建诊断

## 目标

本轮不针对 Geometry3K 增加规则，而是验证以下通用链路：

```text
公开图片 + 可选题目文本
  -> VLM 对象与关系假设
  -> OpenCV 局部视觉证据
  -> ParseGraph
  -> admission / conflict audit
  -> exact reconstruction 或 structural SeedScene
  -> 可审阅重绘
```

精确重建、结构种子和 OpenCV 调试覆盖图必须分开统计。SeedScene 只表示一个满足已提取拓扑的非精确几何实例，不表示恢复了原图坐标，也不表示 Parser 假设已成为事实。

## 通用改动

- `SeedRealizer.realize()` 继续只消费 admitted graph。
- `SeedRealizer.realize_proposal()` 可消费 semantic ParseGraph，但只接受显式对象清单或已绑定对象；禁止把关系字符串拆出的点名、匿名 OpenCV 点或未解决冲突直接用于重建。
- proposal seed 保留 `input_status=proposal`、provenance、满足与未满足约束；不写入持久 facts。
- `partial` seed 允许作为审阅图输出，但不得计入 exact reconstruction 或完整约束恢复。
- 批量审阅目录新增独立 `seed.png`，选择顺序为 exact redraw、structural seed、scaffold、失败诊断。
- 正式 Parser 代码中未加入 Geometry3K、MATH-Vision 或 PGDP 的数据集特例。

## 新增公开数据覆盖

### MATH-Vision

已准备 6 个真实竞赛几何样本，覆盖矩形与交点、内切圆、半圆阴影面积、角平分线、延长边复合图和立方体路径。

- 输入：`Geosym-World/artifacts/parser/sources/mathvision_public_sample_v1/parser-input.jsonl`
- 图片：`Geosym-World/artifacts/parser/sources/mathvision_public_sample_v1/images/`
- 当前无模型运行：`Geosym-World/artifacts/parser/evaluations/mathvision_public_sample_v2/`

无 VLM 的 scripted backend 在严格门控下 6/6 保守拒绝，没有输出伪 SeedScene。

### PGDP 官方 demo

已从 PGDP 官方仓库准备 6 张图，覆盖多边形面积、圆与切线、平行截线及代数标注等。

- catalog：`Geosym-World/artifacts/parser/sources/pgdp_demo_v1/catalog.jsonl`
- 公开输入：`Geosym-World/artifacts/parser/sources/pgdp_demo_v1/prepared/parser-input.jsonl`
- 原始运行：`Geosym-World/artifacts/parser/evaluations/pgdp_demo_v1/parsed/pgdp_demo_scripted_v1/`
- 紧凑审阅：`Geosym-World/artifacts/parser/evaluations/pgdp_demo_v1/review/samples/`

OpenCV/结构阶段 6/6 正常结束，但由于当前环境没有 OCR/VLM 语义输出，6/6 在 semantic extraction 阶段保守拒绝，0 张伪重建。复杂彩色图上的 OpenCV 候选较密，只能作为绑定证据，不能直接当作语义结构。

## Geometry3K 历史输出复核

使用当前 compiler、admission 与严格 proposal seed 门控，离线重放了既有 50 条 image-only 和 50 条 image+question VLM 输出：

- image-only：48 条有历史输出，0 条可生成受支持的 named seed；2 条缺历史输出。
- image+question：48 条有历史输出，0 条可生成受支持的 named seed；2 条缺历史输出。
- 主要原因：历史 VLM 输出没有当前契约要求的显式对象 inventory 或可靠 visual binding；点名多为关系字符串拆解所得。

产物位于 `Geosym-World/artifacts/parser/evaluations/geometry3k_50_seed_replay_v3_20260919/`。

这不是当前新提示词的在线准确率，而是证明旧输出不满足新契约，不能继续作为成功重建结果。

## 当前阻塞与下一次实验

项目现已通过 Git 忽略的 `Geosym-World/.env` 配置 OpenAI-compatible VLM endpoint，并使用 `qwen/qwen3.7-flash` 作为默认视觉模型。Parser CLI 会显式加载项目 `.env`，而库 import 不会读取凭据；显式进程环境变量优先于文件配置。

新 provider 已成功完成 1 张 PGDP staged smoke，产物位于 `Geosym-World/artifacts/parser/evaluations/pgdp_demo_v1/parsed/pgdp_demo_qnaigc_smoke_v1/`。该样本同时暴露出 VLM 在无点名 image-only 图上虚构 A–F 的问题。通用 proposal seed 门控已收紧：image-origin 点名必须具有视觉绑定，未绑定点名只有在 text-origin 关系明确引用时才能用于 SeedScene。修复后离线重放该语义图返回 `unsupported/no_named_points`，不再输出错误 seed。

恢复 provider 后，先运行 12 张小样本，不做大规模 benchmark：

1. 6 张 MATH-Vision 使用 image+question；
2. 6 张 PGDP demo 使用 image-only；
3. 每条保存 relation pass、semantic ParseGraph、admission、SeedScene 和 compact review；
4. 人工核对对象召回、边拓扑、显式关系、约束满足率和错误重建率；
5. 只有跨两类数据均成立的修复才进入正式代码。

## 测试

Parser/SeedScene 邻近测试共 `107 passed`。尚未把新 VLM 样本计入成功率，因为当前没有完成新的模型调用。
