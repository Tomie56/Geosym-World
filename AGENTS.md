# Geosym-World 开发规范

- 开始工作前同时读取 `/data/Geosym127K/AGENTS.md`；本文件补充 Geosym-World 仓库规则。
- 本仓库只负责几何状态契约、2D/3D 内核、构造、求解、验证、渲染、合成、Parser、数据适配和组件 Benchmark。
- 不得导入 `geoclaw`，不得复制 GeoClaw 的 Agent、模型、策略、trajectory 或实验实现。
- 正式 Python 包位于 `src/geosym_world/`。新增模块必须归入明确职责目录，不在仓库根新增平铺运行时代码。
- Parser 输出默认是带 confidence/provenance 的 hypothesis；未经 admission 和验证不得写入持久 facts。
- 本地数据、生成、Parser、Benchmark 和渲染产物写入 `artifacts/`，不得提交；测试只保留最小 fixture。
- 使用 `/data/miniconda3/envs/geosym/bin/python`。测试命令为 `python -m pytest -q`，批量生成前先做结构审计。
