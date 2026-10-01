# 工作区旧资产最终删除清单

## 状态

状态：`executed`

完成时间：`2026-08-09T17:20:52+08:00`

以下资产已完成用途判定并按用户精确授权删除。源码和 Git 历史证据位于 `/data/Geosym127K/archive/pre_refactor_20260809/`；大体积数据保存文件级 inventory，没有复制数据本体。删除总量约 1.2 GB。

## 已删除目录

| 路径 | 当前体积 | 文件数 | 判定 | 恢复依据 |
|---|---:|---:|---|---|
| `/data/Geosym127K/datas` | 1.1 GB | 6711 | 旧 Parser replay、smoke、debug 和派生数据 | `archive/pre_refactor_20260809/artifact_inventory.tsv` |
| `/data/Geosym127K/output` | 95 MB | 2036 | 旧 3D demo、batch、smoke 和渲染输出 | inventory；可由新管线重建 |
| `/data/Geosym127K/results_n1000_easy` | 0 | 0 | 空旧结果目录 | 无需恢复 |
| `/data/Geosym127K/temp` | 1.3 MB | 61 | 一次性脚本、缓存和调试产物 | 旧源码 archive 与 inventory |
| `/data/Geosym127K/configs` | 4 KB | 1 | 根旧 3D 测试配置 | 两仓配置已独立维护 |

## 已删除根文件

```text
/data/Geosym127K/.git
/data/Geosym127K/.pytest_cache
/data/Geosym127K/__pycache__
/data/Geosym127K/MathGeo.log
/data/Geosym127K/README.md
/data/Geosym127K/README_zh.md
/data/Geosym127K/IMPLEMENTATION_SUMMARY.md
/data/Geosym127K/requirements.txt
/data/Geosym127K/setup.sh
```

## 明确保留

```text
/data/Geosym127K/AGENTS.md
/data/Geosym127K/Geosym-World
/data/Geosym127K/GeoClaw
/data/Geosym127K/docs
/data/Geosym127K/archive
/data/Geosym127K/.agents
/data/Geosym127K/.codex
```

`archive/` 不建立 Git 仓库；`docs/` 保存内部复现材料和中文用户报告；`.agents/` 与 `.codex/` 保存工作区 Agent 配置。

## 已执行的仓内删除

- GeoClaw `paper/`、fair-v2、deliverables、E1/E2/E3、复制几何源码和旧实验材料；
- Geosym-World/GeoClaw 旧 outputs、缓存、运行结果和空结果目录；
- 含退化样本的预验证 500 条合成产物；
- 最终大批量验证图片在 checksum 与审计摘要写入 `validation_evidence.json` 后删除。

## 已收到的精确授权

用户已于 2026-08-09 明确回复：

```text
批准删除 deletion_manifest.md 中列出的全部待删除路径，并允许在 Geosym-World 和 GeoClaw 分别创建重构 commit。
```

## 删除后验收

1. 宿主文件系统执行 `find /data/Geosym127K -type d -name .git -prune -print` 只返回两个仓库；
2. 工作区根不再存在旧单仓 README、requirements、setup、日志或缓存；
3. 两仓 `artifacts/` 只保留各自 README；
4. 两仓无 `__pycache__`、`.pytest_cache`、`*.egg-info` 和旧结果目录；
5. 再运行双仓完整测试、`pip check`、console CLI 和跨仓 import；
6. 两仓分别提交重构，fresh clone 完整测试和最小运行通过。

上述六项均已完成。宿主文件系统确认根 `.git` 不存在，活动 Git 根仅为 `Geosym-World/.git` 和 `GeoClaw/.git`。
