# Parser v1 Contracts

Parser v1 将图像证据、语义候选和可实现几何分成三个阶段：

```text
ParseGraph -> ObservationGraph + ClauseGraph -> ConstraintState
                                      |              |
                                      +-> admission  +-> realization
```

`ObservationGraph` 保存像素、文本和局部视觉证据。`ClauseGraph` 保存对象、关系、数量、区域和目标的候选语义；每个 Clause 都带有 `confidence`、`provenance`、`dependencies` 和 `status`。Parser 产生的候选默认是 `proposed`。

`ConstraintState` 只描述当前约束能支持的结论。欠约束状态必须公开 `free_parameters`；没有明确坐标时不能自动生成 realization。`realize_exact_points` 只对 Parser 明确提供的 `exact_coords` 创建确定性 witness。

使用 admission 时，调用方必须提供 verifier 对每个候选的显式 `accepted: true` 报告。依赖必须形成闭包，`rejected` 和 `derived` Clause 不能重新变成 `admitted`。所有交换对象都带 schema version 和 checksum。

模型输出使用 `clause_graph_from_model_payload` 接入。payload 必须包含 `schema_version`、`clauses`、`target_ids` 和 `metadata`，且所有 Clause 必须是 `proposed`；模型不能自行声明 admission 结果。模型调用、verifier 和 GeometryState 之间应保持这个边界。

```python
from geosym_world.parser.v1 import compile_parse_graph_state

observations, clauses, constraints = compile_parse_graph_state(parse_graph)
```

也可以从模型的结构化 JSON 直接创建候选图：

```python
from geosym_world.parser import clause_graph_from_model_payload

clauses = clause_graph_from_model_payload(
    payload,
    graph_id="clauses_problem_1",
    source_id="problem_1",
)
```

只有完成 verifier admission 的 Clause 才能投影到现有 `FactStore`：

```python
from geosym_world.parser import admitted_clauses_to_facts

facts = admitted_clauses_to_facts(clauses, source_id="problem_1")
```

投影会把 `admitted` 记录为 `SOURCE_ASSERTION`，把 `derived` 记录为 `ENTAILED_FACT`，并保留 Clause provenance；`proposed` 不会进入 facts。

GeometryState bridge 当前保守支持 exact point、line、segment、ray、circle、arc、polygon、simple region、带已注册 hole 的 region，以及明确 component 的 composite region。公共入口为 `materialize_clause_graph`；旧的 `materialize_exact_point_state` 仅作为兼容别名。关系层支持 incidence、intersection、midpoint、垂足、角平分线、切线、弦、平行、垂直、等长、等角、比例、相似和全等；数量 clause 会投影为带 exact value 的 FactStore fact，target 会保留 mode、expectation 和表达式。物化按依赖排序执行，不依赖输入 clause 的排列顺序，并为 relation 保留 `parser_clause_id`、`provenance` 和 `dependencies`。每个不满足引用或表达式条件的已 admission Clause 都会出现在 `unsupported_clause_ids` 中；bridge 不会静默删除候选，也不会用像素近似替代数学坐标。

这一层不执行完整几何定理推导，也不把视觉近似当作数学事实；正式 solver、grounding 和 GeometryState admission 应在后续阶段接入。
