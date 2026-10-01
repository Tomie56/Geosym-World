# GeoSym Hybrid Real Geometry Parser

> Implementation note (2026-09-10): the maintained runtime is in
> `Geosym-World/src/geosym_world`. Use [`Geosym-World/docs/PARSER_PIPELINE.md`](../Geosym-World/docs/PARSER_PIPELINE.md)
> for executable commands and artifact paths. Any historical command below
> that references `temp/` is design history, not a maintained entry point.

## Core Principle

真实图片转 GeoSym JSON 不是 image-to-coordinate，而是 image-assisted reconstruction。

图片像素坐标只作为视觉证据，不进入 GeoSym 的数学坐标系统。GeoSym state 里的坐标仍然必须由 canonical builder / solver 在真实数学坐标系中生成。

## Runtime Input Boundary

Parser 的正式运行时输入严格是 `ParserInput`：

```json
{
  "schema_version": "1.0",
  "id": "sample-id",
  "image": "/path/to/diagram.png",
  "question": "Find ..."
}
```

图片和题干之外的字段都属于数据集私有信息或评测监督，不能传给
`ScriptObservationExtractor`、VLM、grounding 或 canonical builder。原始数据行可以
同时保存 `answer`、`solution`、formal state、scene metadata 等字段，但正式 Parser
入口必须先投影成上述四个字段；这些私有字段只允许被独立 evaluator 读取。

因此，Parser 不以答案一致性作为输入提示，也不使用 formal state、scene metadata
或 renderer metadata 反推结构。Parser 输出始终是带 `confidence` 和 `provenance`
的 proposed hypothesis，只有独立 grounding、admission 和验证通过后才能进入持久
事实或 canonical state。

## Updated Design: Script + VLM Cooperative Extraction

Geometry3K 这类真实题不能依赖 VLM 一步生成 GeoSym JSON。更靠谱的结构是 candidate-first：

1. 脚本先提取视觉候选集合：
   - 线段、圆/弧、文本区域、端点、交点、线圆接触点；
   - 每个候选都有稳定 ID，例如 `vl_*`、`vc_*`、`vt_*`、`vx_*`；
   - 这些像素候选只作为 evidence，不作为数学坐标。

2. VLM 只做脚本难做的语义任务：
   - 把标签绑定到候选对象；
   - 判断题干里的数值条件绑定到哪条边、哪个角、哪个圆；
   - 补充脚本漏掉但图题明显存在的关系假设；
   - 输出它认为当前 GeoSym builder 能执行的 `build_instructions`；
   - 对当前样本给出 `engine_assessment.expected_mode`。

3. 规则层验收 VLM 输出：
   - `SemanticOutputValidator` 先检查 JSON schema、字段类型、未知 relation/action；
   - `RelationGrounder` 检查所有 `visual_ids` 是否来自脚本候选集合；
   - `BuildInstructionPlanner` 检查指令是否属于当前 GeoSym builder 能力；
   - VLM 没有证据绑定的 claim 不直接进入 canonical builder。

4. Solver/planner 负责重建 canonical state：
   - 对 `coordinate_buildable` 样本，用支持的约束生成 SymPy 真实数学坐标；
   - 对 `theorem_only` 样本，保留 formal graph，不强行生成错误草图；
   - 对 `unsupported/failed` 样本，输出 review image 和 stage diagnosis，作为后续 detector/schema 扩展依据。

这个设计把失败拆成可统计的层级：

- script detector 没提到候选；
- VLM 没有识别/绑定语义；
- VLM 输出 JSON 不合格；
- claim 无法 grounding 到 visual ids；
- 当前 GeoSym builder 不支持；
- canonical build/verification 失败；
- image alignment 或 answer consistency 失败。

## Pipeline

1. Script visual extraction
   - 输入原始图片；
   - 用 OpenCV 提取 line / circle / text-region candidates；
   - 额外生成 model-free topology candidates：线线交点、线圆接触、线段端点；
   - 输出 `visual_state/*.json`。

2. Script observations
   - 将低层 visual candidates 摘要成 VLM 更容易使用的结构化观察；
   - 输出 dominant lines、近似平行/垂直、相近长度候选、交点候选、线圆接触候选；
   - 从题干中抽取点名、线段名、数字、简单长度 claim 和 target hint；
   - 从题干中生成 `semantic_candidates`：relation candidates 和 target candidates；
   - 基于当前 builder capability 生成 `engine_build_suggestions`，作为 VLM 可参考但必须确认的 builder 指令草案；
   - 这些观察仍然只是 evidence，不是 GeoSym 数学事实；
   - 输出 `script_observations/*.json`。

3. Evidence graph
   - 将 visual candidates、text-derived semantic candidates、engine build suggestions 和当前 GeoSym builder capability 合并成统一 `evidence_graph`；
   - `evidence_graph` 是给 VLM 的硬输入 contract，不是数学状态；
   - 每个 relation candidate 会标注 `buildability`：`coordinate_constraint`、`convertible_coordinate_constraint`、`theorem_only` 或 `unsupported`；
   - 每个 relation candidate 同时会标注 `builder_mapping`，显式说明这个语义关系能否转成当前 GeoSym builder action，例如 `length -> add_point/add_segment/set_length`，或 `right_angle -> set_perpendicular`；
   - `builder_contract.relation_mapping_contract` 会全局列出当前 GeoSym 2D builder 支持的 relation-to-action 映射，VLM 不需要从自然语言 prompt 猜测 engine 能力；
   - 每个 build suggestion 会标注是否属于当前 builder 支持的 action；
   - `build_tasks` 会按 `candidate_id` 聚合需要 stage 2 覆盖的 required actions，例如 `sc_len_0 -> add_point/add_segment/set_length`；
   - 输出 `evidence_graphs/*.json`。

4. VLM semantic proposal
   - VLM 读取原图、带 ID 的 visual overlay、题干和 visual candidates；
   - 同时读取 `script_observations`、`evidence_graph` 和当前 GeoSym builder capability；
   - 输出 label bindings、关系候选、target；
   - 必须确认、修正或拒绝脚本给出的 `semantic_candidates`；
   - 必须保留或修正 relation 的 `buildability` / `builder_mapping`，让后续 stage 2 明确知道哪些关系能被当前 GeoSym builder 构造，哪些只能进入 theorem/formal graph；
   - 必须用 `candidate_decisions` 记录每个已检查候选是 `confirm`、`correct`、`reject` 还是 `uncertain`；
   - 只有确认可 grounding 且当前 builder 支持的内容，才复制/改写成 `build_instructions`；
   - 同时输出 `build_instructions`，把可 build 的几何内容转换成 GeoSym 数据引擎指令；
   - 同时输出 `engine_assessment.expected_mode`，自评该样本对当前 GeoSym 引擎是 `coordinate_buildable`、`theorem_only`、`unsupported` 还是 `failed`；
   - VLM 必须尽量把关系绑定到 `vl_* / vc_* / vt_* / vx_* / vcl_*`；
   - 输出 `semantic_relations/*.json`。

   语义提取支持三种 backend：

- `scripted`：不调用 VLM，直接把 `evidence_graph` 中的脚本候选转换成 relations、candidate decisions 和 build instructions；
- `openrouter`：调用 OpenRouter VLM；
- `mock`：故意不抽取几何，用于测试失败路径。

   `scripted` 是离线下界和诊断基线，不是最终 parser。它用于回答三个问题：

- 如果 `scripted` 都不能通过，问题在脚本候选、build task 或 GeoSym adapter；
- 如果 `scripted` 能通过但 `openrouter` 不能通过，问题在 VLM 没遵守 candidate/build contract；
- 如果两者都能 build 但 image alignment 不过，问题在视觉 grounding、canonical sketch 和原图对齐。

   OpenRouter VLM extraction 支持两种模式：

- `single`：一次调用同时输出 relations 和 build instructions；
- `staged`：两次调用，第一轮只抽 geometry relations / target / bindings，第二轮基于第一轮关系和 GeoSym builder capability 转成 build instructions。

   推荐真实数据优先用 `staged`，因为它更符合“先理解几何语义，再判断 GeoSym 能 build 什么”的流程。
   staged 模式会额外保存 `vlm_passes/<sample>.relation_pass.json`、`instruction_task_packets/<sample>.json`、`vlm_passes/<sample>.instruction_pass.json` 和 `vlm_transcripts/<sample>.json`。
   OpenRouter staged 模式还会保存 `*.raw.json`，即 VLM 原始 pass 输出；不带 `.raw` 的文件是经过 stage-level schema normalization 后的 pass 输出。
   在 relation pass 中，pipeline 会先保留 raw VLM 输出，再运行 `RelationPassAdapter` 做受控映射：当 VLM 已经输出了同 type/object/value 的 relation，但漏写 `candidate_id`、`buildability` 或 `builder_mapping` 时，adapter 会从 evidence graph 的 script candidate 补齐这些字段。adapter 不会新增 candidate decision，也不会发明新 relation，因此 raw contract 仍然能评价 VLM 是否真正审阅了 candidate。

5. Grounding and instruction planning
   - `SemanticOutputValidator` 首先检查 VLM JSON 是否满足最小 schema；
   - schema warning 会写入 report，而不是让后续 pipeline 崩溃；
   - `CandidateUsageAnalyzer` 检查 VLM 是否确认/采用了脚本候选；
   - 如果 VLM 引用未知 `candidate_id/from_candidate`，写入 warning；
   - `InstructionTaskPacketBuilder` 根据 relation pass 的 `candidate_decisions`、relations 和 target refs，把 `evidence_graph.build_tasks` 收敛成 stage 2 必须覆盖的 `required_tasks`；
   - `required_tasks` 会携带 `relation_buildability` 和 `builder_mapping`；如果 VLM 用 `decision=correct` 改写了候选关系，task packet 会用 corrected relation 重新生成 builder mapping；
   - 检查 VLM 引用的 visual ids 是否存在；
   - grounded=true 且 visual ids 有效的关系进入 accepted；
   - theorem-level 或无法完全视觉绑定的关系进入 uncertain；
   - 引用不存在 visual id 的关系进入 rejected；
   - `BuildInstructionPlanner` 检查 VLM 提出的 builder 指令是否符合当前 GeoSym 2D builder 能力；
   - planner 同时检查 `build_instructions[*].visual_ids` 是否存在于脚本视觉候选中；
   - `InstructionRepairer` 会对已被 VLM 确认/修正/引用的 script candidates 做有限修复：如果对应 `build_tasks` 缺少 required actions，就从脚本 `build_suggestions` 补齐 `script_build_suggestion` 指令；
   - repair 不会凭空确认候选，也不会把 theorem-level relation 伪装成 coordinate constraint；
   - repair 结果会写入 `instruction_repair` 和 metrics，用于区分 raw VLM instruction conversion 与 framework-repaired conversion；
   - `EngineAssessmentValidator` 比较 VLM 自评和 GeoSym formalizer/planner 的实际判定，不一致时写入 warning；
   - `ExtractionContractAuditor` 综合检查 VLM 是否遵守 script-led contract，包括是否采用脚本候选、是否覆盖可 build relations、是否输出了被 planner reject 的指令、engine assessment 是否一致；
   - builder 能力由 `geosym/parsing/hybrid/builder_capabilities.py` 统一定义；
   - 当前支持的 builder 指令为 `add_point`、`add_segment`、`set_length`、`set_parallel`、`set_perpendicular`、`set_target`；
   - 如果 VLM 没有输出 `build_instructions`，但 grounded relations 已经足够表达当前 builder 支持的约束，planner 会从 relations 合成指令；
   - 合成指令会标记为 `instruction_source=synthesized_from_relations`；
   - 圆、弧、弦、相似、全等、角追踪、面积分解等 theorem-level 内容必须保留为 relation/formal graph，不能强行转成坐标 builder 指令；
   - 输出 `grounded_relations/*.json`。

6. Formalization and buildability
   - `coordinate_buildable`：只包含当前 canonical builder 可处理的长度/平行/垂直等低阶关系；
   - `theorem_only`：圆、弦、弧、角、相似、全等等需要 theorem solver 的关系；
   - `unsupported`：有关系但当前 schema/solver 不支持；
   - `failed`：没有可用关系；
   - 输出 `formal_graphs/*.json`。

7. Canonical build and verification
   - 只有 instruction planner 判定 `buildable=true` 的样本进入 `RealDataAdapter -> canonical builder -> GeoSymVerifier`；
   - 其他样本不生成 fake canonical sketch，只生成原图 + extraction summary 的 review image；
   - 硬验证包括 schema/geometry/quantity、image alignment、answer consistency。

8. Optional LLM judge
   - judge 是重要 correctness signal，但不是硬验证；
   - judge 可以在 evaluator 阶段读取独立保存的参考答案，辅助判断 canonical/formal reconstruction 是否和原图、题干一致；
   - judge 失败不会中断样本，会返回 `needs_review`。

## Outputs

```text
/data/Geosym127K/Geosym-World/artifacts/datasets/parsed/<dataset>/
  visual_state/
  visual_overlays/
  script_observations/
  evidence_graphs/
  semantic_relations/
  semantic_raw/
  semantic_repaired/
  vlm_passes/
  vlm_transcripts/
  instruction_task_packets/
  grounded_relations/
  formal_graphs/
  problem_states/
  sketches/
  review_images/
  reports/
```

## Commands

Model-free scripted baseline:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.parse_real_data_hybrid_v0 \
  --dataset-name geometry3k \
  --records-jsonl /data/Geosym127K/Geosym-World/artifacts/datasets/raw/geometry3k/metadata/records.jsonl \
  --backend scripted \
  --judge-backend none \
  --limit 100 \
  --raw-root /data/Geosym127K/Geosym-World/artifacts/datasets/raw \
  --parsed-root /data/Geosym127K/Geosym-World/artifacts/datasets/parsed
```

Hybrid parse Geometry3K with staged VLM:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.parse_real_data_hybrid_v0 \
  --dataset-name geometry3k \
  --records-jsonl /data/Geosym127K/Geosym-World/artifacts/datasets/raw/geometry3k/metadata/records.jsonl \
  --backend openrouter \
  --model qwen/qwen3.7-plus \
  --extraction-mode staged \
  --judge-backend none \
  --limit 100 \
  --raw-root /data/Geosym127K/Geosym-World/artifacts/datasets/raw \
  --parsed-root /data/Geosym127K/Geosym-World/artifacts/datasets/parsed
```

With optional judge:

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.parse_real_data_hybrid_v0 \
  --dataset-name geometry3k \
  --records-jsonl /data/Geosym127K/Geosym-World/artifacts/datasets/raw/geometry3k/metadata/records.jsonl \
  --backend openrouter \
  --model qwen/qwen3.7-plus \
  --extraction-mode staged \
  --judge-backend openrouter \
  --judge-model qwen/qwen3-vl-32b-instruct \
  --limit 100 \
  --raw-root /data/Geosym127K/Geosym-World/artifacts/datasets/raw \
  --parsed-root /data/Geosym127K/Geosym-World/artifacts/datasets/parsed
```

Summarize:

```bash
/data/miniconda3/envs/geosym/bin/python temp/summarize_parse_reports.py \
  --dataset-name geometry3k \
  --parsed-root /data/Geosym127K/Geosym-World/artifacts/datasets/parsed
```

`summarize_parse_reports.py` 会输出 `primary_stages`，用于判断 100 条样本主要卡在哪一层：

- `visual_extraction`
- `semantic_extraction`
- `relation_grounding`
- `theorem_solver_missing`
- `formalization`
- `instruction_planning`
- `canonical_build`
- `answer_consistency`
- `image_alignment`
- `verification`
- `accepted`

它还会统计：

- `semantic_schema_warning_samples`
- `semantic_schema_warnings`
- `script_parallel_hint_samples`
- `script_perpendicular_hint_samples`
- `script_text_hint_samples`
- `script_relation_candidate_samples`
- `script_target_candidate_samples`
- `script_build_suggestion_samples`
- `evidence_graph_available`
- `evidence_relation_candidate_samples`
- `evidence_build_suggestion_samples`
- `instruction_task_packet_available`
- `instruction_task_packet_required_task_samples`
- `instruction_task_packet_required_actions_total`
- `instruction_task_packet_instruction_requirements_total`
- `instruction_task_packet_corrected_relation_task_samples`
- `instruction_task_packet_corrected_relation_tasks_total`
- `instruction_task_packet_builder_mapped_task_samples`
- `instruction_task_packet_builder_mapped_tasks_total`
- `staged_instruction_task_packet_available`
- `staged_instruction_task_packet_required_task_samples`
- `raw_task_packet_missing_task_samples`
- `raw_stage_missing_relation_buildability_samples`
- `raw_stage_missing_relation_builder_mapping_samples`
- `final_task_packet_missing_task_samples`
- `task_packet_repaired_coverage_samples`
- `candidate_confirmed_samples`
- `candidate_reviewed_samples`
- `candidate_unreviewed_samples`
- `candidate_rejected_samples`
- `candidate_corrected_samples`
- `candidate_decision_conflict_samples`
- `candidate_unknown_ref_samples`
- `instruction_coverage_available`
- `instruction_missing_constraint_samples`
- `staged_relation_pass_available`
- `staged_instruction_pass_available`
- `staged_relation_pass_nonempty`
- `staged_instruction_pass_nonempty`
- `staged_relation_pass_schema_warning_samples`
- `staged_instruction_pass_schema_warning_samples`
- `staged_relation_pass_leaked_instruction_samples`
- `staged_instruction_pass_extra_relation_samples`
- `staged_instruction_pass_extra_decision_samples`
- `instruction_rejected_samples`
- `engine_assessment_mismatch`
- `contract_pass`
- `contract_warn`
- `contract_fail`
- `contract_error_samples`
- `contract_warning_samples`
- `contract_missing_build_task_samples`

Triage reports and get representative inspect commands:

```bash
/data/miniconda3/envs/geosym/bin/python temp/triage_hybrid_reports.py \
  --dataset-name geometry3k \
  --parsed-root /data/Geosym127K/Geosym-World/artifacts/datasets/parsed \
  --max-per-group 3
```

`triage_hybrid_reports.py` 会按 primary stage 和 issue code 分组，并输出 `priority_samples`。每个 priority sample 都带有可直接运行的 `inspect_command`，用于继续定位候选、decision、build task 和 instruction 的断点。

Analyze staged parser failure matrix:

```bash
/data/miniconda3/envs/geosym/bin/python temp/analyze_staged_failures.py \
  --dataset-name geometry3k \
  --parsed-root /data/Geosym127K/Geosym-World/artifacts/datasets/parsed \
  --max-samples 5
```

`analyze_staged_failures.py` 专门回答 staged VLM parser 卡在哪层：

- script candidates 是否存在；
- relation pass 是否可用、是否有 schema warning；
- VLM 是否输出 candidate decisions；
- VLM 是否保留 relation 的 buildability；
- VLM 是否保留 relation-to-builder mapping；
- 是否确认/修正了候选；
- task packet 是否有 required tasks；
- raw instruction pass 是否覆盖 required tasks；
- repair 后 final 是否覆盖；
- instruction planner/canonical build 是否继续失败。

输出包括 `funnel_counts/funnel_rates`、`blocker_counts/blocker_rates`、每个 blocker 的代表样本和 `recommended_next_fix`。
matrix 会读取 report 中的 `semantic_backend` 和 `extraction_mode`。只有 `semantic_backend=openrouter` 且 `extraction_mode=staged` 时，缺少 relation pass 才会被归为 OpenRouter/staged 调用失败；`mock/scripted` baseline 会按后续 candidate decision/task packet 等断点归因。
新增 blocker：

- `missing_relation_buildability`：stage-1 已输出 relation，但没有保留 `buildability`；
- `missing_relation_builder_mapping`：stage-1 已输出 relation 和 buildability，但没有保留 relation-to-builder action mapping。

Analyze staged VLM behavior as a two-stage capability table:

```bash
/data/miniconda3/envs/geosym/bin/python temp/analyze_staged_vlm_behavior.py \
  --dataset-name geometry3k \
  --parsed-root /data/Geosym127K/Geosym-World/artifacts/datasets/parsed \
  --max-samples 5
```

这个脚本和 failure matrix 不同：它不是只找第一个 blocker，而是聚合 stage-1 和 stage-2 行为指标：

- stage 1：candidate decision、confirm/correct/reject/unreviewed；
- stage 2：required task、instruction requirement、raw template adherence、corrected relation task；
- repair：repair 是否补齐 final template；
- outcome：buildable、verification、merged decision。

它适合写实验表：例如 `raw_template_fully_matched` 可以直接衡量 VLM 是否理解并复制了 GeoSym builder template，`final_template_fully_matched - raw_template_fully_matched` 则是框架 repair 的可验证收益。
`corrected_relation_task_present` 和 `corrected_relation_tasks` 用来衡量 VLM 是否真的把脚本候选修正成新的几何关系，并且该修正是否进入了 stage-2 GeoSym build instruction contract。`builder_mapped_task_present` 和 `builder_mapped_tasks` 用来衡量 stage-2 required task 是否显式携带了“关系 -> GeoSym builder action”的映射。

Compare raw VLM semantic output against repaired/final semantic output:

```bash
/data/miniconda3/envs/geosym/bin/python temp/compare_semantic_versions.py \
  --dataset-name geometry3k \
  --parsed-root /data/Geosym127K/Geosym-World/artifacts/datasets/parsed \
  --max-samples 20
```

这个脚本统计：

- raw relation / instruction / candidate-decision coverage；
- final instruction coverage；
- repair used rate；
- raw 没有指令但 final 有指令的样本比例；
- raw-to-final instruction delta 最大的样本。

Replay saved VLM semantic JSON without calling the model:

```bash
/data/miniconda3/envs/geosym/bin/python temp/replay_hybrid_semantic.py \
  --semantic-json /data/Geosym127K/Geosym-World/artifacts/datasets/parsed/<dataset>/semantic_raw/<sample>.json \
  --visual-json /data/Geosym127K/Geosym-World/artifacts/datasets/parsed/<dataset>/visual_state/<sample>.json \
  --script-observations-json /data/Geosym127K/Geosym-World/artifacts/datasets/parsed/<dataset>/script_observations/<sample>.json \
  --evidence-graph-json /data/Geosym127K/Geosym-World/artifacts/datasets/parsed/<dataset>/evidence_graphs/<sample>.json \
  --enable-repair
```

`semantic_raw/` 保存 VLM/后端原始语义输出，`semantic_repaired/` 保存经过 `InstructionRepairer` 后的语义，`semantic_relations/` 是最终进入 grounding/formalization 的兼容路径。对比 raw 和 repaired 可以直接看到 VLM 自己输出了多少 builder 指令，以及框架补了多少。

如果省略 `--visual-json`，带 `visual_ids` 的 VLM 指令会因为无法验证视觉 ID 而被拒绝；这对纯 schema 调试有用，但不适合评估 grounding 质量。

如果提供 `--script-observations-json`，replay 会额外输出 `candidate_usage` 和 `instruction_coverage`，用于离线判断 VLM 是否采用了脚本候选，以及可 build relations 是否被转成 GeoSym constraints。

如果同时提供 `--evidence-graph-json`，replay 还会输出 `extraction_contract`，用于判断 VLM 输出是否遵守脚本候选和 GeoSym builder capability 的硬约束；如果只提供 `--script-observations-json`，replay 会临时重建一个 evidence graph。

Replay saved staged VLM passes without calling the model:

```bash
/data/miniconda3/envs/geosym/bin/python temp/replay_staged_outputs.py \
  --relation-pass-json /data/Geosym127K/Geosym-World/artifacts/datasets/parsed/<dataset>/vlm_passes/<sample>.relation_pass.json \
  --instruction-pass-json /data/Geosym127K/Geosym-World/artifacts/datasets/parsed/<dataset>/vlm_passes/<sample>.instruction_pass.json \
  --visual-json /data/Geosym127K/Geosym-World/artifacts/datasets/parsed/<dataset>/visual_state/<sample>.json \
  --script-observations-json /data/Geosym127K/Geosym-World/artifacts/datasets/parsed/<dataset>/script_observations/<sample>.json \
  --evidence-graph-json /data/Geosym127K/Geosym-World/artifacts/datasets/parsed/<dataset>/evidence_graphs/<sample>.json \
  --enable-repair
```

`replay_staged_outputs.py` 会额外输出 `stage_contract.raw/final`，其中 `stage1.num_missing_relation_buildability_refs` 和 `stage1.num_missing_relation_builder_mapping_refs` 可用于离线判断 stage-1 VLM 是否理解并保留了 GeoSym builder contract。

这个脚本复盘 staged pipeline 的核心断点：

- relation pass 是否有 relations 和 candidate decisions；
- `instruction_task_packet.required_tasks` 是否正确生成；
- raw instruction pass 是否覆盖 required actions；
- `InstructionRepairer` 是否需要介入；
- final semantic 是否覆盖 task packet；
- instruction planner 是否能 build；
- 当前 blocker 和 recommendation。

它适合分析“VLM 看懂了几何关系，但不会把关系翻译成 GeoSym builder 指令”的样本。raw coverage 和 final coverage 会同时输出，因此可以区分 VLM 原生指令转换能力与框架 repair 后的可验证上界。

Prepare a stage-2 prompt review bundle without calling the model:

```bash
/data/miniconda3/envs/geosym/bin/python temp/prepare_staged_prompt_review.py \
  --relation-pass-json /data/Geosym127K/Geosym-World/artifacts/datasets/parsed/<dataset>/vlm_passes/<sample>.relation_pass.json \
  --visual-json /data/Geosym127K/Geosym-World/artifacts/datasets/parsed/<dataset>/visual_state/<sample>.json \
  --script-observations-json /data/Geosym127K/Geosym-World/artifacts/datasets/parsed/<dataset>/script_observations/<sample>.json \
  --evidence-graph-json /data/Geosym127K/Geosym-World/artifacts/datasets/parsed/<dataset>/evidence_graphs/<sample>.json \
  --output-dir /data/Geosym127K/Geosym-World/artifacts/datasets/parsed/<dataset>/prompt_reviews/<sample>
```

输出文件包括：

- `stage2_instruction_prompt.txt`：真实会给 stage-2 VLM 的 prompt；
- `stage1_relation_prompt.txt`：真实会给 stage-1 VLM 的 prompt；
- `builder_contract.json`：当前 GeoSym builder 能力；
- `relation_mapping_contract.json`：全局 relation-to-builder action 表；
- `instruction_task_packet.json`：根据 relation pass 生成的任务包；
- `response_template.json`：GeoSym builder 输出草稿；
- `template_coverage.json`：检查 response template 是否覆盖全部 instruction requirements；
- `review_bundle.json`：路径、统计和建议。

`review_bundle.summary` 会检查：

- relation prompt 是否包含 `relation_mapping_contract`；
- instruction prompt 是否包含 `relation_mapping_contract` 和 candidate/task 级 `builder_mapping`；
- task packet 是否携带全局 relation mapping；
- response template 是否覆盖全部 instruction requirements。

这个工具用于在调用 VLM 前先审查 prompt/task packet/template 是否清楚。如果 `template_coverage` 自己不通过，或 prompt 缺少 relation mapping contract，应先修框架，不要调 prompt。

Inspect one parsed sample as candidate-to-instruction chain:

```bash
/data/miniconda3/envs/geosym/bin/python temp/inspect_hybrid_sample.py \
  --report-json /data/Geosym127K/Geosym-World/artifacts/datasets/parsed/<dataset>/reports/<sample>.json
```

也可以按 dataset/sample id 自动定位：

```bash
/data/miniconda3/envs/geosym/bin/python temp/inspect_hybrid_sample.py \
  --dataset-name geometry3k \
  --sample-id <sample> \
  --parsed-root /data/Geosym127K/Geosym-World/artifacts/datasets/parsed
```

这个 inspect 脚本不调用 VLM，只读取已有 `report/evidence_graph/semantic/formal`，输出每个 candidate 的：

- 原始 script candidate；
- VLM candidate decision；
- relation references；
- instruction task packet required/deferred tasks；
- build task；
- accepted/rejected/synthesized instructions；
- contract issues。

Offline smoke test for parser logic:

```bash
/data/miniconda3/envs/geosym/bin/python temp/smoke_hybrid_parser.py
```

该 smoke 覆盖：

- VLM 显式 `build_instructions` 可 build；
- VLM 只给 relations 时从 grounded relations 合成指令；
- build instruction 引用未知 visual id 时被 reject；
- theorem-only 样本不进入 canonical builder；
- semantic 空输出被归因到 `semantic_extraction`。
- malformed VLM JSON 被 `SemanticOutputValidator` 归一化并统计 warning。
- mock end-to-end pipeline 会生成 `script_observations/*.json`，并把 script hints 写入 report metrics。
- mock end-to-end pipeline 会生成 `evidence_graphs/*.json`，并把 evidence summary 和 extraction contract 写入 report。
- staged parser 会生成 `instruction_task_packets/*.json`，并把 required/deferred task metrics 写入 report。
- staged OpenRouter parser 会生成 `vlm_transcripts/*.json`，保存 relation/instruction 两轮 prompt、task packet 和 parsed response，方便审查 VLM 到底看到了什么 builder contract。
- task packet coverage 会同时统计 raw VLM output 和 repaired/final output，用于区分 VLM-native conversion 与 framework repair。
- replay CLI 会输出 `candidate_usage`、`instruction_coverage` 和 `extraction_contract`。

## Evidence Graph Contract

`evidence_graph` 是 VLM 和脚本共同提取的中间层。它把脚本能确定或怀疑的东西组织成候选，而不是直接变成 GeoSym state：

```json
{
  "relation_candidates": [
    {
      "id": "sc_len_0",
      "type": "length",
      "object": "AB",
      "value": "3",
      "buildability": {
        "mode": "coordinate_constraint",
        "can_generate_build_instruction": true
      }
    }
  ],
  "build_suggestions": [
    {
      "action": "set_length",
      "points": ["A", "B"],
      "value": "3",
      "from_candidate": "sc_len_0",
      "supported_by_current_builder": true
    }
  ],
  "build_tasks": [
    {
      "candidate_id": "sc_len_0",
      "candidate_type": "length",
      "required_actions": ["add_point", "add_segment", "set_length"],
      "instruction_requirements": [
        {
          "requirement_id": "sc_len_0:0:add_point",
          "action": "add_point",
          "expected_instruction": {"action": "add_point", "id": "A", "from_candidate": "sc_len_0"}
        },
        {
          "requirement_id": "sc_len_0:1:add_point",
          "action": "add_point",
          "expected_instruction": {"action": "add_point", "id": "B", "from_candidate": "sc_len_0"}
        },
        {
          "requirement_id": "sc_len_0:2:add_segment",
          "action": "add_segment",
          "expected_instruction": {"action": "add_segment", "points": ["A", "B"], "from_candidate": "sc_len_0"}
        },
        {
          "requirement_id": "sc_len_0:3:set_length",
          "action": "set_length",
          "expected_instruction": {"action": "set_length", "points": ["A", "B"], "value": "3", "from_candidate": "sc_len_0"}
        }
      ],
      "must_emit_when_decision": ["confirm", "correct"]
    }
  ],
  "builder_contract": {
    "supported_actions": ["add_point", "add_segment", "set_length", "set_parallel", "set_perpendicular", "set_target"],
    "action_schema": [
      {
        "action": "set_length",
        "required_fields": ["points", "value"],
        "field_schema": {
          "points": "two point labels, e.g. ['A','B']",
          "value": "exact symbolic/numeric value from public question text",
          "from_candidate": "optional script candidate id"
        },
        "example": {"action": "set_length", "points": ["A", "B"], "value": "4", "from_candidate": "sc_len_0"}
      }
    ],
    "rule": "build_instructions must use only supported_actions and must not encode theorem-only facts as fake coordinate constraints"
  }
}
```

VLM 的 stage 1 必须基于 `relation_candidates` 做确认/修正/拒绝；stage 2 必须基于 `builder_contract` 输出指令。stage 1 输出里应该包含：

```json
{
  "candidate_decisions": [
    {
      "candidate_id": "sc_len_0",
      "decision": "confirm",
      "corrected_relation": {},
      "reason": "AB=3 is stated in the problem and matches the visible segment",
      "confidence": 0.92
    },
    {
      "candidate_id": "sc_right_triangle_0",
      "decision": "correct",
      "corrected_relation": {"type": "right_angle", "objects": ["AB", "BC"], "angle": "ABC"},
      "reason": "the text only says right triangle, but the diagram marks the right angle at B",
      "confidence": 0.81
    }
  ]
}
```

## Instruction Task Packet

`instruction_task_packet` 是 relation pass 和 instruction pass 之间的硬约束层。它不是 VLM 自由生成的 JSON，而是脚本根据以下信息构造：

- `evidence_graph.build_tasks`；
- relation pass 中的 `candidate_decisions`；
- relation pass 中 relations/target 引用的 `candidate_id/from_candidate`；
- 当前 GeoSym builder capability。

它把 build tasks 分成两类：

- `required_tasks`：stage 1 已 confirm/correct，或已经被 relation/target 使用，stage 2 必须尽量输出 required actions；
- `deferred_tasks`：脚本提出但 VLM 没确认/没使用的候选，stage 2 不应强行转成数学约束。
- `builder_contract.action_schema`：每个 supported builder action 的字段级 schema 和示例。stage 2 必须按这个 schema 写 `build_instructions`，不要发明 `line_name`、`coordinate`、`pixel_point` 等 GeoSym builder 不认识的字段。
- `builder_contract.relation_mapping_contract`：全局 relation-to-builder action 表，例如 `length -> add_point/add_segment/set_length`、`right_angle -> add_point/add_segment/set_perpendicular`。candidate 级 `builder_mapping` 必须与这个全局表一致，除非 VLM 用 `decision=correct` 修正了 relation type。
- `instruction_requirements`：实例级 expected instruction 列表。它比 `required_actions` 更严格，例如 `AB=3` 需要两个独立的 `add_point(A)` 和 `add_point(B)`，不能只用一个 `add_point` action 覆盖。
- `response_template`：由 `instruction_requirements` 自动合成的 stage-2 输出草稿。VLM 应优先复制 `response_template.build_instructions / target / engine_assessment`，只有发现和 stage-1 关系、原图或题干矛盾时才修改。

典型结构：

```json
{
  "required_tasks": [
    {
      "candidate_id": "sc_len_0",
      "candidate_type": "length",
      "decision": "confirm",
      "required_actions": ["add_point", "add_segment", "set_length"],
      "instruction_requirements": [
        {
          "requirement_id": "sc_len_0:0:add_point",
          "action": "add_point",
          "expected_instruction": {"action": "add_point", "id": "A", "from_candidate": "sc_len_0"}
        },
        {
          "requirement_id": "sc_len_0:1:add_point",
          "action": "add_point",
          "expected_instruction": {"action": "add_point", "id": "B", "from_candidate": "sc_len_0"}
        },
        {
          "requirement_id": "sc_len_0:2:add_segment",
          "action": "add_segment",
          "expected_instruction": {"action": "add_segment", "points": ["A", "B"], "from_candidate": "sc_len_0"}
        },
        {
          "requirement_id": "sc_len_0:3:set_length",
          "action": "set_length",
          "expected_instruction": {"action": "set_length", "points": ["A", "B"], "value": "3", "from_candidate": "sc_len_0"}
        }
      ],
      "suggested_instructions": [
        {"action": "add_point", "id": "A", "from_candidate": "sc_len_0"},
        {"action": "add_point", "id": "B", "from_candidate": "sc_len_0"},
        {"action": "add_segment", "points": ["A", "B"], "from_candidate": "sc_len_0"},
        {"action": "set_length", "points": ["A", "B"], "value": "3", "from_candidate": "sc_len_0"}
      ]
    },
    {
      "candidate_id": "sc_target_0",
      "candidate_type": "length",
      "decision": "confirm",
      "required_actions": ["set_target"]
    }
  ],
  "response_template": {
    "build_instructions": [
      {"action": "add_point", "id": "A", "from_candidate": "sc_len_0"},
      {"action": "add_point", "id": "B", "from_candidate": "sc_len_0"},
      {"action": "add_segment", "points": ["A", "B"], "from_candidate": "sc_len_0"},
      {"action": "set_length", "points": ["A", "B"], "value": "3", "from_candidate": "sc_len_0"},
      {"action": "set_target", "target": {"type": "length", "object": "AB", "value_symbol": "x"}, "from_candidate": "sc_target_0"}
    ],
    "target": {"type": "length", "object": "AB", "value_symbol": "x"},
    "engine_assessment": {
      "expected_mode": "coordinate_buildable",
      "reason": "required tasks are covered by current GeoSym builder instruction requirements"
    }
  },
  "metrics": {
    "num_required_tasks": 2,
    "num_deferred_tasks": 0,
    "num_required_actions": 4,
    "num_instruction_requirements": 5,
    "num_template_build_instructions": 5
  }
}
```

这层的作用是把失败继续细分：

- `evidence_graph` 没有候选：脚本/OCR/vision detector 失败；
- `instruction_task_packet.required_tasks=0`：VLM 没确认候选或 relation pass 没引用候选；
- `required_tasks>0` 但 instruction pass 指令为空：VLM 的 candidate-to-builder conversion 失败；
- `response_template` 自己不能覆盖 `instruction_requirements`：task packet 生成逻辑错误，先修框架；
- repaired/final 有指令但 raw 没有：GeoSym 框架能补齐，但 VLM 自己没有完成 build conversion。
- `response_template_adherence.raw` 缺失多：VLM 没有照着 GeoSym template 输出；
- `response_template_adherence.raw.extra_instructions` 多：VLM 在 template 之外发明了额外 builder 指令，需要检查是否越过当前 builder contract。

pipeline 会额外保存 task packet coverage：

```json
{
  "instruction_task_packet_coverage": {
    "raw": {
      "num_required_tasks": 2,
      "num_covered_tasks": 0,
      "num_missing_tasks": 2,
      "num_required_actions": 4,
      "num_covered_actions": 0,
      "num_missing_actions": 4,
      "num_instruction_requirements": 5,
      "num_covered_instruction_requirements": 0,
      "num_missing_instruction_requirements": 5
    },
    "final": {
      "num_required_tasks": 2,
      "num_covered_tasks": 2,
      "num_missing_tasks": 0,
      "num_required_actions": 4,
      "num_covered_actions": 4,
      "num_missing_actions": 0,
      "num_instruction_requirements": 5,
      "num_covered_instruction_requirements": 5,
      "num_missing_instruction_requirements": 0
    }
  }
}
```

对应 report metrics：

- `raw_task_packet_required_tasks`
- `raw_task_packet_covered_tasks`
- `raw_task_packet_missing_tasks`
- `raw_task_packet_required_actions`
- `raw_task_packet_covered_actions`
- `raw_task_packet_missing_actions`
- `raw_task_packet_instruction_requirements`
- `raw_task_packet_covered_instruction_requirements`
- `raw_task_packet_missing_instruction_requirements`
- `final_task_packet_required_tasks`
- `final_task_packet_covered_tasks`
- `final_task_packet_missing_tasks`
- `final_task_packet_required_actions`
- `final_task_packet_covered_actions`
- `final_task_packet_missing_actions`
- `final_task_packet_instruction_requirements`
- `final_task_packet_covered_instruction_requirements`
- `final_task_packet_missing_instruction_requirements`
- `raw_template_matched_instructions`
- `raw_template_missing_instructions`
- `raw_template_extra_instructions`
- `final_template_matched_instructions`
- `final_template_missing_instructions`
- `final_template_extra_instructions`

`task_packet_coverage` 回答“是否足够覆盖 required tasks”；`response_template_adherence` 回答“是否真的按 GeoSym template 生成”。两者都需要看：一个模型可能覆盖了所有 required actions，但用了不一致的字段或额外发明了 unsupported instruction。

如果 raw 缺失但 final 覆盖，report issues 会有 `task_packet.raw_missing_instruction`。这说明 VLM stage 1 关系抽取已经足够触发 build task，但 stage 2 没有自己完成 GeoSym instruction conversion，后续是由框架 repair 补齐的。

`CandidateUsageAnalyzer` 会区分：

- confirmed：候选被 relation/instruction/target 使用，或 decision 为 `confirm/correct`；
- corrected：VLM 认为脚本候选接近但需改 type/object/value；
- rejected：VLM 明确拒绝脚本候选；
- uncertain：VLM 看过但无法判断；
- unreviewed：脚本候选没有被 VLM 明确处理；
- decision conflict：VLM 声称 reject，但 relation/instruction 又使用了该 candidate。

`ExtractionContractAuditor` 会在后处理里检查这件事，并在 report 中写入：

- `extraction_contract.status`: `pass|warn|fail`
- `contract_errors`
- `contract_warnings`
- `contract_buildable_script_candidates`
- `contract_missing_build_constraints`
- `contract_required_build_tasks`
- `contract_covered_build_tasks`
- `contract_missing_build_tasks`

如果 VLM 只输出了 grounded relations，没有输出 `build_instructions`，planner 仍会尽量从 relations 合成 canonical constraints，保证可验证框架能继续跑通；但 `ExtractionContractAuditor` 会产生 `contract.vlm_instruction_missing_but_synthesized` warning。这个 warning 很重要：它表示系统最终可 build，但 VLM 的 stage 2 instruction conversion 没有完成。

如果 VLM 确认或修正了某个 candidate，但 stage 2 没有输出带 `from_candidate`/`candidate_id` 的 required actions，`ExtractionContractAuditor` 会产生 `contract.build_task_missing_instruction` warning。这用于判断“候选关系已经被 VLM 理解，但没有被正确转换成 GeoSym builder 指令”。

## Instruction Repair

`InstructionRepairer` 是 VLM staged parser 和 scripted baseline 的连接层。

它的输入是：

- VLM 输出的 `relations / candidate_decisions / target`；
- `CandidateUsageAnalyzer` 判断出的 confirmed/corrected candidates；
- `evidence_graph.build_tasks` 和 `build_suggestions`；
- 当前 GeoSym builder capability。

它只在以下条件同时成立时补指令：

1. candidate 已被 VLM confirm/correct，或已经被 VLM relation/target 引用；
2. candidate 对应的 `build_tasks.required_actions` 未被 VLM build instructions 覆盖；
3. script suggestion 的 action 在当前 builder contract 中受支持。

补出的指令会包含：

```json
{
  "action": "set_length",
  "points": ["A", "B"],
  "value": "3",
  "from_candidate": "sc_len_0",
  "repair_source": "script_build_suggestion"
}
```

report 中会记录：

- `instruction_repair.used`
- `instruction_repair.num_original_instructions`
- `instruction_repair.num_added_instructions`
- `instruction_repair.num_final_instructions`
- `semantic_versions.raw.path`
- `semantic_versions.final.path`
- `semantic_versions.repaired.path`
- `verification.metrics.instruction_repair_used`
- `verification.metrics.instruction_repair_added`
- warning issue: `instruction.script_repair_applied`

评测时建议同时报告 raw 和 repaired：

- raw VLM instruction conversion：看 VLM 自己是否能把关系转成 GeoSym 指令；
- repaired conversion：看 VLM 是否至少抽对关系，GeoSym 框架能否根据 engine contract 补齐可验证指令。

## Staged Pass Compliance

`staged` 模式要求两轮 VLM 各司其职：

- relation pass：只输出 `bindings`、`relations`、`candidate_decisions`、`target`、`engine_assessment`；
- instruction pass：只输出 `build_instructions`、`target`、`engine_assessment`、`notes`。

relation pass 结束后，pipeline 会先生成 `instruction_task_packet`，再把它传给 instruction pass。instruction pass 的首要任务不是重新理解图，而是覆盖 `instruction_task_packet.required_tasks`。

OpenRouter staged 模式还会写出 `vlm_transcripts/<sample>.json`。这个文件不保存 base64 图片，只保存：

- model/backend；
- 原图和 overlay 路径；
- relation pass prompt 和 parsed response；
- instruction pass prompt、task packet 和 parsed response。

这个 transcript 用于调 prompt 和 builder contract。如果 `instruction_task_packet.required_tasks` 是对的，但 instruction pass response 没覆盖 required actions，就可以直接打开 transcript 判断是 prompt 不清楚、VLM 忽略 contract，还是 task packet 本身不够明确。

每个 staged pass 会先经过 `SemanticOutputValidator` 的阶段模式：

- `relation_pass`：规范化 `relations / candidate_decisions / target / engine_assessment`；
- `instruction_pass`：规范化 `build_instructions / target / engine_assessment`。

这样第一阶段的小 JSON 错误不会直接导致 task packet 构造失效。pipeline 会同时保存：

- `vlm_passes/<sample>.relation_pass.raw.json`
- `vlm_passes/<sample>.relation_pass.json`
- `vlm_passes/<sample>.instruction_pass.raw.json`
- `vlm_passes/<sample>.instruction_pass.json`

如果 normalization 产生 warning，report issues 会包含 `staged.relation_pass_schema_warning` 或 `staged.instruction_pass_schema_warning`。

如果 relation pass 输出了 `build_instructions`，pipeline 会忽略这些指令，并在 report issues 中加入 `staged.relation_pass_leaked_instructions`。如果 instruction pass 又输出了 `relations` 或 `candidate_decisions`，pipeline 也会保留 relation pass 作为权威来源，并记录：

- `staged.instruction_pass_extra_relations`
- `staged_instruction_pass_extra_candidate_decisions`

这些指标用于判断 VLM 是否真的遵守“先抽几何关系，再按 GeoSym 能力转指令”的分阶段协议。

## Frontend

当前仓库不维护独立 frontend；Parser 的正式入口是 Python module CLI：

```bash
/data/miniconda3/envs/geosym/bin/python -m geosym_world.tools.parse_real_data_hybrid_v0 --help
```

Hybrid batch 默认使用 `--extraction-mode staged`，也可以切换到 `single` 做对照。
运行报告由 `run_manifest.json` 和每条记录的 `reports/*.json` 提供；分析工具
应直接读取这些正式产物，不再依赖 `temp/` 下的一次性脚本。

## Current Limitation

Geometry3K 中大量题是圆、弧、弦、相似、角追踪和定理应用题。当前 hybrid parser 会把这些样本稳定归到 `theorem_only`，避免错误地画成一维线段图。要让这些样本进入完整 GeoSym JSON，还需要后续补 theorem-level solver/schema，而不是继续放宽 canonical builder。

## VLM Output Contract

VLM 需要同时输出两类信息：

1. `relations`：尽量完整描述图像和题干中的几何语义；
2. `build_instructions`：只包含当前 GeoSym builder 能执行的构造/约束指令。

示例：

```json
{
  "relations": [
    {"type": "length", "objects": ["A", "B"], "visual_ids": ["vl_1"], "value": "4", "grounded": true}
  ],
  "build_instructions": [
    {"action": "add_point", "id": "A", "visual_ids": ["vt_1"]},
    {"action": "add_point", "id": "B", "visual_ids": ["vt_2"]},
    {"action": "add_segment", "points": ["A", "B"], "visual_ids": ["vl_1"]},
    {"action": "set_length", "points": ["A", "B"], "value": "4"},
    {"action": "set_target", "target": {"type": "length", "object": "AB", "value_symbol": "x"}}
  ],
  "engine_assessment": {"expected_mode": "coordinate_buildable", "reason": "only length/segment constraints are needed"}
}
```

这样 VLM 会显式表达“我认为 GeoSym 应该 build 什么”，而不是只给自然语言关系。脚本端仍然负责校验这些指令是否属于当前 builder 能力范围。

如果 VLM 只输出了可 grounding 的 `relations`，planner 会用这些 relations 合成 builder 指令作为兜底。但如果样本被 formalizer 判为 `theorem_only`，不会合成 canonical builder 指令，避免把圆弧/相似/角追踪题错误压扁成线段图。

## Script Candidate Contract

`script_observations` 现在包含三层候选：

```json
{
  "semantic_candidates": {
    "relation_candidates": [
      {
        "id": "sc_len_0",
        "type": "length",
        "objects": ["A", "B"],
        "object": "AB",
        "value": "3",
        "source": "problem_text_regex",
        "needs_vlm_confirmation": true
      }
    ],
    "target_candidates": [
      {
        "id": "sc_target_0",
        "type": "length",
        "object": "BC",
        "value_symbol": "x"
      }
    ]
  },
  "engine_build_suggestions": {
    "expected_mode_hint": "coordinate_buildable",
    "suggested_build_instructions": [
      {"action": "add_point", "id": "A", "from_candidate": "sc_len_0"},
      {"action": "add_segment", "points": ["A", "B"], "from_candidate": "sc_len_0"},
      {"action": "set_length", "points": ["A", "B"], "value": "3", "from_candidate": "sc_len_0"}
    ],
    "instruction_note": "VLM must confirm these suggestions against image/overlay before copying into build_instructions"
  }
}
```

脚本候选的作用是减少 VLM 漏抽关系，但它不能绕过验证。VLM 需要把候选绑定到图像/overlay ID，或明确保留为 ungrounded/theorem-level relation。最终仍由 `SemanticOutputValidator`、`RelationGrounder`、`BuildInstructionPlanner` 和 verifier 决定是否能进入 canonical state。

## Candidate Usage Audit

VLM 如果采用脚本候选，需要在输出中保留来源：

```json
{
  "relations": [
    {
      "type": "length",
      "objects": ["A", "B"],
      "value": "3",
      "candidate_id": "sc_len_0",
      "visual_ids": ["vl_1"],
      "grounded": true
    }
  ],
  "build_instructions": [
    {"action": "set_length", "points": ["A", "B"], "value": "3", "from_candidate": "sc_len_0"}
  ]
}
```

`CandidateUsageAnalyzer` 会输出：

- `num_script_candidates`
- `num_confirmed_candidates`
- `num_unconfirmed_candidates`
- `num_unknown_candidate_refs`
- `confirmed_candidate_ids`
- `unknown_candidate_refs`

这能直接衡量 VLM 是否真正利用了脚本提取的几何关系候选，而不是忽略候选重新自由发挥。

## Stage-1 Candidate Review Template

为了避免 VLM 在第一阶段自由发挥，`evidence_graph` 现在会生成两个 stage-1 合同字段：

```json
{
  "candidate_review_checklist": [
    {
      "candidate_id": "sc_len_0",
      "candidate_kind": "relation",
      "candidate_type": "length",
      "candidate_object": "AB",
      "candidate_value": "3",
      "buildability_mode": "coordinate_constraint",
      "expected_relation": {
        "type": "length",
        "objects": ["A", "B"],
        "object": "AB",
        "value": "3",
        "candidate_id": "sc_len_0",
        "grounded": false,
        "source": "text_regex"
      },
      "expected_output": {
        "candidate_id": "sc_len_0",
        "decision": "confirm",
        "corrected_relation": {
          "type": "length",
          "objects": ["A", "B"],
          "object": "AB",
          "value": "3",
          "candidate_id": "sc_len_0",
          "grounded": false,
          "source": "text_regex"
        },
        "reason": "brief evidence from image/text",
        "confidence": 0.0
      }
    }
  ],
  "relation_pass_response_template": {
    "bindings": [],
    "relations": [
      {
        "type": "length",
        "objects": ["A", "B"],
        "object": "AB",
        "value": "3",
        "candidate_id": "sc_len_0",
        "grounded": false,
        "source": "text_regex"
      }
    ],
    "candidate_decisions": [
      {
        "candidate_id": "sc_len_0",
        "decision": "confirm",
        "corrected_relation": {
          "type": "length",
          "objects": ["A", "B"],
          "object": "AB",
          "value": "3",
          "candidate_id": "sc_len_0",
          "grounded": false,
          "source": "text_regex"
        },
        "reason": "brief evidence from image/text",
        "confidence": 0.0
      }
    ],
    "target": {
      "type": "length",
      "object": "AB",
      "value_symbol": "x",
      "candidate_id": "sc_target_0",
      "from_candidate": "sc_target_0"
    },
    "engine_assessment": {
      "expected_mode": "coordinate_buildable"
    }
  }
}
```

第一阶段 VLM 的任务变成：

1. 逐项复制或修正 `relation_pass_response_template.relations / target / candidate_decisions`；
2. 对每个 `candidate_review_checklist` item 输出 `confirm|correct|reject|uncertain`；
3. `expected_relation` / `expected_target` 是 GeoSym semantic JSON 草稿，只有和图像/题干矛盾时才改；
4. 不输出 `build_instructions`，不伪造数学坐标。

这样 stage 1 和 stage 2 是对称的：

- stage 1 用 `relation_pass_response_template` 约束候选审阅；
- stage 2 用 `instruction_task_packet.response_template` 约束 GeoSym build instruction 生成。

## Instruction Coverage Audit

`InstructionCoverageAnalyzer` 检查 stage 1 / formal graph 中可 build 的关系是否被 stage 2 / planner 转成了约束。

当前覆盖的 relation 类型：

- `length`
- `parallel`
- `perpendicular`
- `right_angle`

它会输出：

- `num_required_constraints`
- `num_covered_constraints`
- `num_missing_constraints`
- `coverage`
- `missing_constraints`

如果 `num_missing_constraints > 0`，通常说明 VLM stage 1 已经抽到了可 build 的几何关系，但 stage 2 没有正确生成对应 `build_instructions`，或者 instruction 被 planner 拒绝。

## Stage Contract Audit

为了让“脚本和 VLM 共同提取”变成可验证框架，pipeline 现在额外输出 `stage_contract`：

```json
{
  "stage_contract": {
    "raw": {
      "stage1": {
        "num_candidates": 2,
        "num_reviewed_candidates": 1,
        "num_missing_candidate_decisions": 1,
        "num_expected_relation_refs": 1,
        "num_covered_relation_refs": 1,
        "num_missing_relation_refs": 0,
        "num_expected_target_refs": 1,
        "num_covered_target_refs": 0,
        "num_missing_target_refs": 1,
        "candidate_decision_coverage": 0.5
      },
      "stage2": {
        "num_required_tasks": 1,
        "num_covered_required_tasks": 0,
        "num_missing_required_tasks": 1,
        "required_task_coverage": 0.0
      }
    },
    "final": {}
  }
}
```

它和已有指标的分工：

- `candidate_usage`：统计 VLM 有没有引用、确认、拒绝脚本候选；
- `relation_template_adherence`：检查 stage 1 是否按 `relation_pass_response_template` 输出候选决策、relation drafts 和 target refs；
- `instruction_task_packet_coverage`：检查 build instructions 是否覆盖 required actions / requirements；
- `response_template_adherence`：检查 stage 2 是否贴近 `response_template`；
- `stage_contract`：把 stage 1 和 stage 2 合起来看，直接回答 VLM 是否完成了当前样本的 extraction checklist。stage 1 不只要求 `candidate_decisions`，也要求 `relations` 覆盖 relation candidate refs、`target` 覆盖 target refs。

`raw` 表示 VLM 原始输出，适合评估模型是否理解任务；`final` 表示 repair 后输出，适合评估当前 pipeline 是否能把样本推进到 canonical builder。

`temp/analyze_staged_vlm_behavior.py` 会聚合这些字段，新增：

- `raw_relation_template_fully_matched`
- `final_relation_template_fully_matched`
- `raw_relation_drafts_fully_matched`
- `final_relation_drafts_fully_matched`
- `raw_relation_target_matched`
- `final_relation_target_matched`
- `raw_relation_template_adherence`
- `final_relation_template_adherence`
- `raw_relation_draft_adherence`
- `final_relation_draft_adherence`
- `candidate_decision_contract_fully_reviewed`
- `stage1_contract_relation_refs_covered`
- `stage1_contract_target_refs_covered`
- `raw_stage2_contract_fully_covered`
- `final_stage2_contract_fully_covered`
- `raw_stage1_candidate_decision_coverage`
- `raw_stage1_relation_ref_coverage`
- `raw_stage1_target_ref_coverage`
- `raw_stage2_task_coverage`
- `final_stage2_task_coverage`

`temp/analyze_staged_failures.py` 也新增 stage-1 blocker：

- `missing_relation_drafts`：VLM 确认了候选，但没有复制/输出 `relation_pass_response_template.relations`；
- `missing_target_draft`：VLM 确认了 target 候选，但没有复制/输出 `relation_pass_response_template.target`。

这些指标比单纯看最终通过率更接近 GeoSym-Debug 的目标：定位失败是在候选关系审阅、GeoSym build instruction 转换、repair 之后仍缺 task，还是后续 canonical verification。

## Stage Contract Repair

pipeline 会保留 raw VLM 输出，同时允许 final semantic 做轻量 bookkeeping repair：

- 如果 VLM 的 `relations` / `target` / `build_instructions` 已经引用某个 script candidate，但漏写 `candidate_decisions`，final semantic 会补一条 `decision=confirm`；
- 如果 `candidate_review_checklist` 中某个 candidate 完全没有被 VLM 审阅，final semantic 会补一条 `decision=uncertain`；
- repair 添加的条目会带 `repair_source="stage_contract_candidate_decision"`；
- 它不会新增 geometry relation，也不会新增 build instruction。

这个 repair 的目的不是给 VLM 算成功，而是让后续 task-packet、candidate usage、final contract 分析稳定运行。raw/final 分工如下：

- `raw_stage_contract_*`：评价 VLM 本身是否完成候选审阅；
- `stage_contract_repair_*`：记录框架补了多少 bookkeeping decision；
- `final_stage_contract_*`：评价 repair 后的 final semantic 是否能被后续可验证框架消费；
- `analyze_staged_failures.py` 仍优先用 raw stage contract 判定 `no_candidate_decisions`，不会因为 final repair 掩盖 VLM 第一阶段失败。

## VLM Request Audit

staged OpenRouter 调用会同时发送：

1. 原始几何题图片；
2. 如果存在，则发送脚本生成的 overlay 图片，包含 `vl_*`、`vt_*`、`vc_*` 等可 grounding 的候选 ID。

为了方便复盘，`staged_transcript` 中每个 stage 会记录 `request_audit`：

```json
{
  "request_audit": {
    "prompt_chars": 16802,
    "num_content_parts": 3,
    "num_image_parts": 2,
    "input_images": [
      {"role": "original", "path": ".../images/sample.png", "exists": true, "bytes": 12345},
      {"role": "overlay", "path": ".../visual_overlays/sample.png", "exists": true, "bytes": 23456}
    ]
  }
}
```

这个 audit 不保存 base64 图像内容，只保存路径、role、大小和 MIME。它的用途是排查真实 VLM 失败时到底是：

- 没有 overlay；
- overlay 路径不存在；
- prompt 太长；
- stage1/stage2 实际只收到原图；
- 或者模型收到了两张图但仍没有使用 overlay ID。

## Prompt Context Compaction

staged VLM prompt 不再把完整 `script_observations` 和完整 `evidence_graph` 原样塞给模型。真实 Geometry3K 样本里视觉候选、topology hints、build suggestions 可能很长，重复上下文会让模型忽略真正重要的 GeoSym contract。

现在 `HybridSemanticExtractor` 会按 stage 构造 compact context：

- `visual_candidates`：只保留有限数量的 lines/circles/text/topology IDs，用于 grounding；
- `script_observations compact`：保留 text hints、semantic candidates、少量 relation/topology hints、engine build suggestions；
- `evidence_graph compact`：保留 relation/target candidates、candidate review checklist、builder contract、stage-specific template。

stage-specific 裁剪规则：

- relation stage 保留 `candidate_review_checklist` 和 `relation_pass_response_template`，但不携带 `build_tasks`；
- instruction stage 保留 `build_tasks`、`build_suggestions` 和 `builder_contract`，但不重复携带 `relation_pass_response_template`；
- 两个 stage 都不会把 `evidence_graph.visual_candidates` 整块重复塞入 prompt，只使用单独的 compact visual block。

离线 smoke 会检查：

- relation prompt 仍包含 `candidate_review_checklist` 和 `relation_pass_response_template`；
- instruction prompt 仍包含 `required_tasks`、`action_schema` 和 `from_candidate`；
- relation context 不含 `build_tasks`；
- instruction context 不含 `relation_pass_response_template`；
- instruction context 不重复携带 `evidence_graph.visual_candidates`。

## Theorem-Level Relation Types

当前 parser 允许 VLM 抽取 theorem-level relation，但不会把它们强行送入 canonical builder。典型类型包括：

- `circle`
- `chord`
- `arc_measure`
- `angle`
- `similar`
- `congruent`
- `incidence`
- `intersecting_chords`
- `right_triangle`

例如 `right triangle ABC` 可以被抽成：

```json
{
  "type": "right_triangle",
  "objects": ["A", "B", "C"],
  "candidate_id": "sc_right_triangle_0",
  "grounded": false,
  "raw": "right triangle ABC"
}
```

但它本身不等于可 build 的 `set_perpendicular` 指令。只有当题干或图像明确给出直角顶点，并且 VLM 能把两条边绑定到 visual ids 时，才应该转换成 `set_perpendicular`。否则样本会被归为 `theorem_solver_missing`，避免伪造 canonical geometry。

相对地，`right_angle` 是更具体、可转换的关系：

```json
{
  "type": "right_angle",
  "objects": ["AB", "BC"],
  "angle": "ABC",
  "vertex": "B",
  "candidate_id": "sc_right_angle_0",
  "grounded": true
}
```

当 `right_angle` 明确包含两条 incident lines，例如 `AB` 和 `BC`，planner 可以把它转换成：

```json
{"action": "set_perpendicular", "lines": ["AB", "BC"]}
```

所以推荐 VLM 在 stage 1 中区分：

- 只有 “right triangle ABC” 但不知道直角顶点：输出 `right_triangle`；
- 明确 “angle ABC is 90 degrees” 或图上有可定位直角标记：输出 `right_angle`，并在 stage 2 转成 `set_perpendicular`。

## Staged VLM Experiment Runner

真实数据解析现在推荐用一个 orchestration 脚本跑小批量实验，而不是手动逐个调用解析、汇总和诊断脚本：

```bash
export OPENROUTER_API_KEY=...

LIMIT=20 \
DATASET_NAME=geometry3k_vlm_staged \
SOURCE_DATASET_NAME=geometry3k \
MODEL=qwen/qwen3.7-plus \
JUDGE_BACKEND=none \
bash temp/run_staged_vlm_experiment.sh
```

这个脚本会依次执行：

1. 可选 ingest：`RUN_INGEST=1` 时从原始 parquet 生成 `/dev/shm/geosym_datas/<dataset>/metadata/records.jsonl`；
2. staged OpenRouter parse：`parse_real_data_hybrid.py --backend openrouter --extraction-mode staged`；
3. parse summary：`summarize_parse_reports.py`；
4. staged failure matrix：`analyze_staged_failures.py`；
5. staged VLM behavior report：`analyze_staged_vlm_behavior.py`；
6. semantic version compare：`compare_semantic_versions.py`；
7. triage：`triage_hybrid_reports.py`；
8. run manifest：`run_manifest.json`。

所有汇总产物会写入：

```text
/data/Geosym127K/Geosym-World/artifacts/datasets/parsed/<DATASET_NAME>/analysis_runs/<RUN_ID>/
```

前端的 `Run Staged VLM Experiment` 表单本质上也是调用这个脚本。前端不会接收或保存 API key，只继承启动服务时的 `OPENROUTER_API_KEY` 环境变量。

这套入口的目标不是直接追求通过率，而是把真实数据解析拆成可诊断的阶段：

- script/VLM 是否从图像和题干中发现了候选关系；
- VLM 是否确认、纠错或拒绝脚本候选；
- VLM 是否保留 `buildability` 和 `builder_mapping`，也就是是否理解当前 GeoSym builder 能 build 什么；
- stage 2 是否按 `instruction_task_packet.response_template` 生成 GeoSym engine 可 build 的指令；
- repair 是否只是补齐格式，还是掩盖了 VLM 没有理解 engine contract；
- canonical builder / verifier / consistency check 最终卡在哪一步。

## Multi-Source Script Candidate Layer

真实几何图片不能只靠 VLM 从图像直接抽完整 GeoSym JSON。当前 parser 已把脚本侧改成多来源候选层：

- `problem_text`：题干，是 primary source；
- visual primitive detector：只提供 `vl_* / vt_* / vx_*` 等像素级 visual evidence，不作为数学坐标。

数据集中的 `solution_text`、`answer`、`raw` logic form、formal state 和 scene
metadata 都是 evaluator/reference 侧字段。它们可以在私有数据行中存在，但不能
进入正式 Parser 的候选层；需要用这些字段做离线诊断时，必须由 evaluator 单独
读取并在报告中标记为 reference-only。

`ScriptObservationExtractor` 会从这些来源抽：

- `length`：如 `AB = 3`、`Equals(LengthOf(Line(A,B)), 3)`；
- `parallel`：如 `AB ∥ CD`、`Parallel(Line(A,B), Line(C,D))`；
- `perpendicular`：如 `AB ⊥ BC`、`Perpendicular(Line(A,B), Line(B,C))`；
- `right_angle`：如 `angle ABC = 90`、`RightAngle(A,B,C)`，可转换为 `set_perpendicular(["AB","BC"])`；
- `angle`：如 `angle ABC = 40`、`Equals(MeasureOfAngle(A,B,C), 40)`，作为 theorem-level relation 保留，不强行转换成坐标约束；
- `area`：如 `area of triangle ABC = 12`、`Equals(AreaOf(Triangle(A,B,C)), 12)`，作为 theorem-level relation；
- `perimeter`：如 `perimeter ABC = 18`、`PerimeterOf(Triangle(A,B,C), 18)`，作为 theorem-level relation；
- `ratio`：如 `AB:BC = 2:3`、`RatioOf(LengthOf(A,B), LengthOf(B,C), 2/3)`，作为 theorem-level relation；
- `triangle/right_triangle/circle` 等 theorem-only 候选：保留给诊断，不伪装成 coordinate build instruction；
- target candidate：如 `Find AB`、`Find angle ABC`、`Find area ABC`、`Find perimeter ABC`、`Find AB:BC`。

这些候选会带 `source_text`，例如 `problem_text` 或视觉证据 ID。它们不是最终事实，只是 VLM stage 1 必须逐项审查的候选。VLM 需要输出：

```json
{
  "candidate_decisions": [
    {"candidate_id": "sc_len_0", "decision": "confirm", "reason": "AB=3 appears in the question and segment AB is visible"},
    {"candidate_id": "sc_right_angle_0", "decision": "uncertain", "reason": "text says right angle but diagram binding is unclear"}
  ]
}
```

之后 stage 2 才能把 `confirm/correct` 的 coordinate-buildable 候选转换成 `build_instructions`。这样设计的原则是：脚本尽量多召回，VLM 负责语义审查，GeoSym builder/verifier 负责最终硬验证。

## Source of Truth for Real-Image Parsing

真实数据解析不能把 VLM 输出当作 GeoSym JSON 的唯一来源。当前结构里 source of truth 分三层：

1. `script_observations` / `evidence_graph` 是候选证据源：
   - 脚本从题干和图像 primitive 中尽量召回候选；
   - 候选可以是可构图约束，也可以是 theorem-level relation；
   - 每个候选都带 `candidate_id`、`source_text`、`buildability` 和 `builder_mapping`。

2. VLM 是候选审查器：
   - stage 1 对每个 candidate 输出 `confirm/correct/reject/uncertain`；
   - 如果修正候选，必须给 `corrected_relation`；
   - 如果抽到新关系，也必须尽量绑定到已有 visual/text evidence；
   - VLM 不直接决定 SymPy 数学坐标。

3. GeoSym builder / verifier 是硬验收层：
   - 只有 `buildability.can_generate_build_instruction=true` 且 stage 2 能映射到 supported builder action 的关系才进入 canonical builder；
- `angle/area/perimeter/ratio/circle/similar/congruent/right_triangle` 等 theorem-only relation 保留在 formal graph，用于诊断或后续 theorem solver；
   - adapter 只做机械吸附：把 VLM 已抽对但漏写的 `candidate_id/from_candidate/buildability/builder_mapping` 补齐，不新增事实；
   - 最终是否通过仍由 canonical build、image alignment、answer consistency 和 optional judge 决定。

因此，一条真实题失败时应优先看断点，而不是只看最终 pass/fail：

- script 没召回候选：补 regex/parser/visual detector；
- VLM 没审查候选：修 stage-1 prompt 或模型；
- VLM 抽到关系但没吸附 builder contract：看 `RelationPassAdapter` 和 stage contract；
- stage 2 没生成指令：看 `instruction_task_packet` 和 `InstructionPassAdapter`；
- 指令可 build 但验证失败：修 canonical builder / constraint solver；
- theorem-only 关系很多：这是当前 solver 覆盖边界，不应该伪装成 coordinate-buildable。

## Partial Canonical Scaffold

真实题经常同时包含两类信息：

- 可由当前 GeoSym 2D builder 构造的低阶约束，例如 `length / parallel / perpendicular / right_angle`；
- 暂时只能保留在 formal graph 中的 theorem-level 关系，例如 `angle / circle / similar / congruent / right_triangle`。

当前 parser 不再因为出现 theorem-level relation 就把整题整体判成 `theorem_only`。如果同一题里仍有可构图的低阶约束，`FormalGraphBuilder` 会输出：

```json
{
  "mode": "coordinate_buildable",
  "requires_theorem_solver": true,
  "coordinate_relation_types": ["length"],
  "theorem_relation_types": ["angle"]
}
```

这表示：GeoSym 可以先生成一个 partial canonical scaffold，用于保存可验证的数学骨架、渲染 review sketch、做 image alignment 和 answer consistency；同时 theorem relation 仍被显式标记为未解决的 solver 边界。

相关 report/summary 指标：

- `formal_requires_theorem_solver`
- `formal_partial_coordinate_scaffold`
- `formal_coordinate_relation_types`
- `formal_theorem_relation_types`
- `formal_unsupported_relation_types`

这个设计对 GeoSym-Hard / GeoSym-Debug 很重要：benchmark 不必等完整 theorem solver 才能纳入真实题，先把题目中可验证的几何子结构转换成 GeoSym state，再把剩余失败归因到 theorem solver 缺口。

另一个保护规则是：如果 VLM 把脚本候选 `correct` 成 theorem-only relation，例如把 `sc_len_0` 修正成 `angle ABC = 40`，`InstructionTaskPacketBuilder` 不会继续沿用原来的 `length -> set_length` build task。该 candidate 会进入 `deferred_tasks`，其 `builder_mapping.status=not_buildable`，避免“VLM 已经纠错，但框架又回退成错误可构图约束”。

## Relation Draft Repair

`RelationDraftRepairer` 是一个受控 final-stage repair，不改变 raw VLM 诊断：

- raw report 仍会记录 `missing_relation_drafts` / `missing_target_draft`；
- 只有当 VLM 已经 `confirm/correct` 某个 candidate，或 semantic 已经引用该 candidate 时，repair 才会从 `relation_pass_response_template` 复制对应 `relations` / `target`；
- 如果 VLM 的 decision 是 `correct` 且提供了有效 `corrected_relation`，repair 会优先使用 VLM 修正后的 relation/target，而不是复制脚本模板；
- 新增的 relation/target 会带 `repair_source`，区分来自 `relation_pass_response_template` 还是 `candidate_decision.corrected_relation`；
- 如果 relation 已存在但漏写 `buildability` 或 `builder_mapping`，repair 会在 candidate 已确认/引用且 relation type 与模板一致时，从 `relation_pass_response_template` 补齐这些 metadata，并记录 `metadata_repair_source`；
- repair 后的 final semantic 可以继续进入 grounding、formal graph、instruction repair 和 canonical builder。

这个 repair 的目的不是替代 VLM，而是把“VLM 已经确认候选但漏写模板字段”的样本继续推进到可验证框架中，同时保留 raw failure matrix 里的模型错误。

对 `decision=correct` 的候选，后续 `InstructionRepairer` 会优先从 final semantic relation/target 生成 build instructions。例如脚本候选是 `AB=3`，VLM 修正为 `AC=5`，final build instructions 会使用 `AC=5`，不会退回旧模板的 `AB=3`。普通 `confirm` 仍使用脚本模板，以保持 task packet template adherence。

同样地，`InstructionTaskPacketBuilder` 在构造 stage-2 `response_template` 时也会尊重 `candidate_decisions[*].corrected_relation`。因此 stage-2 prompt 看到的 build instruction 草稿已经是 VLM 修正后的几何关系，不需要等 final repair 才纠正。

当 task 来自 corrected relation 时，`required_tasks[*]` 会显式记录：

- `corrected_relation_used: true`
- `corrected_relation`
- `effective_candidate_object`
- `instruction_source: "candidate_decision.corrected_relation"`
- `relation_buildability`
- `builder_mapping`

分析指标中对应 `instruction_task_packet_corrected_relation_tasks`，用于统计 stage-2 prompt 有多少任务来自 VLM 纠错。
分析指标中还对应 `instruction_task_packet_builder_mapped_tasks`，用于统计有多少 stage-2 task 明确带有 relation-to-builder action 映射。

metadata repair 对应指标：

- `relation_draft_repair_metadata_patched_relations`
- `relation_draft_repair_added_buildability`
- `relation_draft_repair_added_builder_mapping`

## Relation Pass Adapter

`RelationPassAdapter` 位于 OpenRouter staged relation pass 的 raw output 和 schema validation 之间。

它解决一个真实数据中很常见的问题：VLM 读懂了“AB=3”并输出 relation，但没有严格复制 `candidate_id=sc_len_0` 或 candidate 级 `buildability/builder_mapping`。这种情况下：

- raw response 仍保存到 `vlm_passes/*.relation_pass.raw.json`；
- adapter 后的 relation pass 保存到 `vlm_passes/*.relation_pass.json`；
- adapter 只按 `type + object/objects + value` 匹配唯一 script candidate；
- 匹配成功后补 `candidate_id`、`buildability`、`builder_mapping`，并打 `adapter_source="relation_pass_adapter"`；
- target 也可以按 type/object 补 `candidate_id/from_candidate`；
- adapter 不补 `candidate_decisions`，所以 VLM 没有审阅 checklist 仍会被 raw/final stage contract 记录。

新增指标：

- `staged_relation_pass_adapter_used`
- `staged_relation_pass_adapter_relation_patches`
- `staged_relation_pass_adapter_candidate_id_patches`
- `staged_relation_pass_adapter_buildability_patches`
- `staged_relation_pass_adapter_builder_mapping_patches`

这些指标用于区分两类情况：

- VLM 完全没抽到关系：adapter 无法帮助；
- VLM 抽到正确关系但没遵守 GeoSym 字段合同：adapter 可以把 relation 接回 script candidate 和 builder contract，后续 task packet 能继续生成可验证 build 指令。

## Instruction Pass Adapter

`InstructionPassAdapter` 位于 OpenRouter staged instruction pass 的 raw output 和 `SemanticOutputValidator(stage="instruction_pass")` 之间。

它处理 stage-2 的常见格式问题：VLM 输出了正确的 GeoSym builder action，例如 `set_length(points=["A","B"], value="3")`，但漏写 `from_candidate=sc_len_0`，或者 `set_target.target` 只写了 `type/object/value_symbol`，没有复制 task packet 中的完整 target 字段。

adapter 的规则：

- 输入是 raw instruction pass 和 `instruction_task_packet`；
- 只在 `instruction_task_packet.required_tasks[*].instruction_requirements[*].expected_instruction` 中寻找唯一匹配；
- 匹配规则要求 action 相同，已有字段与 expected instruction 不冲突；
- 匹配成功后补 `from_candidate` 和缺失字段，nested target dict 也只补缺失字段；
- 不新增缺失 instruction，因此 raw/final task coverage 仍能显示 VLM 少输出了哪些 required action；
- 不匹配、多匹配或字段冲突时不 patch。

新增指标：

- `staged_instruction_pass_adapter_used`
- `staged_instruction_pass_adapter_instruction_patches`
- `staged_instruction_pass_adapter_from_candidate_patches`
- `staged_instruction_pass_adapter_field_patches`

这个 adapter 让“VLM 已经知道要调用哪个 builder action，但没按 GeoSym task packet 精确写字段”的样本继续进入可验证构图；同时缺动作、错动作、错字段仍会被 task packet coverage、template adherence 和 instruction repair 记录。
