"""Turn parser hypotheses into deterministic synthetic seed specs."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from geosym_world.schema import ParseGraph, math_expression_from_text
from geosym_world.parser.constraint_completion import CompletionStatus, build_constraint_completion_plan
from geosym_world.synthesis import select_template_families
from geosym_world.synthesis.templates import TemplateFamily


FAMILY_BY_RELATION = {
    "congruent": "congruence_sss_proof",
    "similar": "similarity_dual_proof",
    "area": "triangle_partition_area",
    "ratio": "triangle_partition_area_ratio",
    "circle": "circle_sector",
    "arc_measure": "sector_arc_length",
    "arc_length": "sector_arc_length",
    "diameter": "circle_sector",
    "radius": "circle_sector",
    "right_triangle": "triangle_area_from_height",
    "triangle": "triangle_area_from_height",
    "parallel": "parallel_cut_similarity",
    "perpendicular": "triangle_area_from_height",
    "right_angle": "triangle_area_from_height",
}
RELATION_PRIORITY = {
    "congruent": 100,
    "similar": 95,
    "area": 90,
    "ratio": 85,
    "arc_measure": 80,
    "arc_length": 80,
    "diameter": 75,
    "radius": 75,
    "circle": 75,
    "perimeter": 70,
    "right_triangle": 65,
    "triangle": 60,
    "parallel": 70,
    "perpendicular": 70,
    "right_angle": 65,
}
SEED_MAPPING_MODE = {
    "congruent": "proof_completion",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic", required=True, help="Parser semantic JSON or directory of semantic JSON files")
    parser.add_argument("--grounded", default=None, help="Optional grounded-relation JSON or directory matching --semantic")
    parser.add_argument("--output", required=True, help="Enhanced-spec JSONL consumed by --input-enhanced")
    parser.add_argument("--variants-per-seed", type=int, default=3, help="Number of deterministic template completions per accepted relation")
    parser.add_argument("--rejection-report", default=None, help="Optional JSON report for parser relations rejected as synthesis seeds")
    args = parser.parse_args()
    if args.variants_per_seed <= 0:
        parser.error("--variants-per-seed must be positive")
    semantic_paths = sorted(Path(args.semantic).glob("*.json")) if Path(args.semantic).is_dir() else [Path(args.semantic)]
    if not semantic_paths:
        raise ValueError(f"No semantic JSON files found: {args.semantic}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    rejected = []
    with output.open("w", encoding="utf-8") as handle:
        for path in semantic_paths:
            semantic = _load_seed_semantics(path)
            grounded_path = _matching_grounded_path(args.grounded, path, len(semantic_paths))
            if grounded_path is not None:
                grounded = json.loads(grounded_path.read_text(encoding="utf-8"))
                semantic = _merge_grounding_status(semantic, grounded)
                semantic["metadata"] = {
                    **(semantic.get("metadata") or {}),
                    "grounded_sidecar_path": str(grounded_path),
                }
            completion = (semantic.get("metadata") or {}).get("constraint_completion_plan")
            if isinstance(completion, dict) and completion.get("status") in {
                CompletionStatus.EXACT.value,
                CompletionStatus.DETERMINISTIC_COMPLETION.value,
                CompletionStatus.BRANCHING_COMPLETION.value,
            } and completion.get("trace"):
                if any(item.get("op") == "set_target" for item in completion["trace"] if isinstance(item, dict)):
                    seed_id = str(semantic.get("id") or path.stem)
                    metadata = {
                        "seed_source": "parser_constraint_completion",
                        "seed_source_path": str(path),
                        "seed_evidence_mode": "hypothesis_constraint_completion",
                        "seed_semantic_checksum": _semantic_checksum(semantic),
                        "completion_policy": "typed_constraint_completion",
                        "constraint_completion_plan": completion,
                    }
                    traces = completion.get("branches") or [completion["trace"]]
                    for branch_index, trace in enumerate(traces):
                        suffix = f"_b{branch_index}" if len(traces) > 1 else ""
                        spec = {
                            "id": f"parser_completion_{seed_id}{suffix}",
                            "family": "parser_constraint_completion",
                            "trace": trace,
                            "question": str(
                                semantic.get("question")
                                or (semantic.get("metadata") or {}).get("source_question")
                                or "Complete the geometry task from the recovered constraints.",
                            ),
                            "answer_hint": "",
                            "difficulty": "medium",
                            "diagnostic_axes": ("parser_constraint_completion",),
                            "metadata": {**metadata, "completion_branch_index": branch_index},
                        }
                        handle.write(json.dumps(spec, ensure_ascii=False) + "\n")
                        written += 1
                    continue
                rejected.append({
                    "source": str(path),
                    "reason": "constraint_completion_has_no_value_target",
                    "completion_status": completion.get("status"),
                })
            relations = [item for item in semantic.get("relations", []) if isinstance(item, dict)]
            rejected.extend(
                {
                    "source": str(path),
                    "relation_index": index,
                    "type": relation.get("type"),
                    "objects": relation.get("objects"),
                    "reason": reason,
                }
                for index, relation in enumerate(relations)
                if (reason := _seed_rejection_reason(relation)) is not None
            )
            source_constraints = _source_constraints(semantic)
            candidates = [item for item in relations if _usable_relation(item)]
            relation = max(candidates, key=_relation_score, default=None)
            if relation is None:
                rejected.append({"source": str(path), "reason": "no_grounded_supported_relation", "relation_count": len(relations)})
                continue
            family_name = FAMILY_BY_RELATION[relation["type"]]
            family = select_template_families([family_name])[0]
            family, parameter_projection = _project_seed_parameters(family, relation, constraints=source_constraints)
            if parameter_projection.get("status") == "inconsistent":
                rejected.append({
                    "source": str(path),
                    "type": relation.get("type"),
                    "objects": relation.get("objects"),
                    "reason": "source_constraints_inconsistent_with_completion_family",
                    "parameter_projection": parameter_projection,
                })
                continue
            variants = list(family.variants())
            if not variants:
                continue
            seed_id = str(semantic.get("id") or path.stem)
            for variant_index, variant in enumerate(variants[:args.variants_per_seed]):
                evidence_mode = _seed_evidence_mode(relation)
                spec = replace(
                    variant,
                    id=f"parser_seed_{seed_id}_v{variant_index}",
                    metadata={
                        **variant.metadata,
                        "seed_source": "parser_hypothesis",
                        "seed_source_path": str(path),
                        "seed_relation": relation,
                        "seed_constraint": _seed_constraint(relation),
                        "seed_constraints": [_seed_constraint(item) for item in source_constraints],
                        "seed_constraint_count": len(source_constraints),
                        "seed_constraint_completion": (semantic.get("metadata") or {}).get("constraint_completion_plan"),
                        "seed_parameter_projection": parameter_projection,
                        "seed_evidence_mode": evidence_mode,
                        "seed_semantic_checksum": _semantic_checksum(semantic),
                        "completion_policy": "template_completion_then_verify",
                        "seed_mapping_mode": _mapping_mode(relation, evidence_mode),
                        "completion_relation": family_name,
                        "completion_variant": variant_index,
                        "seed_relation_score": _relation_score(relation),
                    },
                )
                handle.write(json.dumps(asdict(spec), ensure_ascii=False) + "\n")
                written += 1
    if not written:
        raise ValueError("No supported parser relations were available for synthesis")
    if args.rejection_report:
        rejection_path = Path(args.rejection_report)
        rejection_path.parent.mkdir(parents=True, exist_ok=True)
        rejection_path.write_text(json.dumps({"num_rejected": len(rejected), "items": rejected}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "num_seed_specs": written}, ensure_ascii=False))


def _has_evidence(relation: dict) -> bool:
    evidence = relation.get("evidence") or {}
    return bool(
        evidence.get("text_refs")
        or evidence.get("visual_evidence_ids")
        or evidence.get("visual_ids")
        or relation.get("visual_ids")
    )


def _load_seed_semantics(path: Path) -> dict:
    """Load legacy semantic output or a strict ParseGraph proposal."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("semantic_parse_graph"), dict):
        return _semantic_from_parse_graph(
            ParseGraph.from_dict(payload["semantic_parse_graph"]),
            source_path=path,
            source_format="semantic_parse_graph",
        )
    if isinstance(payload, dict) and isinstance(payload.get("parse_graph"), dict):
        return _semantic_from_parse_graph(ParseGraph.from_dict(payload["parse_graph"]), source_path=path)
    if isinstance(payload, dict) and {"objects", "relations", "quantities", "regions", "targets"}.issubset(payload):
        return _semantic_from_parse_graph(ParseGraph.from_dict(payload), source_path=path)
    if isinstance(payload, dict) and payload.get("hypotheses"):
        for hypothesis in payload["hypotheses"]:
            candidate = hypothesis.get("payload") if isinstance(hypothesis, dict) else None
            if isinstance(candidate, dict) and isinstance(candidate.get("parse_graph"), dict):
                return _semantic_from_parse_graph(ParseGraph.from_dict(candidate["parse_graph"]), source_path=path)
    if not isinstance(payload, dict):
        raise ValueError(f"Seed source must be a JSON object: {path}")
    return payload


def _semantic_from_parse_graph(
    graph: ParseGraph,
    *,
    source_path: Path,
    source_format: str = "parse_graph",
) -> dict:
    completion_plan = build_constraint_completion_plan(graph)
    objects = {item.id: item for item in graph.objects}
    evidence = {item.id: item for item in graph.evidence}

    def object_name(reference: str) -> str:
        item = objects.get(reference)
        if item is not None and item.visible_name:
            return item.visible_name
        if reference.startswith("semantic:"):
            return reference[len("semantic:"):]
        return reference

    def relation_evidence(item: Any) -> dict:
        text_refs = []
        visual_refs = []
        for evidence_id in item.evidence_refs:
            evidence_item = evidence.get(evidence_id)
            if evidence_item is None:
                continue
            if evidence_item.kind.value == "text_span":
                text_refs.append(evidence_id)
            elif evidence_item.kind.value in {"image_region", "pixel_point"}:
                visual_refs.append(evidence_id)
        metadata = dict(item.metadata)
        visual_ids = list(metadata.get("visual_ids") or [])
        output = {}
        if text_refs:
            output["text_refs"] = text_refs
        if visual_refs or visual_ids:
            output["visual_evidence_ids"] = visual_refs
        return output

    relations = []
    for item in graph.relations:
        if item.predicate not in FAMILY_BY_RELATION:
            continue
        relation_metadata = dict(item.metadata)
        evidence_payload = relation_evidence(item)
        relation = {
            "type": item.predicate,
            "objects": [object_name(argument.ref) for argument in item.arguments],
            "grounded": False,
            "confidence": item.confidence,
            "source": item.origin.value,
            "evidence": evidence_payload,
            "visual_ids": list(relation_metadata.get("visual_ids") or []),
            "value": relation_metadata.get("value", ""),
            "value_symbol": relation_metadata.get("value_symbol", ""),
            "parse_graph_relation_id": item.id,
            "observation_only": relation_metadata.get("observation_only") is True,
        }
        if item.origin.value == "text" and evidence_payload.get("text_refs"):
            relation["grounded"] = True
        relations.append(relation)
    source_constraints = []
    for item in graph.quantities:
        expression = item.expression.to_dict()
        source_constraints.append({
            "type": item.kind,
            "objects": [object_name(argument.ref) for argument in item.arguments],
            "grounded": False,
            "confidence": item.confidence,
            "source": item.origin.value,
            "evidence": relation_evidence(item),
            "value": item.raw_text or "",
            "value_symbol": "",
            "expression": expression,
            "expression_text": str(item.expression.to_sympy()),
            "parse_graph_quantity_id": item.id,
        })
    return {
        "id": graph.source_id,
        "relations": relations,
        "source_constraints": source_constraints,
        "metadata": {
            "source_format": source_format,
            "source_path": str(source_path),
            "parse_graph_id": graph.id,
            "parse_graph_checksum": graph.checksum(),
            "constraint_completion_plan": completion_plan.to_dict(),
        },
    }


def _semantic_checksum(semantic: dict) -> str | None:
    checksum = semantic.get("checksum")
    if checksum:
        return str(checksum)
    metadata = semantic.get("metadata")
    if isinstance(metadata, dict) and metadata.get("parse_graph_checksum"):
        return str(metadata["parse_graph_checksum"])
    return None


def _seed_constraint(relation: dict) -> dict[str, Any]:
    """Keep the source assertion independently of the chosen completion family."""
    evidence = relation.get("evidence") if isinstance(relation.get("evidence"), dict) else {}
    return {
        "type": str(relation.get("type") or ""),
        "objects": [str(item) for item in relation.get("objects") or []],
        "value": relation.get("value"),
        "value_symbol": relation.get("value_symbol"),
        "expression": relation.get("expression"),
        "expression_text": relation.get("expression_text"),
        "source": relation.get("source"),
        "parse_graph_relation_id": relation.get("parse_graph_relation_id"),
        "parse_graph_quantity_id": relation.get("parse_graph_quantity_id"),
        "evidence": evidence,
    }


def _source_constraints(semantic: dict) -> list[dict[str, Any]]:
    """Collect parser assertions without making them template selectors."""
    sources = [item for item in semantic.get("source_constraints") or [] if isinstance(item, dict)]
    sources.extend(item for item in semantic.get("relations") or [] if isinstance(item, dict))
    for item in semantic.get("quantities") or []:
        if not isinstance(item, dict):
            continue
        source = dict(item)
        source["type"] = source.get("kind") or source.get("type") or "unknown"
        if not source.get("objects") and source.get("object"):
            source["objects"] = [source["object"]]
        if source.get("value") in {None, ""}:
            source["value"] = source.get("value_symbol") or source.get("raw") or ""
        sources.append(source)
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...], str, str]] = set()
    for item in sources:
        key = (
            str(item.get("type") or item.get("kind") or ""),
            tuple(str(value) for value in item.get("objects") or []),
            str(item.get("value") or item.get("value_symbol") or item.get("raw") or ""),
            str(item.get("parse_graph_relation_id") or item.get("parse_graph_quantity_id") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def _project_seed_parameters(
    family: TemplateFamily,
    relation: dict,
    *,
    constraints: list[dict] | None = None,
) -> tuple[TemplateFamily, dict[str, Any]]:
    """Restrict a completion family only for exact, representable source values."""
    relation_type = str(relation.get("type") or "")
    source_relations = constraints or [relation]
    candidates = list(dict.fromkeys(
        value
        for source_relation in source_relations
        for value in _value_candidates(source_relation)
    ))
    overrides: dict[str, tuple[Any, ...]] = {}
    applied: dict[str, Any] = {}
    conflicts: dict[str, list[str]] = {}
    preservation: dict[str, Any] = {}

    if "ratio" in family.parameter_grid and relation_type == "congruent":
        explicit_ratio_values = [
            value
            for source_relation in source_relations
            if str(source_relation.get("type") or source_relation.get("kind") or "")
            in {"ratio", "proportional_lengths"}
            for value in _value_candidates(source_relation)
        ]
        conflicts["ratio"] = ["congruent_requires_dedicated_completion_family"]
        preservation = {
            "status": "conflict",
            "rule": "congruent_requires_dedicated_completion_family",
            "explicit_ratio_values": list(dict.fromkeys(str(value) for value in explicit_ratio_values)),
        }
    elif "ratio" in family.parameter_grid and relation_type in {"ratio", "parallel", "similar"}:
        matches = _matching_values(candidates, family.parameter_grid["ratio"])
        if len(matches) == 1:
            ratio = matches[0]
            overrides["ratio"] = (ratio,)
            applied["ratio"] = ratio
        elif len(matches) > 1:
            conflicts["ratio"] = [str(value) for value in matches]
    if "central_angle" in family.parameter_grid and relation_type in {"arc_measure", "arc_length", "circle"}:
        matches = _matching_values(candidates, family.parameter_grid["central_angle"])
        if len(matches) == 1:
            angle = matches[0]
            overrides["central_angle"] = (angle,)
            applied["central_angle"] = angle
        elif len(matches) > 1:
            conflicts["central_angle"] = [str(value) for value in matches]
    if "radius" in family.parameter_grid and relation_type in {"circle", "arc_measure", "diameter", "radius"}:
        matches = _matching_number_values(
            candidates,
            family.parameter_grid["radius"],
            divide_by_two=relation_type == "diameter",
        )
        if len(matches) == 1:
            radius = matches[0]
            overrides["radius"] = (radius,)
            applied["radius"] = radius
        elif len(matches) > 1:
            conflicts["radius"] = [str(value) for value in matches]

    for parameter in conflicts:
        overrides.pop(parameter, None)
        applied.pop(parameter, None)

    if not overrides and not conflicts:
        return family, {
            "status": "retained_not_applied",
            "source_relation_type": relation_type,
            "available_values": candidates,
            "applied_parameters": {},
            "conflicting_parameters": {},
            "constraint_preservation": preservation,
        }
    projected = TemplateFamily(
        name=family.name,
        parameter_grid={**family.parameter_grid, **overrides},
        builder=family.builder,
        description=family.description,
    )
    return projected, {
        "status": "inconsistent" if relation_type == "congruent" and conflicts else ("applied" if not conflicts else ("partially_applied" if overrides else "retained_not_applied")),
        "source_relation_type": relation_type,
        "available_values": candidates,
        "applied_parameters": applied,
        "conflicting_parameters": conflicts,
        "constraint_preservation": preservation,
    }


def _value_candidates(relation: dict) -> list[str]:
    values: list[str] = []
    for key in ("value", "value_symbol", "expression_text"):
        value = relation.get(key)
        if value is not None and str(value).strip():
            values.append(str(value).strip())
    expression = relation.get("expression")
    if isinstance(expression, dict):
        try:
            values.append(str(math_expression_from_text(str(expression.get("raw") or "")).to_sympy()))
        except (TypeError, ValueError):
            pass
    return list(dict.fromkeys(values))


def _matching_values(candidates: list[str], allowed: tuple[Any, ...]) -> list[Any]:
    allowed_by_expr = {_canonical_expression(str(value)): value for value in allowed}
    matches: list[Any] = []
    for candidate in candidates:
        matched = allowed_by_expr.get(_canonical_expression(candidate))
        if matched is not None and matched not in matches:
            matches.append(matched)
    return matches


def _matching_number_values(
    candidates: list[str],
    allowed: tuple[Any, ...],
    *,
    divide_by_two: bool = False,
) -> list[Any]:
    matches: list[Any] = []
    for candidate in candidates:
        try:
            expression = math_expression_from_text(candidate).to_sympy()
        except (TypeError, ValueError):
            continue
        if divide_by_two:
            expression /= 2
        for value in allowed:
            try:
                if expression == value and value not in matches:
                    matches.append(value)
            except TypeError:
                continue
    return matches


def _canonical_expression(value: str) -> str:
    try:
        return str(math_expression_from_text(value).to_sympy())
    except (TypeError, ValueError):
        return value.replace(" ", "").lower()


def _matching_grounded_path(value: str | None, semantic_path: Path, semantic_count: int) -> Path | None:
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_dir():
        path = candidate / semantic_path.name
    elif semantic_count == 1:
        path = candidate
    else:
        raise ValueError("--grounded must be a directory when --semantic contains multiple files")
    if not path.is_file():
        raise ValueError(f"Grounded relation sidecar does not exist: {path}")
    return path


def _relation_key(relation: dict) -> tuple[str, tuple[str, ...], str]:
    return (
        str(relation.get("type") or ""),
        tuple(str(item) for item in relation.get("objects") or []),
        str(relation.get("value") or ""),
    )


def _merge_grounding_status(semantic: dict, grounded: dict) -> dict:
    merged = dict(semantic)
    relations = [dict(item) for item in semantic.get("relations") or [] if isinstance(item, dict)]
    status_by_key: dict[tuple[str, tuple[str, ...], str], list[dict]] = {}
    for status, field in (("accepted", "accepted_relations"), ("uncertain", "uncertain_relations"), ("rejected", "rejected_relations")):
        for item in grounded.get(field) or []:
            if isinstance(item, dict):
                status_by_key.setdefault(_relation_key(item), []).append({
                    "grounding_status": status,
                    "grounding_reason": item.get("grounding_reason"),
                })
    for relation in relations:
        candidates = status_by_key.get(_relation_key(relation)) or []
        if candidates:
            update = candidates.pop(0)
            relation.update({key: value for key, value in update.items() if value is not None})
    merged["relations"] = relations
    return merged


def _seed_evidence_mode(relation: dict) -> str | None:
    evidence = relation.get("evidence") or {}
    grounded = relation.get("grounded") is True
    if grounded and evidence.get("text_refs"):
        return "verified_text_claim"
    if grounded and relation.get("grounding_status") == "accepted" and (evidence.get("visual_evidence_ids") or evidence.get("visual_ids") or relation.get("visual_ids")):
        return "verified_visual_binding"
    if (
        evidence.get("text_refs")
        or evidence.get("visual_evidence_ids")
        or evidence.get("visual_ids")
        or relation.get("visual_ids")
        or evidence.get("image_refs")
        or evidence.get("supporting_text")
        or relation.get("supporting_text")
    ):
        return "hypothesis_visual_claim"
    return None


def _usable_relation(relation: dict) -> bool:
    return _seed_rejection_reason(relation) is None


def _seed_rejection_reason(relation: dict) -> str | None:
    relation_type = relation.get("type")
    if relation_type not in FAMILY_BY_RELATION:
        return "unsupported_relation_type"
    if relation.get("observation_only") is True:
        return "observation_only_not_seedable"
    if _seed_evidence_mode(relation) is None:
        return "missing_traceable_evidence"
    objects = relation.get("objects") or []
    if not objects or any(not isinstance(item, str) or not item.strip() for item in objects):
        return "missing_objects"
    forbidden = {"THE", "FIRST", "SECOND", "UNKNOWN", "OBJECT"}
    if any(str(item).upper() in forbidden for item in objects):
        return "forbidden_object_token"
    if relation_type in {"congruent", "similar"} and not _is_polygon_pair(objects):
        return "invalid_polygon_pair"
    if relation_type == "ratio" and len(objects) != 2:
        return "ratio_requires_two_objects"
    if relation_type == "ratio" and any(not _is_geometry_reference(item) for item in objects):
        return "ratio_contains_non_geometry_token"
    if relation_type in {
        "area", "arc_length", "arc_measure", "circle", "diameter", "perimeter",
        "radius", "right_triangle", "triangle",
    }:
        if len(objects) != 1:
            return "single_geometry_object_required"
        if not _is_geometry_reference(objects[0]):
            return "relation_contains_non_geometry_token"
    if relation_type in {"parallel", "perpendicular", "right_angle"}:
        if len(objects) != 2:
            return "relation_requires_two_objects"
        if any(not _is_geometry_reference(item) for item in objects):
            return "relation_contains_non_geometry_token"
    return None


def _mapping_mode(relation: dict, evidence_mode: str) -> str:
    prefix = SEED_MAPPING_MODE.get(relation["type"], "direct_completion")
    if evidence_mode.startswith("hypothesis_"):
        return f"hypothesis_{prefix}"
    return prefix


def _is_geometry_reference(value: str) -> bool:
    """Reject algebraic value tokens before mapping a relation to a geometry family."""
    normalized = value.strip().upper().replace("△", "")
    if re.fullmatch(r"[A-Z]{2,8}", normalized):
        return True
    return bool(re.fullmatch(
        r"(?:ARC|CIRCLE|REGION|SEGMENT|LINE|RAY|POLYGON|TRIANGLE)[A-Z0-9_.:-]*",
        normalized,
    ))


def _is_polygon_pair(objects: list) -> bool:
    if len(objects) != 2:
        return False
    for value in objects:
        normalized = str(value).upper().replace("△", "").replace("TRIANGLE", "")
        if "∠" in normalized or len("".join(ch for ch in normalized if ch.isalpha())) != 3:
            return False
    return True


def _relation_score(relation: dict) -> tuple[int, float, int]:
    return (
        RELATION_PRIORITY.get(str(relation.get("type")), 0),
        float(relation.get("confidence") or 0),
        len(relation.get("objects") or []),
    )


if __name__ == "__main__":
    main()
