"""Script-side observations used to guide VLM semantic extraction."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import math
import re


class ScriptObservationExtractor:
    """Summarize weak script evidence before VLM semantic extraction.

    The VLM is responsible for reading the real diagram and extracting named
    geometry semantics. Script observations are weak hints plus structured
    text/metadata candidates that later code can normalize, convert, and verify.
    They are deliberately not GeoSym mathematical facts.
    """

    LABEL_RE = re.compile(r"\b[A-Z]\b")
    SEGMENT_RE = re.compile(r"\b([A-Z]{2})\b")
    NUMBER_RE = re.compile(r"(?<![A-Za-z])(?:\d+(?:\.\d+)?|\d+/\d+|sqrt\(?\d+\)?|\\sqrt\{?\d+\}?)(?![A-Za-z])", re.I)
    VALUE_PATTERN = r"(?:\d+(?:\.\d+)?|\d+/\d+|sqrt\(?\d+\)?|\\sqrt\{?\d+\}?|[a-z])"
    NUMERIC_VALUE_PATTERN = r"(?:\d+(?:\.\d+)?|\d+/\d+|sqrt\(?\d+\)?|\\sqrt\{?\d+\}?)"
    LENGTH_OBJECT_PATTERN = r"(?:\\overline\{([A-Z]{2})\}|\\bar\{([A-Z]{2})\}|segment\s+([A-Z]{2})|side\s+([A-Z]{2})|length\s+of\s+([A-Z]{2})|\b([A-Z]{2})\b)"
    LENGTH_RE = re.compile(rf"{LENGTH_OBJECT_PATTERN}\s*(?:=|is|equals|has length|length)\s*({VALUE_PATTERN})", re.I)
    ALT_LENGTH_RE = re.compile(rf"(?:length of|side|segment)\s+([A-Z]{{2}})\s*(?:is|=|equals)?\s*({VALUE_PATTERN})", re.I)
    LOGIC_LENGTH_RE = re.compile(r"\b(?:LengthOf|Length|Segment)\s*\(\s*(?:Line\s*\()?\s*([A-Z])\s*,\s*([A-Z])\s*\)?\s*\)\s*(?:,\s*({value})(?!\s*/))?".format(value=NUMERIC_VALUE_PATTERN), re.I)
    EQUALS_LENGTH_RE = re.compile(r"Equals\s*\(\s*(?:LengthOf|Length)\s*\(\s*(?:Line\s*\()?\s*([A-Z])\s*,\s*([A-Z])\s*\)?\s*\)\s*,\s*({value})\s*\)".format(value=VALUE_PATTERN), re.I)
    PARALLEL_RE = re.compile(r"\b([A-Z]{2})\s*(?:∥|\|\||\\parallel|parallel(?:\s+to)?)\s*([A-Z]{2})\b", re.I)
    LOGIC_PARALLEL_RE = re.compile(r"(?:Parallel|IsParallel)\s*\(\s*(?:Line\s*\()?\s*([A-Z])\s*,\s*([A-Z])\s*\)?\s*,\s*(?:Line\s*\()?\s*([A-Z])\s*,\s*([A-Z])\s*\)?\s*\)", re.I)
    PERP_RE = re.compile(r"\b([A-Z]{2})\s*(?:⊥|\\perp|perpendicular(?:\s+to)?)\s*([A-Z]{2})\b", re.I)
    LOGIC_PERP_RE = re.compile(r"(?:Perpendicular|IsPerpendicular)\s*\(\s*(?:Line\s*\()?\s*([A-Z])\s*,\s*([A-Z])\s*\)?\s*,\s*(?:Line\s*\()?\s*([A-Z])\s*,\s*([A-Z])\s*\)?\s*\)", re.I)
    RIGHT_ANGLE_RE = re.compile(r"(?:angle|∠|\\angle)\s*([A-Z]{3})\s*(?:=|is|measures)?\s*(?:90|90°|right\s+angle)", re.I)
    LOGIC_RIGHT_ANGLE_RE = re.compile(r"(?:RightAngle|IsRightAngle)\s*\(\s*([A-Z])\s*,\s*([A-Z])\s*,\s*([A-Z])\s*\)", re.I)
    ANGLE_VALUE_RE = re.compile(rf"(?<!tri)(?:\bangle|∠|\\angle)\s*([A-Z]{{3}})\s*(?:=|is|measures|measure(?:s)?(?:\s+is)?)?\s*({VALUE_PATTERN})(?:°|degrees?)?", re.I)
    LOGIC_ANGLE_RE = re.compile(r"\b(?:MeasureOfAngle|AngleMeasure|Angle)\s*\(\s*([A-Z])\s*,\s*([A-Z])\s*,\s*([A-Z])\s*\)\s*(?:,\s*({value}))?".format(value=NUMERIC_VALUE_PATTERN), re.I)
    EQUALS_ANGLE_RE = re.compile(r"Equals\s*\(\s*(?:MeasureOfAngle|AngleMeasure|Angle)\s*\(\s*([A-Z])\s*,\s*([A-Z])\s*,\s*([A-Z])\s*\)\s*,\s*({value})\s*\)".format(value=VALUE_PATTERN), re.I)
    AREA_RE = re.compile(rf"\b(?:area\s+of|area(?!\s*of|\s*\()\s*\(|area(?!of))\s*(?:triangle\s+|△)?([A-Z]{{3,4}}|[A-Z])\)?\s*(?:=|is|equals)?\s*({VALUE_PATTERN})?", re.I)
    LOGIC_AREA_RE = re.compile(r"\b(?:AreaOf|Area)\s*\(\s*(?:Triangle\s*\()?([A-Z])\s*,\s*([A-Z])\s*,\s*([A-Z])\)?\s*\)\s*(?:,\s*({value}))?".format(value=NUMERIC_VALUE_PATTERN), re.I)
    EQUALS_AREA_RE = re.compile(r"Equals\s*\(\s*(?:AreaOf|Area)\s*\(\s*(?:Triangle\s*\()?([A-Z])\s*,\s*([A-Z])\s*,\s*([A-Z])\)?\s*\)\s*,\s*({value})\s*\)".format(value=VALUE_PATTERN), re.I)
    PERIMETER_RE = re.compile(rf"(?:perimeter\s+of|perimeter)\s*(?:triangle\s+|△)?([A-Z]{{3,4}})\s*(?:=|is|equals)?\s*({VALUE_PATTERN})?", re.I)
    LOGIC_PERIMETER_RE = re.compile(r"\b(?:PerimeterOf|Perimeter)\s*\(\s*(?:Triangle\s*\()?([A-Z])\s*,\s*([A-Z])\s*,\s*([A-Z])\)?\s*\)\s*(?:,\s*({value}))?".format(value=NUMERIC_VALUE_PATTERN), re.I)
    EQUALS_PERIMETER_RE = re.compile(r"Equals\s*\(\s*(?:PerimeterOf|Perimeter)\s*\(\s*(?:Triangle\s*\()?([A-Z])\s*,\s*([A-Z])\s*,\s*([A-Z])\)?\s*\)\s*,\s*({value})\s*\)".format(value=VALUE_PATTERN), re.I)
    RATIO_RE = re.compile(rf"(?:ratio\s+of\s+)?([A-Z]{{2}})\s*[:/]\s*([A-Z]{{2}})\s*(?:=|is|equals)?\s*({VALUE_PATTERN}(?::|/){VALUE_PATTERN}|{VALUE_PATTERN})?", re.I)
    LOGIC_RATIO_RE = re.compile(r"(?:RatioOf|Ratio)\s*\(\s*(?:LengthOf\s*\()?([A-Z])\s*,\s*([A-Z])\)?\s*,\s*(?:LengthOf\s*\()?([A-Z])\s*,\s*([A-Z])\)?\s*\)\s*,?\s*({value}(?::|/){value}|{value})?".format(value=VALUE_PATTERN), re.I)
    TRIANGLE_RE = re.compile(r"(?:triangle|△)\s+([A-Z]{3})", re.I)
    RIGHT_TRIANGLE_RE = re.compile(r"right\s+triangle\s+([A-Z]{3})", re.I)
    CIRCLE_RE = re.compile(r"\bcircle\s+([A-Z])\b|\b([A-Z])\s+is\s+the\s+center\b", re.I)
    TARGET_ANGLE_RE = re.compile(r"(?:find|求|what is|determine).*?(?:angle|∠|\\angle)\s*([A-Z]{3})", re.I)
    TARGET_AREA_RE = re.compile(r"(?:find|求|what is|determine).*?(?:area\s+of|area)\s*(?:triangle\s+|△)?([A-Z]{3,4}|[A-Z])?", re.I)
    TARGET_PERIMETER_RE = re.compile(r"(?:find|求|what is|determine).*?(?:perimeter\s+of|perimeter)\s*(?:triangle\s+|△)?([A-Z]{3,4})?", re.I)
    TARGET_RATIO_RE = re.compile(r"(?:find|求|what is|determine).*?(?:ratio\s+of\s+)?([A-Z]{2})\s*[:/]\s*([A-Z]{2})", re.I)
    TARGET_RE = re.compile(r"(?:find|求|what is|determine).*?\b([A-Z]{2}|x)\b", re.I)

    def extract(
        self,
        *,
        visual_state: Dict[str, Any],
        problem_text: str = "",
        solution_text: str | None = None,
        raw_record: Dict[str, Any] | None = None,
        answer: Any = None,
    ) -> Dict[str, Any]:
        lines = list(visual_state.get("lines") or [])
        circles = list(visual_state.get("circles") or [])
        fill_regions = list(visual_state.get("fill_regions") or [])
        markers = list(visual_state.get("markers") or [])
        text_regions = list(visual_state.get("text_regions") or [])
        topology = visual_state.get("topology") or {}
        text_sources = self._text_sources(problem_text, solution_text, raw_record)
        combined_text = "\n".join(source["text"] for source in text_sources if source.get("text"))
        text_hints = self._text_hints(combined_text, answer, text_sources)
        semantic_candidates = self._semantic_candidates(text_sources, text_hints)
        visual_structure_candidates = self._visual_structure_candidates(visual_state)
        named_object_proposals = self._named_object_proposals(text_hints, semantic_candidates, visual_structure_candidates)
        grounding_tasks = self._grounding_tasks(visual_state, semantic_candidates, text_hints, visual_structure_candidates)
        return {
            "source": "script",
            "extraction_strategy": {
                "mode": "vlm_first_semantic_extraction",
                "vlm_role": "read the original image/problem text and extract named objects, relations, labels, target, and diagram notes",
                "script_role": "provide weak hints, normalize VLM output, map relations to current GeoSym builder capability, compile instructions, and verify",
                "primitive_role": "optional weak visual hints for review/alignment; never the primary source of geometric truth",
                "hard_rule": "do not require a VLM relation to bind to OpenCV primitive ids when the named geometry is clear from the original image/text",
            },
            "image_size": visual_state.get("image_size"),
            "counts": {
                "lines": len(lines),
                "circles": len(circles),
                "fill_regions": len(fill_regions),
                "markers": len(markers),
                "text_regions": len(text_regions),
                "line_intersections": len(topology.get("line_intersections") or []),
                "circle_line_hits": len(topology.get("circle_line_hits") or []),
            },
            "dominant_lines": self._dominant_lines(lines),
            "fill_region_hints": [dict(item) for item in fill_regions[:40]],
            "relation_hints": self._relation_hints(lines),
            "marker_hints": [dict(item) for item in markers[:40]],
            "topology_hints": self._topology_hints(topology),
            "visual_structure_candidates": visual_structure_candidates,
            "named_object_proposals": named_object_proposals,
            "text_hints": text_hints,
            "text_sources": text_sources,
            "semantic_candidates": semantic_candidates,
            "grounding_tasks": grounding_tasks,
            "engine_build_suggestions": self._engine_build_suggestions(semantic_candidates),
            "candidate_ids": self._candidate_ids(visual_state),
            "guidance": {
                "vlm_task": "extract named geometry semantics from the original image/text first; use script candidates as optional hints, not as the source of truth",
                "script_task_after_vlm": "normalize object names and relation schema, judge builder capability, compile supported instructions, and verify consistency",
                "do_not_do": "do not use pixel coordinates as GeoSym mathematical coordinates",
                "visual_structure_policy": "vp_*/vs_*/va_* candidates are weak auxiliary evidence ids only; use them when helpful for review/alignment, never as required truth or SymPy coordinates",
                "named_object_policy": "named_object_proposals are symbolic hints from text/script; VLM may confirm, correct, reject, or add named objects read directly from the image",
                "builder_instruction_rule": "VLM stage 1 does not output build_instructions; script/stage 2 may emit only facts supported by current GeoSym builder capability",
            },
        }

    def _dominant_lines(self, lines: List[Dict[str, Any]], max_items: int = 12) -> List[Dict[str, Any]]:
        out = []
        for line in sorted(lines, key=lambda item: item.get("length_px", 0), reverse=True)[:max_items]:
            angle = float(line.get("angle_deg") or 0.0)
            out.append({
                "id": line.get("id"),
                "length_px": line.get("length_px"),
                "angle_deg": line.get("angle_deg"),
                "orientation": self._orientation(angle),
            })
        return out

    def _relation_hints(self, lines: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        parallel = []
        perpendicular = []
        comparable_lengths = []
        top = sorted(lines, key=lambda item: item.get("length_px", 0), reverse=True)[:30]
        for i, a in enumerate(top):
            for b in top[i + 1 :]:
                diff = self._angle_diff(float(a.get("angle_deg") or 0.0), float(b.get("angle_deg") or 0.0))
                pair = {"line_ids": [a.get("id"), b.get("id")], "angle_delta_deg": round(diff, 3)}
                if diff <= 5.0:
                    parallel.append(pair)
                if abs(diff - 90.0) <= 7.0:
                    perpendicular.append(pair)
                alen = float(a.get("length_px") or 0.0)
                blen = float(b.get("length_px") or 0.0)
                if min(alen, blen) > 0:
                    ratio = max(alen, blen) / min(alen, blen)
                    if ratio <= 1.08:
                        comparable_lengths.append({
                            "line_ids": [a.get("id"), b.get("id")],
                            "length_ratio_px": round(ratio, 3),
                        })
                if len(parallel) >= 20 and len(perpendicular) >= 20 and len(comparable_lengths) >= 20:
                    break
            if len(parallel) >= 20 and len(perpendicular) >= 20 and len(comparable_lengths) >= 20:
                break
        return {
            "parallel_line_candidates": parallel[:20],
            "perpendicular_line_candidates": perpendicular[:20],
            "similar_length_line_candidates": comparable_lengths[:20],
        }

    def _topology_hints(self, topology: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "visible_intersections": [
                {"id": item.get("id"), "line_ids": item.get("line_ids")}
                for item in (topology.get("line_intersections") or [])[:30]
            ],
            "circle_line_contacts": [
                {"id": item.get("id"), "circle_id": item.get("circle_id"), "line_id": item.get("line_id")}
                for item in (topology.get("circle_line_hits") or [])[:30]
            ],
        }

    def _visual_structure_candidates(self, visual_state: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate low-level image primitives into bindable evidence objects.

        The output is intentionally visual-only. Candidate ids such as ``vp_0``
        or ``vs_0`` are for VLM grounding and review overlays, not for GeoSym
        mathematical coordinates.
        """
        lines = list(visual_state.get("lines") or [])
        text_regions = list(visual_state.get("text_regions") or [])
        topology = visual_state.get("topology") or {}
        point_candidates: List[Dict[str, Any]] = []
        segment_candidates: List[Dict[str, Any]] = []
        angle_candidates: List[Dict[str, Any]] = []
        line_endpoint_points: Dict[Tuple[str, str], str] = {}
        point_seen: set[Tuple[str, str]] = set()

        def add_point(source: str, visual_id: str, xy: Any, line_ids: List[str] | None = None) -> str:
            key = (source, visual_id)
            for item in point_candidates:
                if (item.get("source"), item.get("visual_id")) == key:
                    return str(item.get("id") or "")
            pid = f"vp_{len(point_candidates)}"
            point_seen.add(key)
            point_candidates.append({
                "id": pid,
                "kind": "point_candidate",
                "source": source,
                "visual_id": visual_id,
                "xy_px": xy,
                "line_ids": line_ids or [],
                "coordinate_policy": "pixel xy is visual evidence only; do not use as GeoSym coordinate",
            })
            return pid

        for line in sorted(lines, key=lambda item: item.get("length_px", 0), reverse=True)[:60]:
            lid = str(line.get("id") or "")
            if not lid:
                continue
            endpoint_ids: List[str] = []
            for name in ["p1", "p2"]:
                if line.get(name) is None:
                    continue
                pid = add_point("line_endpoint", f"{lid}_{name}", line.get(name), [lid])
                line_endpoint_points[(lid, name)] = pid
                endpoint_ids.append(pid)
            segment_candidates.append({
                "id": f"vs_{len(segment_candidates)}",
                "kind": "segment_candidate",
                "visual_id": lid,
                "endpoint_candidate_ids": endpoint_ids,
                "p1_px": line.get("p1"),
                "p2_px": line.get("p2"),
                "length_px": line.get("length_px"),
                "angle_deg": line.get("angle_deg"),
                "orientation": self._orientation(float(line.get("angle_deg") or 0.0)),
                "coordinate_policy": "visual segment evidence only; named GeoSym segment must come from text/VLM binding",
            })

        for item in (topology.get("endpoints") or [])[:120]:
            visual_id = str(item.get("id") or "")
            if not visual_id:
                continue
            add_point("topology_endpoint", visual_id, item.get("xy"), [str(item.get("line_id") or "")])

        for item in (topology.get("line_intersections") or [])[:80]:
            visual_id = str(item.get("id") or "")
            line_ids = [str(value) for value in (item.get("line_ids") or []) if value]
            if not visual_id:
                continue
            pid = add_point("line_intersection", visual_id, item.get("xy"), line_ids)
            if len(line_ids) >= 2:
                angle_candidates.append({
                    "id": f"va_{len(angle_candidates)}",
                    "kind": "angle_candidate",
                    "vertex_point_candidate_id": pid,
                    "intersection_visual_id": visual_id,
                    "line_ids": line_ids[:2],
                    "xy_px": item.get("xy"),
                    "coordinate_policy": "visual angle evidence only; named angle must come from text/VLM binding",
                })

        label_region_candidates = [
            {
                "id": f"vlabel_{idx}",
                "kind": "label_region_candidate",
                "visual_id": item.get("id"),
                "bbox": item.get("bbox"),
                "area_px": item.get("area_px"),
                "coordinate_policy": "OCR/label binding evidence only",
            }
            for idx, item in enumerate(text_regions[:80])
        ]
        return {
            "point_candidates": point_candidates[:160],
            "segment_candidates": segment_candidates[:80],
            "angle_candidates": angle_candidates[:80],
            "label_region_candidates": label_region_candidates,
            "summary": {
                "num_point_candidates": len(point_candidates[:160]),
                "num_segment_candidates": len(segment_candidates[:80]),
                "num_angle_candidates": len(angle_candidates[:80]),
                "num_label_region_candidates": len(label_region_candidates),
            },
            "coordinate_policy": "all candidates are visual evidence only; canonical GeoSym coordinates must be rebuilt by solver/builder from symbolic constraints",
        }

    def _text_sources(
        self,
        problem_text: str,
        solution_text: str | None,
        raw_record: Dict[str, Any] | None,
    ) -> List[Dict[str, str]]:
        sources: List[Dict[str, str]] = []

        def add(name: str, value: Any, confidence: str) -> None:
            text = self._flatten_text(value)
            if not text:
                return
            if any(item.get("text") == text and item.get("name") == name for item in sources):
                return
            sources.append({"name": name, "confidence": confidence, "text": text[:5000]})

        add("problem_text", problem_text, "primary_problem_statement")
        add("solution_text", solution_text, "solution_or_explanation")
        raw = raw_record or {}
        for key in [
            "diagram_logic_form",
            "logic_form",
            "compact_logic_form",
            "precise_value",
            "dissolved_text",
            "annotat_text",
            "solution",
            "choices",
        ]:
            if key in raw:
                add(f"raw.{key}", raw.get(key), "structured_or_auxiliary_dataset_field")
        for key, value in raw.items():
            low = str(key).lower()
            if key in {"images", "problem", "answer"}:
                continue
            if any(token in low for token in ["logic", "formula", "relation", "diagram", "text", "solution"]):
                add(f"raw.{key}", value, "structured_or_auxiliary_dataset_field")
        return sources

    def _flatten_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, list):
            parts = [self._flatten_text(item) for item in value]
            return "\n".join(part for part in parts if part)
        if isinstance(value, dict):
            parts = []
            for key, item in value.items():
                text = self._flatten_text(item)
                if text:
                    parts.append(f"{key}: {text}")
            return "\n".join(parts)
        return str(value).strip()

    def _text_hints(self, problem_text: str, answer: Any, text_sources: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
        text = problem_text or ""
        labels = sorted(set(self.LABEL_RE.findall(text)))
        segments = sorted(set(self.SEGMENT_RE.findall(text)))
        numbers = self.NUMBER_RE.findall(text)
        length_claims = []
        angle_claims = []
        area_claims = []
        perimeter_claims = []
        ratio_claims = []
        triangle_claims = []
        for source_name, source_text in self._iter_source_texts(text_sources, fallback=text):
            for match in self.LENGTH_RE.finditer(source_text):
                if self._match_in_ratio_context(source_text, match.start(), match.end()):
                    continue
                obj = self._first_match_group(match, 1, 7)
                value = match.group(7)
                if obj and value:
                    length_claims.append({"object": obj.upper(), "value": self._normalize_value(value), "source_text": source_name})
            for match in self.EQUALS_LENGTH_RE.finditer(source_text):
                length_claims.append({"object": f"{match.group(1).upper()}{match.group(2).upper()}", "value": self._normalize_value(match.group(3)), "source_text": source_name})
            for match in self.LOGIC_LENGTH_RE.finditer(source_text):
                if self._match_in_ratio_context(source_text, match.start(), match.end()):
                    continue
                value = self._normalize_value(match.group(3)) if match.group(3) else None
                if value:
                    length_claims.append({"object": f"{match.group(1).upper()}{match.group(2).upper()}", "value": value, "source_text": source_name})
            for match in self.EQUALS_ANGLE_RE.finditer(source_text):
                value = self._normalize_value(match.group(4))
                if not self._is_right_angle_value(value):
                    angle_claims.append({"angle": f"{match.group(1).upper()}{match.group(2).upper()}{match.group(3).upper()}", "value": value, "source_text": source_name})
            for match in self.LOGIC_ANGLE_RE.finditer(source_text):
                value = self._normalize_value(match.group(4)) if match.group(4) else None
                if value and not self._is_right_angle_value(value):
                    angle_claims.append({"angle": f"{match.group(1).upper()}{match.group(2).upper()}{match.group(3).upper()}", "value": value, "source_text": source_name})
            for match in self.ANGLE_VALUE_RE.finditer(source_text):
                value = self._normalize_value(match.group(2))
                if value and value not in {"right"} and not self._is_right_angle_value(value):
                    angle_claims.append({"angle": match.group(1).upper(), "value": value, "source_text": source_name})
            for match in self.EQUALS_AREA_RE.finditer(source_text):
                area_claims.append({"object": f"{match.group(1).upper()}{match.group(2).upper()}{match.group(3).upper()}", "value": self._normalize_value(match.group(4)), "source_text": source_name})
            for match in self.LOGIC_AREA_RE.finditer(source_text):
                value = self._normalize_value(match.group(4)) if match.group(4) else None
                area_claims.append({"object": f"{match.group(1).upper()}{match.group(2).upper()}{match.group(3).upper()}", "value": value, "source_text": source_name})
            for match in self.AREA_RE.finditer(source_text):
                obj = (match.group(1) or "").upper()
                value = self._normalize_value(match.group(2)) if match.group(2) else None
                if obj:
                    area_claims.append({"object": obj, "value": value, "source_text": source_name})
            for match in self.EQUALS_PERIMETER_RE.finditer(source_text):
                perimeter_claims.append({"object": f"{match.group(1).upper()}{match.group(2).upper()}{match.group(3).upper()}", "value": self._normalize_value(match.group(4)), "source_text": source_name})
            for match in self.LOGIC_PERIMETER_RE.finditer(source_text):
                value = self._normalize_value(match.group(4)) if match.group(4) else None
                perimeter_claims.append({"object": f"{match.group(1).upper()}{match.group(2).upper()}{match.group(3).upper()}", "value": value, "source_text": source_name})
            for match in self.PERIMETER_RE.finditer(source_text):
                obj = (match.group(1) or "").upper()
                value = self._normalize_value(match.group(2)) if match.group(2) else None
                if obj:
                    perimeter_claims.append({"object": obj, "value": value, "source_text": source_name})
            for match in self.LOGIC_RATIO_RE.finditer(source_text):
                value = self._normalize_value(match.group(5)) if match.group(5) else None
                ratio_claims.append({"objects": [f"{match.group(1).upper()}{match.group(2).upper()}", f"{match.group(3).upper()}{match.group(4).upper()}"], "value": value, "source_text": source_name})
            for match in self.RATIO_RE.finditer(source_text):
                value = self._normalize_value(match.group(3)) if match.group(3) else None
                ratio_claims.append({"objects": [match.group(1).upper(), match.group(2).upper()], "value": value, "source_text": source_name})
            for match in self.TRIANGLE_RE.finditer(source_text):
                triangle_claims.append({"object": match.group(1).upper(), "source_text": source_name})
        seen_lengths = {(claim["object"], claim["value"]) for claim in length_claims}
        for source_name, source_text in self._iter_source_texts(text_sources, fallback=text):
            for match in self.ALT_LENGTH_RE.finditer(source_text):
                if self._match_in_ratio_context(source_text, match.start(), match.end()):
                    continue
                claim = {"object": match.group(1).upper(), "value": self._normalize_value(match.group(2)), "source_text": source_name}
                key = (claim["object"], claim["value"])
                if key not in seen_lengths:
                    length_claims.append(claim)
                    seen_lengths.add(key)
        deduped_angles = []
        seen_angles = set()
        for claim in angle_claims:
            key = (claim.get("angle"), claim.get("value"))
            if key in seen_angles:
                continue
            seen_angles.add(key)
            deduped_angles.append(claim)
        area_claims = self._dedupe_claims(area_claims, ["object", "value"])
        perimeter_claims = self._dedupe_claims(perimeter_claims, ["object", "value"])
        ratio_claims = self._dedupe_claims(ratio_claims, ["objects", "value"])
        triangle_claims = self._dedupe_claims(triangle_claims, ["object"])
        target_angle_match = self.TARGET_ANGLE_RE.search(text)
        target_area_match = self.TARGET_AREA_RE.search(text)
        target_perimeter_match = self.TARGET_PERIMETER_RE.search(text)
        target_ratio_match = self.TARGET_RATIO_RE.search(text)
        target_match = self.TARGET_RE.search(text)
        return {
            "labels_mentioned": labels[:40],
            "segments_mentioned": segments[:40],
            "numbers_mentioned": numbers[:40],
            "length_claims": length_claims[:40],
            "angle_claims": deduped_angles[:40],
            "area_claims": area_claims[:40],
            "perimeter_claims": perimeter_claims[:40],
            "ratio_claims": ratio_claims[:40],
            "triangle_claims": triangle_claims[:40],
            "target_angle_hint": target_angle_match.group(1).upper() if target_angle_match else None,
            "target_area_hint": (target_area_match.group(1).upper() if target_area_match and target_area_match.group(1) else "area") if target_area_match else None,
            "target_perimeter_hint": (target_perimeter_match.group(1).upper() if target_perimeter_match and target_perimeter_match.group(1) else "perimeter") if target_perimeter_match else None,
            "target_ratio_hint": [target_ratio_match.group(1).upper(), target_ratio_match.group(2).upper()] if target_ratio_match else None,
            "target_hint": target_match.group(1).upper() if target_match else None,
            "known_answer": answer,
            "num_text_sources": len(text_sources or []),
        }

    def _semantic_candidates(self, text_sources: List[Dict[str, str]], text_hints: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        relation_candidates: List[Dict[str, Any]] = []
        target_candidates: List[Dict[str, Any]] = []
        seen_relations: set[Tuple[str, str, str, str]] = set()
        for claim in text_hints.get("length_claims") or []:
            obj = str(claim.get("object") or "").upper()
            if len(obj) >= 2:
                key = ("length", obj[:2], str(claim.get("value") or ""), "")
                if key in seen_relations:
                    continue
                seen_relations.add(key)
                relation_candidates.append({
                    "id": f"sc_len_{len([x for x in relation_candidates if x.get('type') == 'length'])}",
                    "type": "length",
                    "objects": [obj[0], obj[1]],
                    "object": obj[:2],
                    "value": claim.get("value"),
                    "source": "text_regex",
                    "source_text": claim.get("source_text") or "unknown",
                    "needs_vlm_confirmation": True,
                    "grounding_required": "bind segment to a vl_* candidate if visible",
                })
        for claim in text_hints.get("angle_claims") or []:
            angle = str(claim.get("angle") or "").upper()
            value = claim.get("value")
            if len(angle) == 3 and value not in {None, ""}:
                key = ("angle", angle, str(value), "")
                if key in seen_relations:
                    continue
                seen_relations.add(key)
                relation_candidates.append({
                    "id": f"sc_angle_{len([x for x in relation_candidates if x.get('type') == 'angle'])}",
                    "type": "angle",
                    "objects": [angle],
                    "object": angle,
                    "angle": angle,
                    "vertex": angle[1],
                    "value": value,
                    "source": "text_regex",
                    "source_text": claim.get("source_text") or "unknown",
                    "needs_vlm_confirmation": True,
                    "grounding_required": "bind both rays/segments to vl_* candidates if visible",
                    "engine_note": "current canonical builder keeps general angle values as theorem-level relations",
                })
        for claim in text_hints.get("area_claims") or []:
            obj = str(claim.get("object") or "").upper()
            if obj:
                self._append_relation_candidate(relation_candidates, seen_relations, "area", [obj], claim.get("source_text") or "unknown", {
                    "object": obj,
                    "value": claim.get("value"),
                    "engine_note": "current canonical builder keeps area relations as theorem-level constraints",
                })
        for claim in text_hints.get("perimeter_claims") or []:
            obj = str(claim.get("object") or "").upper()
            if obj:
                self._append_relation_candidate(relation_candidates, seen_relations, "perimeter", [obj], claim.get("source_text") or "unknown", {
                    "object": obj,
                    "value": claim.get("value"),
                    "engine_note": "current canonical builder keeps perimeter relations as theorem-level constraints",
                })
        for claim in text_hints.get("ratio_claims") or []:
            objects = [str(item).upper() for item in (claim.get("objects") or []) if str(item)]
            if len(objects) >= 2:
                self._append_relation_candidate(relation_candidates, seen_relations, "ratio", objects[:2], claim.get("source_text") or "unknown", {
                    "value": claim.get("value"),
                    "engine_note": "current canonical builder keeps ratio relations as theorem-level constraints unless reduced to explicit lengths",
                })
        for claim in text_hints.get("triangle_claims") or []:
            obj = str(claim.get("object") or "").upper()
            if len(obj) == 3:
                self._append_relation_candidate(relation_candidates, seen_relations, "triangle", list(obj), claim.get("source_text") or "unknown", {
                    "object": obj,
                    "engine_note": "triangle declaration is semantic scaffolding; convert only explicit supported constraints",
                })
        for source_name, text in self._iter_source_texts(text_sources):
            for match in self.PARALLEL_RE.finditer(text):
                self._append_relation_candidate(relation_candidates, seen_relations, "parallel", [match.group(1).upper(), match.group(2).upper()], source_name)
            for match in self.LOGIC_PARALLEL_RE.finditer(text):
                self._append_relation_candidate(relation_candidates, seen_relations, "parallel", [f"{match.group(1).upper()}{match.group(2).upper()}", f"{match.group(3).upper()}{match.group(4).upper()}"], source_name)
            for match in self.PERP_RE.finditer(text):
                self._append_relation_candidate(relation_candidates, seen_relations, "perpendicular", [match.group(1).upper(), match.group(2).upper()], source_name)
            for match in self.LOGIC_PERP_RE.finditer(text):
                self._append_relation_candidate(relation_candidates, seen_relations, "perpendicular", [f"{match.group(1).upper()}{match.group(2).upper()}", f"{match.group(3).upper()}{match.group(4).upper()}"], source_name)
            for match in self.RIGHT_ANGLE_RE.finditer(text):
                self._append_right_angle_candidate(relation_candidates, seen_relations, match.group(1).upper(), source_name)
            for match in self.LOGIC_RIGHT_ANGLE_RE.finditer(text):
                self._append_right_angle_candidate(relation_candidates, seen_relations, f"{match.group(1).upper()}{match.group(2).upper()}{match.group(3).upper()}", source_name)
            for match in self.RIGHT_TRIANGLE_RE.finditer(text):
                tri = match.group(1).upper()
                if len(tri) == 3:
                    self._append_relation_candidate(relation_candidates, seen_relations, "right_triangle", list(tri), source_name, {
                        "engine_note": "convert to perpendicular only if right-angle vertex is explicit in text/image",
                    })
            for match in self.CIRCLE_RE.finditer(text):
                center = (match.group(1) or match.group(2) or "").upper()
                if center:
                    self._append_relation_candidate(relation_candidates, seen_relations, "circle", [center], source_name, {
                        "engine_note": "current canonical builder cannot build circle theorem state",
                    })
        target_hint = text_hints.get("target_hint")
        target_angle_hint = text_hints.get("target_angle_hint")
        target_area_hint = text_hints.get("target_area_hint")
        target_perimeter_hint = text_hints.get("target_perimeter_hint")
        target_ratio_hint = text_hints.get("target_ratio_hint")
        if target_angle_hint:
            target_candidates.append({
                "id": f"sc_target_{len(target_candidates)}",
                "type": "angle",
                "object": str(target_angle_hint).upper(),
                "value_symbol": "x",
                "source": "text_regex",
                "needs_vlm_confirmation": True,
            })
        if target_area_hint:
            target_candidates.append({
                "id": f"sc_target_{len(target_candidates)}",
                "type": "area",
                "object": str(target_area_hint).upper(),
                "value_symbol": "x",
                "source": "text_regex",
                "needs_vlm_confirmation": True,
            })
        if target_perimeter_hint:
            target_candidates.append({
                "id": f"sc_target_{len(target_candidates)}",
                "type": "perimeter",
                "object": str(target_perimeter_hint).upper(),
                "value_symbol": "x",
                "source": "text_regex",
                "needs_vlm_confirmation": True,
            })
        if target_ratio_hint:
            target_candidates.append({
                "id": f"sc_target_{len(target_candidates)}",
                "type": "ratio",
                "object": ":".join(str(item).upper() for item in target_ratio_hint),
                "objects": [str(item).upper() for item in target_ratio_hint],
                "value_symbol": "x",
                "source": "text_regex",
                "needs_vlm_confirmation": True,
            })
        if target_hint and not (target_angle_hint or target_area_hint or target_perimeter_hint or target_ratio_hint):
            target_candidates.append({
                "id": f"sc_target_{len(target_candidates)}",
                "type": "length" if len(str(target_hint)) == 2 else "unknown",
                "object": str(target_hint).upper(),
                "value_symbol": "x",
                "source": "text_regex",
                "needs_vlm_confirmation": True,
            })
        return {
            "relation_candidates": relation_candidates[:60],
            "target_candidates": target_candidates[:10],
        }

    def _named_object_proposals(
        self,
        text_hints: Dict[str, Any],
        semantic_candidates: Dict[str, List[Dict[str, Any]]],
        visual_structure_candidates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create symbolic object proposals that the VLM can bind to pixels.

        These proposals bridge text/script symbols such as A, AB, and ABC with
        visual evidence candidates. They are not mathematical coordinates.
        """
        proposals: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str]] = set()

        def add(kind: str, object_id: str, source: str, candidate_ids: List[str] | None = None) -> None:
            obj = str(object_id or "").upper()
            if not obj:
                return
            key = (kind, obj)
            if key in seen:
                return
            seen.add(key)
            proposals.append({
                "id": f"no_{len(proposals)}",
                "kind": kind,
                "object": obj,
                "source": source,
                "candidate_ids": candidate_ids or [],
                "binding_status": "unbound",
                "required_visual_evidence_kind": self._proposal_visual_kind(kind),
                "visual_choices": self._proposal_visual_choices(kind, visual_structure_candidates),
                "coordinate_policy": "symbolic object proposal only; bind to visual evidence, then rebuild canonical GeoSym coordinates through solver/builder",
            })

        for label in text_hints.get("labels_mentioned") or []:
            if len(str(label)) == 1:
                add("point_label", str(label), "text_hints.labels_mentioned")
        for segment in text_hints.get("segments_mentioned") or []:
            if len(str(segment)) == 2:
                seg = str(segment).upper()
                add("point_label", seg[0], "text_hints.segments_mentioned")
                add("point_label", seg[1], "text_hints.segments_mentioned")
                add("segment", seg, "text_hints.segments_mentioned")

        for candidate in (semantic_candidates.get("relation_candidates") or []) + (semantic_candidates.get("target_candidates") or []):
            cid = str(candidate.get("id") or "")
            objects = []
            if candidate.get("object"):
                objects.append(str(candidate.get("object")))
            objects.extend(str(item) for item in (candidate.get("objects") or []) if str(item))
            if candidate.get("angle"):
                objects.append(str(candidate.get("angle")))
            for obj in objects:
                clean = str(obj).upper()
                if len(clean) == 1 and clean.isalpha():
                    add("point_label", clean, f"candidate:{cid}", [cid] if cid else [])
                elif len(clean) == 2 and clean.isalpha():
                    add("point_label", clean[0], f"candidate:{cid}", [cid] if cid else [])
                    add("point_label", clean[1], f"candidate:{cid}", [cid] if cid else [])
                    add("segment", clean, f"candidate:{cid}", [cid] if cid else [])
                elif len(clean) == 3 and clean.isalpha():
                    add("angle_or_triangle", clean, f"candidate:{cid}", [cid] if cid else [])
                    for point in clean:
                        add("point_label", point, f"candidate:{cid}", [cid] if cid else [])
                    add("segment", clean[0] + clean[1], f"candidate:{cid}", [cid] if cid else [])
                    add("segment", clean[1] + clean[2], f"candidate:{cid}", [cid] if cid else [])
                elif len(clean) >= 4 and clean.isalpha():
                    add("polygon", clean, f"candidate:{cid}", [cid] if cid else [])
                    for idx, point in enumerate(clean):
                        add("point_label", point, f"candidate:{cid}", [cid] if cid else [])
                        add("segment", point + clean[(idx + 1) % len(clean)], f"candidate:{cid}", [cid] if cid else [])

        return {
            "items": proposals[:120],
            "summary": {
                "num_items": len(proposals[:120]),
                "num_point_labels": sum(1 for item in proposals[:120] if item.get("kind") == "point_label"),
                "num_segments": sum(1 for item in proposals[:120] if item.get("kind") == "segment"),
                "num_angle_or_triangle": sum(1 for item in proposals[:120] if item.get("kind") == "angle_or_triangle"),
                "num_polygons": sum(1 for item in proposals[:120] if item.get("kind") == "polygon"),
            },
            "vlm_instruction": (
                "First bind named objects to visual evidence. Use point labels A/B and segments AB/CD "
                "from these proposals when confirming relations. Do not replace them with generic objects."
            ),
        }

    def _proposal_visual_kind(self, kind: str) -> str:
        if kind == "point_label":
            return "label_region_candidate or point_candidate"
        if kind == "segment":
            return "segment_candidate or line"
        if kind == "angle_or_triangle":
            return "angle_candidate plus supporting segments"
        if kind == "polygon":
            return "supporting segment_candidates"
        return "visual evidence"

    def _proposal_visual_choices(self, kind: str, visual_structure: Dict[str, Any], limit: int = 12) -> List[Dict[str, Any]]:
        if kind == "point_label":
            return self._label_region_choices(visual_structure, limit=limit) + [
                {
                    "visual_id": item.get("id"),
                    "kind": "point_candidate",
                    "source": item.get("source"),
                    "visual_source_id": item.get("visual_id"),
                    "line_ids": item.get("line_ids") or [],
                    "coordinate_policy": "point candidate is visual evidence only",
                }
                for item in (visual_structure.get("point_candidates") or [])[:limit]
            ]
        if kind == "segment":
            return self._segment_choices(visual_structure, limit=limit)
        if kind == "angle_or_triangle":
            return self._angle_choices(visual_structure, limit=limit) + self._segment_choices(visual_structure, limit=limit)
        if kind == "polygon":
            return self._segment_choices(visual_structure, limit=limit)
        return []

    def _grounding_tasks(
        self,
        visual_state: Dict[str, Any],
        semantic_candidates: Dict[str, List[Dict[str, Any]]],
        text_hints: Dict[str, Any],
        visual_structure_candidates: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Create script-side visual binding tasks for the VLM.

        The tasks use pixel primitives only as evidence ids. They are not
        mathematical coordinates and should never be copied into GeoSym point
        coordinates.
        """
        lines = list(visual_state.get("lines") or [])
        circles = list(visual_state.get("circles") or [])
        text_regions = list(visual_state.get("text_regions") or [])
        topology = visual_state.get("topology") or {}
        visual_structure = visual_structure_candidates or {}
        relation_candidates = semantic_candidates.get("relation_candidates") or []
        target_candidates = semantic_candidates.get("target_candidates") or []
        tasks: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str]] = set()

        for label in text_hints.get("labels_mentioned") or []:
            self._append_grounding_task(
                tasks,
                seen,
                task_type="label_to_text_region",
                semantic_ref=str(label).upper(),
                candidate_id=None,
                candidate_type="label",
                visual_choices=self._label_region_choices(visual_structure) + self._nearest_text_choices(text_regions),
                instruction="Bind this point/label symbol to a visible text region if present.",
            )

        for candidate in relation_candidates:
            cid = str(candidate.get("id") or "")
            ctype = str(candidate.get("type") or "")
            if ctype == "length":
                obj = str(candidate.get("object") or "").upper()
                if len(obj) >= 2:
                    self._append_grounding_task(
                        tasks,
                        seen,
                        task_type="segment_to_line",
                        semantic_ref=obj[:2],
                        candidate_id=cid,
                        candidate_type=ctype,
                        visual_choices=self._segment_choices(visual_structure) + self._line_choices(lines),
                        instruction="Choose the visible segment candidate corresponding to this named segment, or mark absent/uncertain.",
                    )
            elif ctype in {"parallel", "perpendicular", "right_angle"}:
                for line in [str(item).upper() for item in candidate.get("objects") or [] if len(str(item)) >= 2]:
                    self._append_grounding_task(
                        tasks,
                        seen,
                        task_type="segment_to_line",
                        semantic_ref=line[:2],
                        candidate_id=cid,
                        candidate_type=ctype,
                        visual_choices=self._segment_choices(visual_structure) + self._line_choices(lines),
                        instruction="Bind each named line/segment to a visual segment candidate before confirming the relation.",
                    )
                if ctype in {"perpendicular", "right_angle"}:
                    self._append_grounding_task(
                        tasks,
                        seen,
                        task_type="relation_to_topology",
                        semantic_ref="|".join(str(item).upper() for item in candidate.get("objects") or []),
                        candidate_id=cid,
                        candidate_type=ctype,
                        visual_choices=self._angle_choices(visual_structure) + self._intersection_choices(topology),
                        instruction="Check whether the two lines visibly intersect at the claimed right-angle vertex.",
                    )
            elif ctype == "circle":
                for center in [str(item).upper() for item in candidate.get("objects") or [] if str(item)]:
                    self._append_grounding_task(
                        tasks,
                        seen,
                        task_type="circle_center_to_circle",
                        semantic_ref=center,
                        candidate_id=cid,
                        candidate_type=ctype,
                        visual_choices=self._circle_choices(circles),
                        instruction="Bind the named center/circle evidence if visible; keep theorem-only if not buildable.",
                    )
            elif ctype in {"angle", "area", "perimeter", "ratio", "triangle", "right_triangle", "similar", "congruent"}:
                ref = str(candidate.get("object") or "|".join(str(item) for item in candidate.get("objects") or [])).upper()
                self._append_grounding_task(
                    tasks,
                    seen,
                    task_type="semantic_object_to_visual_support",
                    semantic_ref=ref,
                    candidate_id=cid,
                    candidate_type=ctype,
                    visual_choices=self._segment_choices(visual_structure, limit=12) + self._angle_choices(visual_structure, limit=8) + self._circle_choices(circles, limit=6),
                    instruction="Identify visual support ids for this theorem-level object; do not convert it into fake coordinates.",
                )

        for target in target_candidates:
            ref = str(target.get("object") or "").upper()
            if not ref:
                continue
            choices = self._segment_choices(visual_structure, limit=12) + self._line_choices(lines, limit=12) if len(ref) == 2 else self._segment_choices(visual_structure, limit=12) + self._angle_choices(visual_structure, limit=8) + self._circle_choices(circles, limit=6)
            self._append_grounding_task(
                tasks,
                seen,
                task_type="target_to_visual_support",
                semantic_ref=ref,
                candidate_id=str(target.get("id") or ""),
                candidate_type=f"target:{target.get('type') or 'unknown'}",
                visual_choices=choices,
                instruction="Bind the requested target to visible support if possible; preserve target even when visual binding is uncertain.",
            )

        return tasks[:120]

    def _append_grounding_task(
        self,
        tasks: List[Dict[str, Any]],
        seen: set[Tuple[str, str, str]],
        *,
        task_type: str,
        semantic_ref: str,
        candidate_id: str | None,
        candidate_type: str,
        visual_choices: List[Dict[str, Any]],
        instruction: str,
    ) -> None:
        key = (task_type, semantic_ref, candidate_id or "")
        if key in seen:
            return
        seen.add(key)
        tasks.append({
            "id": f"gt_{len(tasks)}",
            "task_type": task_type,
            "semantic_ref": semantic_ref,
            "candidate_id": candidate_id,
            "candidate_type": candidate_type,
            "visual_choices": visual_choices,
            "expected_output": {
                "semantic_ref": semantic_ref,
                "candidate_id": candidate_id,
                "visual_ids": [],
                "decision": "bind|absent|uncertain",
                "confidence": 0.0,
            },
            "instruction": instruction,
            "coordinate_policy": "visual ids are evidence only; do not output pixel coordinates as GeoSym coordinates",
        })

    def _line_choices(self, lines: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
        out = []
        for line in sorted(lines, key=lambda item: item.get("length_px", 0), reverse=True)[:limit]:
            out.append({
                "visual_id": line.get("id"),
                "kind": "line",
                "length_px": line.get("length_px"),
                "angle_deg": line.get("angle_deg"),
                "p1": line.get("p1"),
                "p2": line.get("p2"),
            })
        return out

    def _segment_choices(self, visual_structure: Dict[str, Any], limit: int = 20) -> List[Dict[str, Any]]:
        out = []
        for segment in (visual_structure.get("segment_candidates") or [])[:limit]:
            out.append({
                "visual_id": segment.get("id"),
                "kind": "segment_candidate",
                "source_visual_id": segment.get("visual_id"),
                "endpoint_candidate_ids": segment.get("endpoint_candidate_ids") or [],
                "length_px": segment.get("length_px"),
                "angle_deg": segment.get("angle_deg"),
                "orientation": segment.get("orientation"),
                "coordinate_policy": "segment candidate is visual evidence only",
            })
        return out

    def _angle_choices(self, visual_structure: Dict[str, Any], limit: int = 20) -> List[Dict[str, Any]]:
        out = []
        for angle in (visual_structure.get("angle_candidates") or [])[:limit]:
            out.append({
                "visual_id": angle.get("id"),
                "kind": "angle_candidate",
                "vertex_point_candidate_id": angle.get("vertex_point_candidate_id"),
                "intersection_visual_id": angle.get("intersection_visual_id"),
                "line_ids": angle.get("line_ids") or [],
                "coordinate_policy": "angle candidate is visual evidence only",
            })
        return out

    def _label_region_choices(self, visual_structure: Dict[str, Any], limit: int = 20) -> List[Dict[str, Any]]:
        out = []
        for region in (visual_structure.get("label_region_candidates") or [])[:limit]:
            out.append({
                "visual_id": region.get("id"),
                "kind": "label_region_candidate",
                "source_visual_id": region.get("visual_id"),
                "bbox": region.get("bbox"),
                "area_px": region.get("area_px"),
                "coordinate_policy": "label region candidate is visual evidence only",
            })
        return out

    def _circle_choices(self, circles: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        out = []
        for circle in sorted(circles, key=lambda item: item.get("radius_px", 0), reverse=True)[:limit]:
            out.append({
                "visual_id": circle.get("id"),
                "kind": "circle",
                "center": circle.get("center"),
                "radius_px": circle.get("radius_px"),
            })
        return out

    def _nearest_text_choices(self, text_regions: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
        out = []
        for region in text_regions[:limit]:
            out.append({
                "visual_id": region.get("id"),
                "kind": "text_region",
                "bbox": region.get("bbox"),
                "area_px": region.get("area_px"),
            })
        return out

    def _intersection_choices(self, topology: Dict[str, Any], limit: int = 20) -> List[Dict[str, Any]]:
        out = []
        for item in (topology.get("line_intersections") or [])[:limit]:
            out.append({
                "visual_id": item.get("id"),
                "kind": "intersection",
                "line_ids": item.get("line_ids"),
                "xy": item.get("xy"),
            })
        return out

    def _iter_source_texts(self, text_sources: List[Dict[str, str]] | None, fallback: str = "") -> List[Tuple[str, str]]:
        if text_sources:
            return [
                (str(item.get("name") or f"text_source_{idx}"), str(item.get("text") or ""))
                for idx, item in enumerate(text_sources)
                if item.get("text")
            ]
        return [("problem_text", fallback or "")]

    def _first_match_group(self, match: re.Match[str], start: int, end: int) -> str:
        for idx in range(start, end):
            value = match.group(idx)
            if value:
                return value
        return ""

    def _normalize_value(self, value: Any) -> str:
        text = str(value or "").strip()
        if text.startswith("\\sqrt"):
            inner = re.sub(r"\\sqrt\{?([0-9]+)\}?", r"sqrt(\1)", text)
            return inner
        return text

    def _dedupe_claims(self, claims: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
        out = []
        seen = set()
        for claim in claims:
            key_parts = []
            for field in fields:
                value = claim.get(field)
                if isinstance(value, list):
                    value = tuple(str(item) for item in value)
                key_parts.append(value)
            key = tuple(key_parts)
            if key in seen:
                continue
            seen.add(key)
            out.append(claim)
        return out

    def _match_in_ratio_context(self, text: str, start: int, end: int) -> bool:
        left = text[max(0, start - 4) : start]
        right = text[end : min(len(text), end + 4)]
        local = text[max(0, start - 8) : min(len(text), end + 8)]
        wide = text[max(0, start - 32) : min(len(text), end + 32)].lower()
        return bool(
            "ratioof" in wide
            or re.search(r"[A-Z]{2}\s*[:/]\s*$", left, re.I)
            or re.search(r"^\s*[:/]\s*[A-Z]{2}", right, re.I)
            or re.search(r"[A-Z]{2}\s*[:/]\s*[A-Z]{2}", local, re.I)
        )

    def _is_right_angle_value(self, value: Any) -> bool:
        text = str(value or "").strip().lower().rstrip("°")
        return text in {"90", "90.0", "right"}

    def _append_relation_candidate(
        self,
        relation_candidates: List[Dict[str, Any]],
        seen_relations: set[Tuple[str, str, str, str]],
        ctype: str,
        objects: List[str],
        source_name: str,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        clean_objects = [str(item).upper() for item in objects if str(item)]
        key = (ctype, "|".join(clean_objects), "", source_name)
        global_key = (ctype, "|".join(clean_objects), "", "")
        if global_key in seen_relations:
            return
        seen_relations.add(global_key)
        prefix = {
            "parallel": "sc_parallel",
            "perpendicular": "sc_perp",
            "right_triangle": "sc_right_triangle",
            "circle": "sc_circle",
            "area": "sc_area",
            "perimeter": "sc_perimeter",
            "ratio": "sc_ratio",
            "triangle": "sc_triangle",
        }.get(ctype, f"sc_{ctype}")
        candidate = {
            "id": f"{prefix}_{len([item for item in relation_candidates if item.get('type') == ctype])}",
            "type": ctype,
            "objects": clean_objects,
            "source": "text_regex",
            "source_text": source_name,
            "needs_vlm_confirmation": True,
        }
        if ctype in {"parallel", "perpendicular"}:
            candidate["grounding_required"] = "bind both lines to vl_* candidates if visible"
        if extra:
            candidate.update(extra)
        relation_candidates.append(candidate)

    def _append_right_angle_candidate(
        self,
        relation_candidates: List[Dict[str, Any]],
        seen_relations: set[Tuple[str, str, str, str]],
        angle: str,
        source_name: str,
    ) -> None:
        if len(angle) != 3:
            return
        lines = [angle[0] + angle[1], angle[1] + angle[2]]
        key = ("right_angle", "|".join(lines), "", "")
        if key in seen_relations:
            return
        seen_relations.add(key)
        relation_candidates.append({
            "id": f"sc_right_angle_{len([item for item in relation_candidates if item.get('type') == 'right_angle'])}",
            "type": "right_angle",
            "objects": lines,
            "angle": angle,
            "vertex": angle[1],
            "source": "text_regex",
            "source_text": source_name,
            "needs_vlm_confirmation": True,
            "grounding_required": "bind both rays/segments to vl_* candidates if visible",
            "engine_note": "can convert to set_perpendicular for the two incident segments",
        })

    def _engine_build_suggestions(self, semantic_candidates: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        relation_candidates = semantic_candidates.get("relation_candidates") or []
        target_candidates = semantic_candidates.get("target_candidates") or []
        buildable_types = {"length", "parallel", "perpendicular", "right_angle"}
        theorem_only_types = {"circle", "right_triangle", "angle", "arc_measure", "similar", "congruent", "area", "perimeter", "ratio", "triangle"}
        unsupported_types = sorted({str(item.get("type")) for item in relation_candidates if item.get("type") not in buildable_types})
        suggested = []
        declared_points: set[str] = set()
        declared_segments: set[str] = set()
        for candidate in relation_candidates:
            ctype = candidate.get("type")
            if ctype == "length":
                obj = str(candidate.get("object") or "").upper()
                if len(obj) >= 2:
                    for point in obj[:2]:
                        if point not in declared_points:
                            suggested.append({"action": "add_point", "id": point, "from_candidate": candidate.get("id")})
                            declared_points.add(point)
                    if obj[:2] not in declared_segments:
                        suggested.append({"action": "add_segment", "points": [obj[0], obj[1]], "from_candidate": candidate.get("id")})
                        declared_segments.add(obj[:2])
                    suggested.append({"action": "set_length", "points": [obj[0], obj[1]], "value": candidate.get("value"), "from_candidate": candidate.get("id")})
            elif ctype in {"parallel", "perpendicular", "right_angle"}:
                lines = [str(item).upper() for item in candidate.get("objects") or [] if len(str(item)) >= 2]
                for line in lines:
                    for point in line[:2]:
                        if point not in declared_points:
                            suggested.append({"action": "add_point", "id": point, "from_candidate": candidate.get("id")})
                            declared_points.add(point)
                    if line[:2] not in declared_segments:
                        suggested.append({"action": "add_segment", "points": [line[0], line[1]], "from_candidate": candidate.get("id")})
                        declared_segments.add(line[:2])
                if len(lines) >= 2:
                    action = "set_perpendicular" if ctype == "right_angle" else f"set_{ctype}"
                    suggested.append({"action": action, "lines": lines[:2], "from_candidate": candidate.get("id")})
        for target in target_candidates:
            suggested.append({"action": "set_target", "target": target, "from_candidate": target.get("id")})
        if set(unsupported_types) & theorem_only_types:
            expected_mode = "theorem_only"
        elif unsupported_types:
            expected_mode = "unsupported"
        elif suggested:
            expected_mode = "coordinate_buildable"
        else:
            expected_mode = "failed"
        return {
            "expected_mode_hint": expected_mode,
            "unsupported_candidate_types": unsupported_types,
            "suggested_build_instructions": suggested[:80],
            "instruction_note": "VLM must confirm these suggestions against image/overlay before copying into build_instructions",
        }

    def _candidate_ids(self, visual_state: Dict[str, Any]) -> Dict[str, List[str]]:
        topology = visual_state.get("topology") or {}
        return {
            "lines": [str(item.get("id")) for item in visual_state.get("lines") or [] if item.get("id")],
            "circles": [str(item.get("id")) for item in visual_state.get("circles") or [] if item.get("id")],
            "fill_regions": [str(item.get("id")) for item in visual_state.get("fill_regions") or [] if item.get("id")],
            "markers": [str(item.get("id")) for item in visual_state.get("markers") or [] if item.get("id")],
            "text_regions": [str(item.get("id")) for item in visual_state.get("text_regions") or [] if item.get("id")],
            "endpoints": [str(item.get("id")) for item in topology.get("endpoints") or [] if item.get("id")],
            "intersections": [str(item.get("id")) for item in topology.get("line_intersections") or [] if item.get("id")],
            "circle_line_hits": [str(item.get("id")) for item in topology.get("circle_line_hits") or [] if item.get("id")],
        }

    def _orientation(self, angle: float) -> str:
        normalized = abs(((angle + 90.0) % 180.0) - 90.0)
        if normalized <= 10.0:
            return "horizontal"
        if abs(normalized - 90.0) <= 10.0:
            return "vertical"
        return "diagonal"

    def _angle_diff(self, a: float, b: float) -> float:
        diff = abs((a - b) % 180.0)
        return min(diff, 180.0 - diff)
