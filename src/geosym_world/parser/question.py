"""Conservative target-type extraction from public question text."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TargetTypeMatch:
    target_type: str
    span: tuple[int, int]
    pattern: str


@dataclass(frozen=True)
class TargetReferenceMatch:
    point_names: tuple[str, ...]
    span: tuple[int, int]
    pattern: str
    reference_kind: str = "segment"


@dataclass(frozen=True)
class QuantityAssertionMatch:
    quantity_kind: str
    point_names: tuple[str, ...]
    expression_text: str
    span: tuple[int, int]
    expression_span: tuple[int, int]
    pattern: str


@dataclass(frozen=True)
class RelationReferenceMatch:
    role: str
    reference_kind: str
    point_names: tuple[str, ...]


@dataclass(frozen=True)
class RelationAssertionMatch:
    predicate: str
    references: tuple[RelationReferenceMatch, ...]
    span: tuple[int, int]
    pattern: str


_TARGET_PATTERNS = (
    ("proof", r"\b(prove|show that)\b|证明"),
    (
        "shadow_ratio",
        r"\bratio\s+of\s+(?:(?:the|a|an|its|their)\s+)?shaded\b"
        r"|阴影(?:部分|区域)?(?:面积)?(?:的)?(?:比值|比例)",
    ),
    ("ratio", r"\bratio\b|比值|比例"),
    (
        "shadow_area",
        r"\b(?:area\s+of\s+(?:(?:the|a|an|its|their)\s+)?shaded|shaded(?:\s+[\w-]+){0,3}\s+area)\b"
        r"|阴影(?:部分|区域)?(?:的)?面积|阴影面积",
    ),
    ("area", r"\barea\b|面积"),
    ("perimeter", r"\bperimeter\b|周长"),
    ("arc_length", r"\barc length\b|弧长"),
    ("volume", r"\bvolume\b|体积"),
    ("angle", r"\bangle\b|角度|求角"),
    ("length", r"\b(length|distance)\b|长度|距离"),
)

_LENGTH_REFERENCE_PATTERNS = (
    r"\b(?:length|distance)\s+(?:of\s+)?(?:segment\s+)?([A-Z])([A-Z])\b",
    r"\b(?:segment\s+)?([A-Z])([A-Z])(?:'s)?\s+(?:length|distance)\b",
    r"(?:线段)?\s*([A-Z])([A-Z])\s*(?:的)?(?:长度|距离)",
    r"(?:长度|距离)\s*(?:为|是|of)?\s*(?:线段)?\s*([A-Z])([A-Z])",
)

_POLYGON_REFERENCE_PATTERNS = (
    r"\b(?:area|perimeter)\s+(?:of\s+)?(?:the\s+)?(?:triangle|rectangle|quadrilateral|polygon|trapezoid|parallelogram|kite)\s+([A-Z]{3,12})\b",
    r"\b(?:triangle|rectangle|quadrilateral|polygon|trapezoid|parallelogram|kite)\s+([A-Z]{3,12})(?:'s)?\s+(?:area|perimeter)\b",
    r"(?:三角形|矩形|四边形|多边形|梯形|平行四边形|风筝形)\s*([A-Z]{3,12})\s*(?:的)?(?:面积|周长)",
    r"(?:面积|周长)\s*(?:为|是|of)?\s*(?:三角形|矩形|四边形|多边形|梯形|平行四边形|风筝形)\s*([A-Z]{3,12})",
)

_LENGTH_ASSERTION_PATTERNS = (
    r"\b(?:length\s+of\s+)?(?:segment\s+)?([A-Z])([A-Z])\s*(?:=|is)\s*([^\s,，;；。!?！？]+)",
    r"(?:线段)?\s*([A-Z])([A-Z])\s*(?:的)?(?:长度|长)\s*(?:为|是|=)\s*([^\s,，;；。!?！？]+)",
)

_BINARY_SEGMENT_RELATION_PATTERNS = (
    (
        "parallel",
        r"\b(?:segment\s+)?([A-Z])([A-Z])\s+(?:is\s+)?parallel\s+to\s+(?:segment\s+)?([A-Z])([A-Z])\b",
    ),
    ("parallel", r"\b([A-Z])([A-Z])\s*(?://|∥)\s*([A-Z])([A-Z])\b"),
    ("parallel", r"(?:线段\s*)?([A-Z])([A-Z])\s*(?:平行于|∥)\s*(?:线段\s*)?([A-Z])([A-Z])"),
    (
        "perpendicular",
        r"\b(?:segment\s+)?([A-Z])([A-Z])\s+(?:is\s+)?perpendicular\s+to\s+(?:segment\s+)?([A-Z])([A-Z])\b",
    ),
    ("perpendicular", r"\b([A-Z])([A-Z])\s*⊥\s*([A-Z])([A-Z])\b"),
    ("perpendicular", r"(?:线段\s*)?([A-Z])([A-Z])\s*(?:垂直于|⊥)\s*(?:线段\s*)?([A-Z])([A-Z])"),
    (
        "equal_length",
        r"\b(?:segment\s+)?([A-Z])([A-Z])\s+(?:is\s+)?(?:equal(?:\s+in\s+length)?\s+to|has\s+the\s+same\s+length\s+as)\s+(?:segment\s+)?([A-Z])([A-Z])\b",
    ),
    ("equal_length", r"\b([A-Z])([A-Z])\s*=\s*([A-Z])([A-Z])\b"),
    ("equal_length", r"(?:线段\s*)?([A-Z])([A-Z])\s*等长于\s*(?:线段\s*)?([A-Z])([A-Z])"),
    ("equal_length", r"(?:线段\s*)?([A-Z])([A-Z])\s*与\s*(?:线段\s*)?([A-Z])([A-Z])\s*(?:等长|长度相等)"),
)

_MIDPOINT_PATTERNS = (
    r"\b(?:point\s+)?([A-Z])\s+is\s+(?:the\s+)?midpoint\s+of\s+(?:segment\s+)?([A-Z])([A-Z])\b",
    r"(?:点\s*)?([A-Z])\s*(?:是|为)\s*(?:线段\s*)?([A-Z])([A-Z])\s*(?:的)?中点",
)

_COLLINEAR_PATTERNS = (
    r"\b(?:points?\s+)?([A-Z])\s*[,，]?\s*([A-Z])\s*[,，]?\s*(?:and\s+)?([A-Z])\s+are\s+collinear\b",
    r"(?:点\s*)?([A-Z])\s*[、,，]\s*([A-Z])\s*[、,，]\s*([A-Z])\s*(?:三点)?共线",
    r"\b([A-Z])([A-Z])([A-Z])\s*(?:三点)?共线",
)


def infer_target_type(question: str) -> TargetTypeMatch | None:
    for target_type, pattern in _TARGET_PATTERNS:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match is not None:
            return TargetTypeMatch(target_type, match.span(), pattern)
    return None


def infer_target_reference(question: str, target_type: str) -> TargetReferenceMatch | None:
    """Extract only explicit object notation adjacent to a supported target cue."""

    for pattern in _LENGTH_REFERENCE_PATTERNS if target_type == "length" else ():
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match is not None:
            names = tuple(value.upper() for value in match.groups())
            if len(set(names)) == len(names):
                return TargetReferenceMatch(names, match.span(), pattern)
    for pattern in _POLYGON_REFERENCE_PATTERNS if target_type in {"area", "perimeter"} else ():
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match is not None:
            names = tuple(match.group(1).upper())
            if len(names) >= 3 and len(set(names)) == len(names):
                return TargetReferenceMatch(names, match.span(), pattern, reference_kind="polygon")
    return None


def infer_quantity_assertions(question: str) -> tuple[QuantityAssertionMatch, ...]:
    """Extract only explicit segment-length equalities from public question text."""

    matches = []
    occupied_spans = set()
    relation_spans = tuple(match.span for match in infer_relation_assertions(question))
    for pattern in _LENGTH_ASSERTION_PATTERNS:
        for match in re.finditer(pattern, question, flags=re.IGNORECASE):
            if match.span() in occupied_spans:
                continue
            if any(_spans_overlap(match.span(), span) for span in relation_spans):
                continue
            names = tuple(value.upper() for value in match.groups()[:2])
            if len(set(names)) != 2:
                continue
            expression_text = match.group(3).rstrip(".:")
            expression_end = match.end(3) - (len(match.group(3)) - len(expression_text))
            if not expression_text:
                continue
            matches.append(
                QuantityAssertionMatch(
                    quantity_kind="length",
                    point_names=names,
                    expression_text=expression_text,
                    span=match.span(),
                    expression_span=(match.start(3), expression_end),
                    pattern=pattern,
                )
            )
            occupied_spans.add(match.span())
    return tuple(sorted(matches, key=lambda item: item.span))


def infer_relation_assertions(question: str) -> tuple[RelationAssertionMatch, ...]:
    """Extract a conservative set of explicit geometric relations from public text."""

    matches = []
    occupied_spans: list[tuple[int, int]] = []
    for predicate, pattern in _BINARY_SEGMENT_RELATION_PATTERNS:
        for match in re.finditer(pattern, question, flags=re.IGNORECASE):
            if any(_spans_overlap(match.span(), span) for span in occupied_spans):
                continue
            names = tuple(value.upper() for value in match.groups())
            if len(names) != 4 or len(set(names[:2])) != 2 or len(set(names[2:])) != 2:
                continue
            matches.append(
                RelationAssertionMatch(
                    predicate=predicate,
                    references=(
                        RelationReferenceMatch("object_0", "segment", names[:2]),
                        RelationReferenceMatch("object_1", "segment", names[2:]),
                    ),
                    span=match.span(),
                    pattern=pattern,
                )
            )
            occupied_spans.append(match.span())

    for pattern in _MIDPOINT_PATTERNS:
        for match in re.finditer(pattern, question, flags=re.IGNORECASE):
            if any(_spans_overlap(match.span(), span) for span in occupied_spans):
                continue
            midpoint_name, first_name, second_name = (value.upper() for value in match.groups())
            if len({midpoint_name, first_name, second_name}) != 3:
                continue
            matches.append(
                RelationAssertionMatch(
                    predicate="midpoint",
                    references=(
                        RelationReferenceMatch("point", "point", (midpoint_name,)),
                        RelationReferenceMatch("of_0", "point", (first_name,)),
                        RelationReferenceMatch("of_1", "point", (second_name,)),
                    ),
                    span=match.span(),
                    pattern=pattern,
                )
            )
            occupied_spans.append(match.span())

    for pattern in _COLLINEAR_PATTERNS:
        for match in re.finditer(pattern, question, flags=re.IGNORECASE):
            if any(_spans_overlap(match.span(), span) for span in occupied_spans):
                continue
            names = tuple(value.upper() for value in match.groups())
            if len(names) != 3 or len(set(names)) != 3:
                continue
            matches.append(
                RelationAssertionMatch(
                    predicate="collinear",
                    references=tuple(
                        RelationReferenceMatch(f"point_{index}", "point", (name,))
                        for index, name in enumerate(names)
                    ),
                    span=match.span(),
                    pattern=pattern,
                )
            )
            occupied_spans.append(match.span())
    return tuple(sorted(matches, key=lambda item: item.span))


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]
