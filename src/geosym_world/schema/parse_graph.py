"""Typed, evidence-bearing geometry parse graphs before state admission."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import sympy as sp

from geosym_world.errors import InvalidGeometryInput

from .canonical import stable_checksum
from .ids import validate_geometry_id
from .relation_normalization import RELATION_TYPES


PARSE_GRAPH_SCHEMA_VERSION = "1.0"
PARSE_GRAPH_HYPOTHESIS_KIND = "geometry_parse_graph"


class ParseEvidenceKind(str, Enum):
    IMAGE_REGION = "image_region"
    PIXEL_POINT = "pixel_point"
    TEXT_SPAN = "text_span"
    NATIVE_ENTITY = "native_entity"
    SOURCE_DOCUMENT = "source_document"


class EvidencePrecision(str, Enum):
    IMAGE_ONLY = "image_only"
    LOCALIZED = "localized"
    BOUND = "bound"


class ParseOrigin(str, Enum):
    IMAGE = "image"
    TEXT = "text"
    IMAGE_TEXT = "image_text"
    NATIVE = "native"
    COMPILED = "compiled"


class ParseObjectKind(str, Enum):
    POINT = "point"
    LINE = "line"
    SEGMENT = "segment"
    RAY = "ray"
    CIRCLE = "circle"
    ARC = "arc"
    POLYGON = "polygon"
    PLANE = "plane"
    POLYHEDRON = "polyhedron"
    TEXT = "text"
    MARKER = "marker"
    UNKNOWN = "unknown"


class ConflictStatus(str, Enum):
    UNRESOLVED = "unresolved"
    RESOLVED = "resolved"


PARSE_RELATION_TYPES = frozenset(
    {
        *RELATION_TYPES,
        "binds",
        "boundary_of",
        "contains",
        "difference_of",
        "disjoint",
        "endpoint_of",
        "hidden_edge",
        "inside",
        "intersection_of",
        "ordered_points",
        "overlaps",
        "shaded_region",
        "support_of",
        "touches",
        "union_of",
        "visible_extension",
    }
)

QUANTITY_KINDS = frozenset(
    {
        "angle",
        "arc_length",
        "area",
        "coordinate",
        "diameter",
        "length",
        "perimeter",
        "radius",
        "ratio",
        "unknown",
        "volume",
    }
)

EXPRESSION_OPERATORS = frozenset({"literal", "symbol", "neg", "add", "sub", "mul", "div", "pow", "sqrt"})
REGION_OPERATORS = frozenset({"reference", "union", "intersection", "difference"})
_LITERAL_PATTERN = re.compile(r"^[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$")
_SYMBOL_PATTERN = re.compile(r"^(?:[A-Za-z][A-Za-z0-9_]*|pi)$")
_MAX_EXPRESSION_TEXT_LENGTH = 256
_MAX_EXPRESSION_AST_NODES = 96
_OBJECT_GEOMETRY_FIELDS = {
    ParseObjectKind.POINT: frozenset({"pixel", "exact_coords"}),
    ParseObjectKind.LINE: frozenset({"p1", "p2", "angle_deg"}),
    ParseObjectKind.SEGMENT: frozenset({"p1", "p2", "angle_deg"}),
    ParseObjectKind.RAY: frozenset({"p1", "p2", "angle_deg"}),
    ParseObjectKind.CIRCLE: frozenset({"center", "radius_px", "exact_radius"}),
    ParseObjectKind.ARC: frozenset({"orientation", "exact_central_angle"}),
    ParseObjectKind.POLYGON: frozenset(),
    ParseObjectKind.PLANE: frozenset(),
    ParseObjectKind.POLYHEDRON: frozenset(),
    ParseObjectKind.TEXT: frozenset({"bbox"}),
    ParseObjectKind.MARKER: frozenset({"bbox"}),
    ParseObjectKind.UNKNOWN: frozenset(),
}


@dataclass(frozen=True)
class ParseReference:
    role: str
    ref: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role.strip():
            raise InvalidGeometryInput("Parse reference role must be non-empty")
        validate_geometry_id(self.ref)

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "ref": self.ref}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParseReference":
        _check_fields(data, {"role", "ref"}, set(), "parse reference")
        return cls(role=str(data["role"]), ref=str(data["ref"]))


@dataclass(frozen=True)
class ParseEvidence:
    id: str
    source_id: str
    kind: ParseEvidenceKind
    precision: EvidencePrecision
    bbox: tuple[float, float, float, float] | None = None
    point: tuple[float, float] | None = None
    span: tuple[int, int] | None = None
    raw_content: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_geometry_id(self.id)
        validate_geometry_id(self.source_id)
        _require_enum(self.kind, ParseEvidenceKind, "evidence kind")
        _require_enum(self.precision, EvidencePrecision, "evidence precision")
        if self.bbox is not None:
            _validate_finite_tuple(self.bbox, 4, "evidence bbox")
            if self.bbox[0] > self.bbox[2] or self.bbox[1] > self.bbox[3]:
                raise InvalidGeometryInput(f"Evidence {self.id} has an inverted bbox")
        if self.point is not None:
            _validate_finite_tuple(self.point, 2, "evidence point")
        if self.span is not None:
            if len(self.span) != 2 or any(not isinstance(value, int) for value in self.span):
                raise InvalidGeometryInput(f"Evidence {self.id} text span must contain two integers")
            if self.span[0] < 0 or self.span[0] > self.span[1]:
                raise InvalidGeometryInput(f"Evidence {self.id} has an invalid text span")
        required_location = {
            ParseEvidenceKind.IMAGE_REGION: self.bbox,
            ParseEvidenceKind.PIXEL_POINT: self.point,
            ParseEvidenceKind.TEXT_SPAN: self.span,
        }.get(self.kind)
        if self.kind in {ParseEvidenceKind.IMAGE_REGION, ParseEvidenceKind.PIXEL_POINT, ParseEvidenceKind.TEXT_SPAN} and required_location is None:
            raise InvalidGeometryInput(f"Evidence {self.id} is missing its required location")
        _require_mapping(self.metadata, f"Evidence {self.id} metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "kind": self.kind.value,
            "precision": self.precision.value,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "point": list(self.point) if self.point is not None else None,
            "span": list(self.span) if self.span is not None else None,
            "raw_content": self.raw_content,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParseEvidence":
        _check_fields(
            data,
            {"id", "source_id", "kind", "precision"},
            {"bbox", "point", "span", "raw_content", "metadata"},
            "parse evidence",
        )
        return cls(
            id=str(data["id"]),
            source_id=str(data["source_id"]),
            kind=ParseEvidenceKind(data["kind"]),
            precision=EvidencePrecision(data["precision"]),
            bbox=_float_tuple(data.get("bbox"), 4),
            point=_float_tuple(data.get("point"), 2),
            span=_int_tuple(data.get("span"), 2),
            raw_content=str(data["raw_content"]) if data.get("raw_content") is not None else None,
            metadata=_metadata(data),
        )


@dataclass(frozen=True)
class MathExpression:
    operator: str
    value: str | None = None
    operands: tuple["MathExpression", ...] = ()

    def __post_init__(self) -> None:
        if self.operator not in EXPRESSION_OPERATORS:
            raise InvalidGeometryInput(f"Unknown math expression operator: {self.operator}")
        operand_count = len(self.operands)
        if self.operator == "literal":
            if self.value is None or not _LITERAL_PATTERN.fullmatch(self.value) or operand_count:
                raise InvalidGeometryInput("Literal expressions require one numeric value and no operands")
        elif self.operator == "symbol":
            if self.value is None or not _SYMBOL_PATTERN.fullmatch(self.value) or operand_count:
                raise InvalidGeometryInput("Symbol expressions require one safe symbol and no operands")
        elif self.value is not None:
            raise InvalidGeometryInput(f"Expression operator {self.operator} cannot define a value")
        elif self.operator in {"neg", "sqrt"} and operand_count != 1:
            raise InvalidGeometryInput(f"Expression operator {self.operator} requires one operand")
        elif self.operator in {"sub", "div", "pow"} and operand_count != 2:
            raise InvalidGeometryInput(f"Expression operator {self.operator} requires two operands")
        elif self.operator in {"add", "mul"} and operand_count < 2:
            raise InvalidGeometryInput(f"Expression operator {self.operator} requires at least two operands")

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "value": self.value,
            "operands": [operand.to_dict() for operand in self.operands],
        }

    def to_sympy(self) -> sp.Expr:
        """Compile the validated expression tree without evaluating source text."""

        if self.operator == "literal":
            return sp.Rational(self.value)
        if self.operator == "symbol":
            return sp.pi if self.value == "pi" else sp.Symbol(self.value)
        values = tuple(operand.to_sympy() for operand in self.operands)
        if self.operator == "neg":
            return -values[0]
        if self.operator == "add":
            return sp.Add(*values)
        if self.operator == "sub":
            return values[0] - values[1]
        if self.operator == "mul":
            return sp.Mul(*values)
        if self.operator == "div":
            if values[1].is_zero is True:
                raise InvalidGeometryInput("Math expression division by zero")
            return values[0] / values[1]
        if self.operator == "pow":
            return values[0] ** values[1]
        if self.operator == "sqrt":
            return sp.sqrt(values[0])
        raise AssertionError(f"Unhandled math expression operator: {self.operator}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MathExpression":
        _check_fields(data, {"operator"}, {"value", "operands"}, "math expression")
        return cls(
            operator=str(data["operator"]),
            value=str(data["value"]) if data.get("value") is not None else None,
            operands=tuple(cls.from_dict(item) for item in data.get("operands", ())),
        )


def math_expression_from_sympy(value: Any) -> MathExpression:
    """Convert a supported exact SymPy value into the safe expression AST."""

    if isinstance(value, bool) or not isinstance(value, (int, float, sp.Basic)):
        raise InvalidGeometryInput(f"Expected a numeric or SymPy expression, got {type(value).__name__}")
    expression = sp.Integer(value) if isinstance(value, int) else sp.Float(value) if isinstance(value, float) else value
    if expression.is_finite is False:
        raise InvalidGeometryInput("Math expressions must be finite")
    if expression == sp.pi:
        return MathExpression("symbol", value="pi")
    if isinstance(expression, sp.Integer):
        return MathExpression("literal", value=str(expression))
    if isinstance(expression, sp.Rational):
        return MathExpression(
            "div",
            operands=(
                MathExpression("literal", value=str(expression.p)),
                MathExpression("literal", value=str(expression.q)),
            ),
        )
    if isinstance(expression, sp.Float):
        return MathExpression("literal", value=str(expression))
    if isinstance(expression, sp.Symbol):
        return MathExpression("symbol", value=str(expression))
    if isinstance(expression, sp.Add):
        return MathExpression("add", operands=tuple(math_expression_from_sympy(item) for item in expression.args))
    if isinstance(expression, sp.Mul):
        return MathExpression("mul", operands=tuple(math_expression_from_sympy(item) for item in expression.args))
    if isinstance(expression, sp.Pow):
        if expression.exp == sp.Rational(1, 2):
            return MathExpression("sqrt", operands=(math_expression_from_sympy(expression.base),))
        return MathExpression(
            "pow",
            operands=(math_expression_from_sympy(expression.base), math_expression_from_sympy(expression.exp)),
        )
    raise InvalidGeometryInput(f"Unsupported exact expression type: {type(expression).__name__}")


def math_expression_from_text(value: str) -> MathExpression:
    """Parse a small mathematical expression language without evaluating source text."""

    if not isinstance(value, str) or not value.strip():
        raise InvalidGeometryInput("Math expression text must be non-empty")
    if len(value) > _MAX_EXPRESSION_TEXT_LENGTH:
        raise InvalidGeometryInput("Math expression text is too long")
    normalized = _normalize_expression_text(value)
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise InvalidGeometryInput(f"Invalid math expression syntax: {value!r}") from exc
    if sum(1 for _ in ast.walk(tree)) > _MAX_EXPRESSION_AST_NODES:
        raise InvalidGeometryInput("Math expression is too complex")
    return _math_expression_from_ast(tree.body)


def _math_expression_from_ast(node: ast.AST) -> MathExpression:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise InvalidGeometryInput("Math expression literals must be real numbers")
        if isinstance(node.value, float) and not math.isfinite(node.value):
            raise InvalidGeometryInput("Math expression literals must be finite")
        return MathExpression("literal", value=str(node.value))
    if isinstance(node, ast.Name):
        if not _SYMBOL_PATTERN.fullmatch(node.id) or "__" in node.id:
            raise InvalidGeometryInput(f"Unsafe math symbol: {node.id!r}")
        return MathExpression("symbol", value=node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _math_expression_from_ast(node.operand)
        return operand if isinstance(node.op, ast.UAdd) else MathExpression("neg", operands=(operand,))
    if isinstance(node, ast.BinOp):
        operators = {
            ast.Add: "add",
            ast.Sub: "sub",
            ast.Mult: "mul",
            ast.Div: "div",
            ast.Pow: "pow",
        }
        operator = next((name for kind, name in operators.items() if isinstance(node.op, kind)), None)
        if operator is None:
            raise InvalidGeometryInput(f"Unsupported math operator: {type(node.op).__name__}")
        return MathExpression(
            operator,
            operands=(_math_expression_from_ast(node.left), _math_expression_from_ast(node.right)),
        )
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sqrt"
        and len(node.args) == 1
        and not node.keywords
    ):
        return MathExpression("sqrt", operands=(_math_expression_from_ast(node.args[0]),))
    raise InvalidGeometryInput(f"Unsupported math expression element: {type(node).__name__}")


def _normalize_expression_text(value: str) -> str:
    text = value.strip().replace("−", "-").replace("×", "*").replace("÷", "/").replace("π", "pi")
    for symbol, fraction in {"½": "(1/2)", "⅓": "(1/3)", "⅔": "(2/3)", "¼": "(1/4)", "¾": "(3/4)"}.items():
        text = text.replace(symbol, fraction)
    text = _rewrite_latex_command(text, "frac", 2)
    text = _rewrite_latex_command(text, "sqrt", 1)
    text = re.sub(r"√\s*\(([^()]*)\)", r"sqrt(\1)", text)
    text = re.sub(r"√\s*([A-Za-z0-9_.]+)", r"sqrt(\1)", text)
    text = text.replace("^", "**")
    text = re.sub(r"(?<=[0-9A-Za-z_)])(?=sqrt\s*\()", "*", text)
    text = re.sub(r"(?<=[0-9_)])(?=(?:pi\b|[A-DF-Za-df-z]))", "*", text)
    text = re.sub(r"(?<=\))(?=\()", "*", text)
    return text


def _rewrite_latex_command(value: str, command: str, arity: int) -> str:
    marker = f"\\{command}"
    text = value
    while marker in text:
        start = text.find(marker)
        cursor = start + len(marker)
        groups = []
        for _ in range(arity):
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            group, cursor = _read_braced_group(text, cursor)
            groups.append(_normalize_expression_text(group))
        replacement = f"(({groups[0]})/({groups[1]}))" if command == "frac" else f"sqrt({groups[0]})"
        text = text[:start] + replacement + text[cursor:]
    return text


def _read_braced_group(value: str, start: int) -> tuple[str, int]:
    if start >= len(value) or value[start] != "{":
        raise InvalidGeometryInput("LaTeX math commands require braced arguments")
    depth = 0
    for index in range(start, len(value)):
        if value[index] == "{":
            depth += 1
        elif value[index] == "}":
            depth -= 1
            if depth == 0:
                return value[start + 1 : index], index + 1
    raise InvalidGeometryInput("Unclosed LaTeX math group")


@dataclass(frozen=True)
class ParseObject:
    id: str
    kind: ParseObjectKind
    origin: ParseOrigin
    confidence: float
    score_kind: str
    references: tuple[ParseReference, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    visible_name: str | None = None
    aliases: tuple[str, ...] = ()
    visible: bool | None = None
    geometry: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_geometry_id(self.id)
        _require_enum(self.kind, ParseObjectKind, "parse object kind")
        _validate_candidate(self.id, self.origin, self.confidence, self.score_kind, self.evidence_refs)
        if self.visible_name is not None and not self.visible_name.strip():
            raise InvalidGeometryInput(f"Parse object {self.id} has an empty visible name")
        if any(not isinstance(alias, str) or not alias.strip() for alias in self.aliases):
            raise InvalidGeometryInput(f"Parse object {self.id} has an empty alias")
        if len(set(self.aliases)) != len(self.aliases):
            raise InvalidGeometryInput(f"Parse object {self.id} has duplicate aliases")
        if self.visible is not None and not isinstance(self.visible, bool):
            raise InvalidGeometryInput(f"Parse object {self.id} visibility must be boolean or null")
        _validate_object_geometry(self)
        _require_mapping(self.metadata, f"Parse object {self.id} metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "origin": self.origin.value,
            "confidence": self.confidence,
            "score_kind": self.score_kind,
            "references": [reference.to_dict() for reference in self.references],
            "evidence_refs": list(self.evidence_refs),
            "visible_name": self.visible_name,
            "aliases": list(self.aliases),
            "visible": self.visible,
            "geometry": dict(self.geometry),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParseObject":
        _check_fields(
            data,
            {"id", "kind", "origin", "confidence", "score_kind"},
            {"references", "evidence_refs", "visible_name", "aliases", "visible", "geometry", "metadata"},
            "parse object",
        )
        return cls(
            id=str(data["id"]),
            kind=ParseObjectKind(data["kind"]),
            origin=ParseOrigin(data["origin"]),
            confidence=float(data["confidence"]),
            score_kind=str(data["score_kind"]),
            references=tuple(ParseReference.from_dict(item) for item in data.get("references", ())),
            evidence_refs=_string_tuple(data.get("evidence_refs")),
            visible_name=str(data["visible_name"]) if data.get("visible_name") is not None else None,
            aliases=_string_tuple(data.get("aliases")),
            visible=data.get("visible"),
            geometry=_mapping_field(data, "geometry", "Parse object geometry"),
            metadata=_metadata(data),
        )


@dataclass(frozen=True)
class ParseRelation:
    id: str
    predicate: str
    arguments: tuple[ParseReference, ...]
    origin: ParseOrigin
    confidence: float
    score_kind: str
    evidence_refs: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_geometry_id(self.id)
        if self.predicate not in PARSE_RELATION_TYPES:
            raise InvalidGeometryInput(f"Unknown parse relation predicate: {self.predicate}")
        if not self.arguments:
            raise InvalidGeometryInput(f"Parse relation {self.id} requires at least one argument")
        _validate_candidate(self.id, self.origin, self.confidence, self.score_kind, self.evidence_refs)
        for dependency in self.dependencies:
            validate_geometry_id(dependency)
        _require_mapping(self.metadata, f"Parse relation {self.id} metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "predicate": self.predicate,
            "arguments": [argument.to_dict() for argument in self.arguments],
            "origin": self.origin.value,
            "confidence": self.confidence,
            "score_kind": self.score_kind,
            "evidence_refs": list(self.evidence_refs),
            "dependencies": list(self.dependencies),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParseRelation":
        _check_fields(
            data,
            {"id", "predicate", "arguments", "origin", "confidence", "score_kind"},
            {"evidence_refs", "dependencies", "metadata"},
            "parse relation",
        )
        return cls(
            id=str(data["id"]),
            predicate=str(data["predicate"]),
            arguments=tuple(ParseReference.from_dict(item) for item in data["arguments"]),
            origin=ParseOrigin(data["origin"]),
            confidence=float(data["confidence"]),
            score_kind=str(data["score_kind"]),
            evidence_refs=_string_tuple(data.get("evidence_refs")),
            dependencies=_string_tuple(data.get("dependencies")),
            metadata=_metadata(data),
        )


@dataclass(frozen=True)
class ParseQuantity:
    id: str
    kind: str
    arguments: tuple[ParseReference, ...]
    expression: MathExpression
    origin: ParseOrigin
    confidence: float
    score_kind: str
    unit: str | None = None
    raw_text: str | None = None
    evidence_refs: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_geometry_id(self.id)
        if self.kind not in QUANTITY_KINDS:
            raise InvalidGeometryInput(f"Unknown parse quantity kind: {self.kind}")
        if not self.arguments:
            raise InvalidGeometryInput(f"Parse quantity {self.id} requires at least one argument")
        _validate_candidate(self.id, self.origin, self.confidence, self.score_kind, self.evidence_refs)
        if self.unit is not None and not self.unit.strip():
            raise InvalidGeometryInput(f"Parse quantity {self.id} has an empty unit")
        for dependency in self.dependencies:
            validate_geometry_id(dependency)
        _require_mapping(self.metadata, f"Parse quantity {self.id} metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "arguments": [argument.to_dict() for argument in self.arguments],
            "expression": self.expression.to_dict(),
            "origin": self.origin.value,
            "confidence": self.confidence,
            "score_kind": self.score_kind,
            "unit": self.unit,
            "raw_text": self.raw_text,
            "evidence_refs": list(self.evidence_refs),
            "dependencies": list(self.dependencies),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParseQuantity":
        _check_fields(
            data,
            {"id", "kind", "arguments", "expression", "origin", "confidence", "score_kind"},
            {"unit", "raw_text", "evidence_refs", "dependencies", "metadata"},
            "parse quantity",
        )
        return cls(
            id=str(data["id"]),
            kind=str(data["kind"]),
            arguments=tuple(ParseReference.from_dict(item) for item in data["arguments"]),
            expression=MathExpression.from_dict(data["expression"]),
            origin=ParseOrigin(data["origin"]),
            confidence=float(data["confidence"]),
            score_kind=str(data["score_kind"]),
            unit=str(data["unit"]) if data.get("unit") is not None else None,
            raw_text=str(data["raw_text"]) if data.get("raw_text") is not None else None,
            evidence_refs=_string_tuple(data.get("evidence_refs")),
            dependencies=_string_tuple(data.get("dependencies")),
            metadata=_metadata(data),
        )


@dataclass(frozen=True)
class BoundaryReference:
    ref: str
    orientation: str = "unknown"

    def __post_init__(self) -> None:
        validate_geometry_id(self.ref)
        if self.orientation not in {"forward", "reverse", "unknown"}:
            raise InvalidGeometryInput(f"Unknown boundary orientation: {self.orientation}")

    def to_dict(self) -> dict[str, str]:
        return {"ref": self.ref, "orientation": self.orientation}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BoundaryReference":
        _check_fields(data, {"ref"}, {"orientation"}, "boundary reference")
        return cls(ref=str(data["ref"]), orientation=str(data.get("orientation", "unknown")))


@dataclass(frozen=True)
class RegionExpression:
    operator: str
    ref: str | None = None
    operands: tuple["RegionExpression", ...] = ()

    def __post_init__(self) -> None:
        if self.operator not in REGION_OPERATORS:
            raise InvalidGeometryInput(f"Unknown region expression operator: {self.operator}")
        if self.operator == "reference":
            if self.ref is None or self.operands:
                raise InvalidGeometryInput("Region references require one ref and no operands")
            validate_geometry_id(self.ref)
        elif self.ref is not None:
            raise InvalidGeometryInput(f"Region operator {self.operator} cannot define ref")
        elif self.operator in {"union", "intersection"} and len(self.operands) < 2:
            raise InvalidGeometryInput(f"Region operator {self.operator} requires at least two operands")
        elif self.operator == "difference" and len(self.operands) != 2:
            raise InvalidGeometryInput("Region operator difference requires two operands")

    def references(self) -> set[str]:
        if self.ref is not None:
            return {self.ref}
        return {ref for operand in self.operands for ref in operand.references()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "ref": self.ref,
            "operands": [operand.to_dict() for operand in self.operands],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegionExpression":
        _check_fields(data, {"operator"}, {"ref", "operands"}, "region expression")
        return cls(
            operator=str(data["operator"]),
            ref=str(data["ref"]) if data.get("ref") is not None else None,
            operands=tuple(cls.from_dict(item) for item in data.get("operands", ())),
        )


@dataclass(frozen=True)
class ParseRegion:
    id: str
    boundary: tuple[BoundaryReference, ...]
    origin: ParseOrigin
    confidence: float
    score_kind: str
    holes: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    expression: RegionExpression | None = None
    evidence_refs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_geometry_id(self.id)
        if not self.boundary and self.expression is None:
            raise InvalidGeometryInput(f"Parse region {self.id} requires a boundary or set expression")
        _validate_candidate(self.id, self.origin, self.confidence, self.score_kind, self.evidence_refs)
        for reference in (*self.holes, *self.components):
            validate_geometry_id(reference)
        _require_mapping(self.metadata, f"Parse region {self.id} metadata")

    def references(self) -> set[str]:
        refs = {*self.holes, *self.components}
        if self.expression is not None:
            refs.update(self.expression.references())
        return refs

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "boundary": [reference.to_dict() for reference in self.boundary],
            "origin": self.origin.value,
            "confidence": self.confidence,
            "score_kind": self.score_kind,
            "holes": list(self.holes),
            "components": list(self.components),
            "expression": self.expression.to_dict() if self.expression is not None else None,
            "evidence_refs": list(self.evidence_refs),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParseRegion":
        _check_fields(
            data,
            {"id", "boundary", "origin", "confidence", "score_kind"},
            {"holes", "components", "expression", "evidence_refs", "metadata"},
            "parse region",
        )
        return cls(
            id=str(data["id"]),
            boundary=tuple(BoundaryReference.from_dict(item) for item in data["boundary"]),
            origin=ParseOrigin(data["origin"]),
            confidence=float(data["confidence"]),
            score_kind=str(data["score_kind"]),
            holes=_string_tuple(data.get("holes")),
            components=_string_tuple(data.get("components")),
            expression=RegionExpression.from_dict(data["expression"]) if data.get("expression") is not None else None,
            evidence_refs=_string_tuple(data.get("evidence_refs")),
            metadata=_metadata(data),
        )


@dataclass(frozen=True)
class ParseTarget:
    id: str
    kind: str
    mode: str
    arguments: tuple[ParseReference, ...]
    origin: ParseOrigin
    confidence: float
    score_kind: str
    expression: MathExpression | None = None
    evidence_refs: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_geometry_id(self.id)
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise InvalidGeometryInput("Parse target kind must be non-empty")
        if self.mode not in {"value", "proof", "constraint"}:
            raise InvalidGeometryInput(f"Unknown parse target mode: {self.mode}")
        _validate_candidate(self.id, self.origin, self.confidence, self.score_kind, self.evidence_refs)
        for dependency in self.dependencies:
            validate_geometry_id(dependency)
        _require_mapping(self.metadata, f"Parse target {self.id} metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "mode": self.mode,
            "arguments": [argument.to_dict() for argument in self.arguments],
            "origin": self.origin.value,
            "confidence": self.confidence,
            "score_kind": self.score_kind,
            "expression": self.expression.to_dict() if self.expression is not None else None,
            "evidence_refs": list(self.evidence_refs),
            "dependencies": list(self.dependencies),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParseTarget":
        _check_fields(
            data,
            {"id", "kind", "mode", "arguments", "origin", "confidence", "score_kind"},
            {"expression", "evidence_refs", "dependencies", "metadata"},
            "parse target",
        )
        return cls(
            id=str(data["id"]),
            kind=str(data["kind"]),
            mode=str(data["mode"]),
            arguments=tuple(ParseReference.from_dict(item) for item in data["arguments"]),
            origin=ParseOrigin(data["origin"]),
            confidence=float(data["confidence"]),
            score_kind=str(data["score_kind"]),
            expression=MathExpression.from_dict(data["expression"]) if data.get("expression") is not None else None,
            evidence_refs=_string_tuple(data.get("evidence_refs")),
            dependencies=_string_tuple(data.get("dependencies")),
            metadata=_metadata(data),
        )


@dataclass(frozen=True)
class ParseConflictGroup:
    id: str
    members: tuple[str, ...]
    status: ConflictStatus = ConflictStatus.UNRESOLVED
    selected: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        validate_geometry_id(self.id)
        if len(self.members) < 2 or len(set(self.members)) != len(self.members):
            raise InvalidGeometryInput(f"Parse conflict {self.id} requires at least two unique members")
        for member in self.members:
            validate_geometry_id(member)
        _require_enum(self.status, ConflictStatus, "parse conflict status")
        if self.status is ConflictStatus.RESOLVED and self.selected not in self.members:
            raise InvalidGeometryInput(f"Resolved parse conflict {self.id} must select one member")
        if self.status is ConflictStatus.UNRESOLVED and self.selected is not None:
            raise InvalidGeometryInput(f"Unresolved parse conflict {self.id} cannot select a member")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "members": list(self.members),
            "status": self.status.value,
            "selected": self.selected,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParseConflictGroup":
        _check_fields(data, {"id", "members"}, {"status", "selected", "reason"}, "parse conflict group")
        return cls(
            id=str(data["id"]),
            members=_string_tuple(data["members"]),
            status=ConflictStatus(data.get("status", ConflictStatus.UNRESOLVED.value)),
            selected=str(data["selected"]) if data.get("selected") is not None else None,
            reason=str(data.get("reason", "")),
        )


@dataclass(frozen=True)
class ParseGraph:
    id: str
    source_id: str
    evidence: tuple[ParseEvidence, ...] = ()
    objects: tuple[ParseObject, ...] = ()
    relations: tuple[ParseRelation, ...] = ()
    quantities: tuple[ParseQuantity, ...] = ()
    regions: tuple[ParseRegion, ...] = ()
    targets: tuple[ParseTarget, ...] = ()
    conflicts: tuple[ParseConflictGroup, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = PARSE_GRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        validate_geometry_id(self.id)
        validate_geometry_id(self.source_id)
        if self.schema_version != PARSE_GRAPH_SCHEMA_VERSION:
            raise InvalidGeometryInput(f"Unsupported parse graph schema version: {self.schema_version}")
        _require_mapping(self.metadata, f"Parse graph {self.id} metadata")
        self._validate_graph()

    def _validate_graph(self) -> None:
        evidence_ids = _unique_ids(self.evidence, "parse evidence")
        candidate_groups: tuple[Sequence[Any], ...] = (
            self.objects,
            self.relations,
            self.quantities,
            self.regions,
            self.targets,
            self.conflicts,
        )
        candidate_ids: set[str] = set()
        for group in candidate_groups:
            group_ids = _unique_ids(group, "parse candidate")
            duplicate = candidate_ids & group_ids
            if duplicate:
                raise InvalidGeometryInput(f"Duplicate parse candidate IDs: {sorted(duplicate)}")
            candidate_ids.update(group_ids)
        overlap = evidence_ids & candidate_ids
        if overlap:
            raise InvalidGeometryInput(f"Evidence and parse candidates share IDs: {sorted(overlap)}")

        object_ids = {item.id for item in self.objects}
        region_ids = {item.id for item in self.regions}
        quantity_ids = {item.id for item in self.quantities}
        relation_ids = {item.id for item in self.relations}
        referable_ids = object_ids | region_ids | quantity_ids

        for item in (*self.objects, *self.relations, *self.quantities, *self.regions, *self.targets):
            missing = set(item.evidence_refs) - evidence_ids
            if missing:
                raise InvalidGeometryInput(f"Parse candidate {item.id} references unknown evidence: {sorted(missing)}")
        for item in self.objects:
            _require_references(item.id, item.references, object_ids)
        for item in self.relations:
            _require_references(item.id, item.arguments, referable_ids)
            missing = set(item.dependencies) - candidate_ids
            if missing:
                raise InvalidGeometryInput(f"Parse relation {item.id} has unknown dependencies: {sorted(missing)}")
        for item in self.quantities:
            _require_references(item.id, item.arguments, object_ids | region_ids)
            missing = set(item.dependencies) - candidate_ids
            if missing:
                raise InvalidGeometryInput(f"Parse quantity {item.id} has unknown dependencies: {sorted(missing)}")
            if item.id in item.dependencies:
                raise InvalidGeometryInput(f"Parse quantity {item.id} cannot depend on itself")
        for item in self.targets:
            _require_references(item.id, item.arguments, referable_ids)
            missing = set(item.dependencies) - candidate_ids
            if missing:
                raise InvalidGeometryInput(f"Parse target {item.id} has unknown dependencies: {sorted(missing)}")
            if item.id in item.dependencies:
                raise InvalidGeometryInput(f"Parse target {item.id} cannot depend on itself")
        for item in self.regions:
            missing_boundary = {reference.ref for reference in item.boundary} - object_ids
            if missing_boundary:
                raise InvalidGeometryInput(f"Parse region {item.id} has unknown boundary objects: {sorted(missing_boundary)}")
            missing_regions = item.references() - region_ids
            if missing_regions:
                raise InvalidGeometryInput(f"Parse region {item.id} references unknown regions: {sorted(missing_regions)}")
            if item.id in item.references():
                raise InvalidGeometryInput(f"Parse region {item.id} cannot reference itself")
        _check_region_cycles(self.regions)
        for conflict in self.conflicts:
            missing = set(conflict.members) - (candidate_ids - {conflict.id})
            if missing:
                raise InvalidGeometryInput(f"Parse conflict {conflict.id} references unknown members: {sorted(missing)}")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "source_id": self.source_id,
            "evidence": [item.to_dict() for item in sorted(self.evidence, key=lambda item: item.id)],
            "objects": [item.to_dict() for item in sorted(self.objects, key=lambda item: item.id)],
            "relations": [item.to_dict() for item in sorted(self.relations, key=lambda item: item.id)],
            "quantities": [item.to_dict() for item in sorted(self.quantities, key=lambda item: item.id)],
            "regions": [item.to_dict() for item in sorted(self.regions, key=lambda item: item.id)],
            "targets": [item.to_dict() for item in sorted(self.targets, key=lambda item: item.id)],
            "conflicts": [item.to_dict() for item in sorted(self.conflicts, key=lambda item: item.id)],
            "metadata": dict(self.metadata),
        }

    def point_names(self) -> dict[str, str]:
        """Project unique point names represented by explicit binding relations."""

        objects = {item.id: item for item in self.objects}
        candidates: dict[str, set[str]] = {}
        for relation in self.relations:
            if relation.predicate != "binds" or relation.metadata.get("binding_kind") != "point_name":
                continue
            label_id = next((item.ref for item in relation.arguments if item.role == "label"), None)
            point_id = next((item.ref for item in relation.arguments if item.role == "object"), None)
            label = objects.get(label_id)
            point = objects.get(point_id)
            if (
                label is None
                or label.kind is not ParseObjectKind.TEXT
                or label.visible_name is None
                or point is None
                or point.kind is not ParseObjectKind.POINT
            ):
                continue
            candidates.setdefault(point.id, set()).add(label.visible_name.strip())
        return {point_id: next(iter(names)) for point_id, names in candidates.items() if len(names) == 1}

    def checksum(self) -> str:
        return stable_checksum(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["checksum"] = stable_checksum(payload)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParseGraph":
        _check_fields(
            data,
            {"schema_version", "id", "source_id", "evidence", "objects", "relations", "quantities", "regions", "targets", "conflicts", "metadata", "checksum"},
            set(),
            "parse graph",
        )
        checksum_payload = {key: value for key, value in data.items() if key != "checksum"}
        if data["checksum"] != stable_checksum(checksum_payload):
            raise InvalidGeometryInput("Parse graph checksum does not match payload")
        graph = cls(
            id=str(data["id"]),
            source_id=str(data["source_id"]),
            evidence=tuple(ParseEvidence.from_dict(item) for item in data["evidence"]),
            objects=tuple(ParseObject.from_dict(item) for item in data["objects"]),
            relations=tuple(ParseRelation.from_dict(item) for item in data["relations"]),
            quantities=tuple(ParseQuantity.from_dict(item) for item in data["quantities"]),
            regions=tuple(ParseRegion.from_dict(item) for item in data["regions"]),
            targets=tuple(ParseTarget.from_dict(item) for item in data["targets"]),
            conflicts=tuple(ParseConflictGroup.from_dict(item) for item in data["conflicts"]),
            metadata=_mapping_field(data, "metadata", "Parse graph metadata"),
            schema_version=str(data["schema_version"]),
        )
        if graph.checksum() != data["checksum"]:
            raise InvalidGeometryInput("Parse graph canonical checksum does not match payload")
        return graph


def _check_fields(data: Mapping[str, Any], required: set[str], optional: set[str], context: str) -> None:
    if not isinstance(data, Mapping):
        raise InvalidGeometryInput(f"{context.capitalize()} must be a mapping")
    keys = set(data)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise InvalidGeometryInput(f"{context.capitalize()} is missing fields: {', '.join(missing)}")
    if unknown:
        raise InvalidGeometryInput(f"{context.capitalize()} has unknown fields: {', '.join(unknown)}")


def _metadata(data: Mapping[str, Any]) -> dict[str, Any]:
    metadata = data["metadata"] if "metadata" in data else {}
    if not isinstance(metadata, Mapping):
        raise InvalidGeometryInput("Parse metadata must be a mapping")
    return dict(metadata)


def _mapping_field(data: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    value = data[key] if key in data else {}
    _require_mapping(value, context)
    return dict(value)


def _require_mapping(value: Any, context: str) -> None:
    if not isinstance(value, Mapping):
        raise InvalidGeometryInput(f"{context} must be a mapping")


def _string_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise InvalidGeometryInput("Expected a sequence of strings")
    return tuple(str(value) for value in values)


def _float_tuple(values: Any, length: int) -> tuple[float, ...] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != length:
        raise InvalidGeometryInput(f"Expected {length} numeric values")
    return tuple(float(value) for value in values)


def _int_tuple(values: Any, length: int) -> tuple[int, ...] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != length:
        raise InvalidGeometryInput(f"Expected {length} integer values")
    if any(not isinstance(value, int) for value in values):
        raise InvalidGeometryInput(f"Expected {length} integer values")
    return tuple(values)


def _require_enum(value: Any, enum_type: type[Enum], name: str) -> None:
    if not isinstance(value, enum_type):
        raise InvalidGeometryInput(f"Unknown {name}: {value}")


def _validate_finite_tuple(values: Sequence[float], length: int, name: str) -> None:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or len(values) != length
        or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values)
    ):
        raise InvalidGeometryInput(f"{name.capitalize()} must contain {length} finite numbers")


def _validate_candidate(
    candidate_id: str,
    origin: ParseOrigin,
    confidence: float,
    score_kind: str,
    evidence_refs: Sequence[str],
) -> None:
    _require_enum(origin, ParseOrigin, "parse origin")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise InvalidGeometryInput(f"Parse candidate {candidate_id} confidence must be in [0, 1]")
    if not isinstance(score_kind, str) or not score_kind.strip():
        raise InvalidGeometryInput(f"Parse candidate {candidate_id} score_kind must be non-empty")
    for evidence_id in evidence_refs:
        validate_geometry_id(evidence_id)


def _validate_object_geometry(candidate: ParseObject) -> None:
    _require_mapping(candidate.geometry, f"Parse object {candidate.id} geometry")
    allowed = _OBJECT_GEOMETRY_FIELDS[candidate.kind]
    unknown = sorted(set(candidate.geometry) - allowed)
    if unknown:
        raise InvalidGeometryInput(
            f"Parse object {candidate.id} has geometry fields incompatible with {candidate.kind.value}: {', '.join(unknown)}"
        )

    for key in ("pixel", "p1", "p2", "center"):
        if key in candidate.geometry:
            _validate_finite_tuple(candidate.geometry[key], 2, f"parse object {candidate.id} {key}")
    if "bbox" in candidate.geometry:
        bbox = candidate.geometry["bbox"]
        _validate_finite_tuple(bbox, 4, f"parse object {candidate.id} bbox")
        if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
            raise InvalidGeometryInput(f"Parse object {candidate.id} has an inverted bbox")
    if "angle_deg" in candidate.geometry:
        _validate_finite_number(candidate.geometry["angle_deg"], f"parse object {candidate.id} angle_deg")
    if "radius_px" in candidate.geometry:
        radius = _validate_finite_number(candidate.geometry["radius_px"], f"parse object {candidate.id} radius_px")
        if radius <= 0:
            raise InvalidGeometryInput(f"Parse object {candidate.id} radius_px must be positive")
    if "orientation" in candidate.geometry and candidate.geometry["orientation"] not in {"ccw", "cw"}:
        raise InvalidGeometryInput(f"Parse object {candidate.id} arc orientation must be 'ccw' or 'cw'")
    if "exact_coords" in candidate.geometry:
        values = candidate.geometry["exact_coords"]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) not in {2, 3}:
            raise InvalidGeometryInput(f"Parse object {candidate.id} exact_coords must contain two or three expressions")
        for value in values:
            _validate_expression_payload(value, f"Parse object {candidate.id} exact_coords")
    for key in ("exact_radius", "exact_central_angle"):
        if key in candidate.geometry:
            expression = _validate_expression_payload(candidate.geometry[key], f"Parse object {candidate.id} {key}")
            value = expression.to_sympy()
            if value.is_number and value.is_positive is not True:
                raise InvalidGeometryInput(f"Parse object {candidate.id} {key} must be positive")


def _validate_expression_payload(value: Any, context: str) -> MathExpression:
    if not isinstance(value, Mapping):
        raise InvalidGeometryInput(f"{context} must use a MathExpression mapping")
    try:
        return MathExpression.from_dict(value)
    except InvalidGeometryInput as exc:
        raise InvalidGeometryInput(f"{context} is invalid: {exc}") from exc


def _validate_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InvalidGeometryInput(f"{name.capitalize()} must be a finite number")
    return float(value)


def _unique_ids(items: Sequence[Any], kind: str) -> set[str]:
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise InvalidGeometryInput(f"Duplicate {kind} IDs")
    return set(ids)


def _require_references(owner_id: str, references: Sequence[ParseReference], known_ids: set[str]) -> None:
    missing = {reference.ref for reference in references} - known_ids
    if missing:
        raise InvalidGeometryInput(f"Parse candidate {owner_id} references unknown candidates: {sorted(missing)}")


def _check_region_cycles(regions: Sequence[ParseRegion]) -> None:
    dependencies = {region.id: region.references() for region in regions}
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(region_id: str) -> None:
        if region_id in visiting:
            raise InvalidGeometryInput(f"Parse region dependency cycle includes {region_id}")
        if region_id in visited:
            return
        visiting.add(region_id)
        for dependency in dependencies[region_id]:
            visit(dependency)
        visiting.remove(region_id)
        visited.add(region_id)

    for region_id in sorted(dependencies):
        visit(region_id)
