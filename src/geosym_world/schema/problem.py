"""Canonical GeoSym problem-state schema.

The schema is intentionally dataclass/dict based so existing JSONL pipelines can
adopt it without a new runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json


JsonDict = Dict[str, Any]


@dataclass
class GeoSymSource:
    type: str
    dataset: Optional[str] = None
    generator: Optional[str] = None
    original_id: Optional[str] = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "type": self.type,
            "dataset": self.dataset,
            "generator": self.generator,
            "original_id": self.original_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "GeoSymSource":
        return cls(
            type=data.get("type", "unknown"),
            dataset=data.get("dataset"),
            generator=data.get("generator"),
            original_id=data.get("original_id"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class GeoSymInput:
    image_path: Optional[str] = None
    problem_text: str = ""
    choices: Optional[List[str]] = None
    solution_text: Optional[str] = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "image_path": self.image_path,
            "problem_text": self.problem_text,
            "choices": self.choices,
            "solution_text": self.solution_text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "GeoSymInput":
        return cls(
            image_path=data.get("image_path"),
            problem_text=data.get("problem_text", ""),
            choices=data.get("choices"),
            solution_text=data.get("solution_text"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class GeoSymGeometry:
    points: List[JsonDict] = field(default_factory=list)
    lines: List[JsonDict] = field(default_factory=list)
    arcs: List[JsonDict] = field(default_factory=list)
    entities: List[JsonDict] = field(default_factory=list)
    relations: List[JsonDict] = field(default_factory=list)
    quantities: List[JsonDict] = field(default_factory=list)
    raw: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "points": self.points,
            "lines": self.lines,
            "arcs": self.arcs,
            "entities": self.entities,
            "relations": self.relations,
            "quantities": self.quantities,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "GeoSymGeometry":
        return cls(
            points=list(data.get("points") or []),
            lines=list(data.get("lines") or []),
            arcs=list(data.get("arcs") or []),
            entities=list(data.get("entities") or []),
            relations=list(data.get("relations") or []),
            quantities=list(data.get("quantities") or []),
            raw=dict(data.get("raw") or {}),
        )


@dataclass
class GeoSymTarget:
    type: str
    object: Optional[str] = None
    answer: Optional[JsonDict] = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "type": self.type,
            "object": self.object,
            "answer": self.answer,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "GeoSymTarget":
        return cls(
            type=data.get("type", "unknown"),
            object=data.get("object"),
            answer=data.get("answer"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class GeoSymProof:
    known_conditions: List[JsonDict] = field(default_factory=list)
    construction_trace: List[JsonDict] = field(default_factory=list)
    derivation_trace: List[JsonDict] = field(default_factory=list)
    required_steps: List[JsonDict] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return {
            "known_conditions": self.known_conditions,
            "construction_trace": self.construction_trace,
            "derivation_trace": self.derivation_trace,
            "required_steps": self.required_steps,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "GeoSymProof":
        return cls(
            known_conditions=list(data.get("known_conditions") or []),
            construction_trace=list(data.get("construction_trace") or []),
            derivation_trace=list(data.get("derivation_trace") or []),
            required_steps=list(data.get("required_steps") or []),
        )


@dataclass
class GeoSymDiagnostic:
    task_type: str = "unknown"
    difficulty: JsonDict = field(default_factory=dict)
    requires: List[str] = field(default_factory=list)
    failure_components: List[str] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "task_type": self.task_type,
            "difficulty": self.difficulty,
            "requires": self.requires,
            "failure_components": self.failure_components,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> "GeoSymDiagnostic":
        return cls(
            task_type=data.get("task_type", "unknown"),
            difficulty=dict(data.get("difficulty") or {}),
            requires=list(data.get("requires") or []),
            failure_components=list(data.get("failure_components") or []),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class VerificationIssue:
    severity: str
    code: str
    message: str
    location: Optional[str] = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "metadata": self.metadata,
        }


@dataclass
class VerificationReport:
    schema_valid: bool = True
    geometry_valid: bool = True
    quantity_valid: bool = True
    derivation_valid: bool = True
    image_alignment_valid: Optional[bool] = None
    answer_valid: Optional[bool] = None
    llm_judge_valid: Optional[bool] = None
    issues: List[VerificationIssue] = field(default_factory=list)
    metrics: JsonDict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        hard_flags = [
            self.schema_valid,
            self.geometry_valid,
            self.quantity_valid,
            self.derivation_valid,
        ]
        if self.answer_valid is not None:
            hard_flags.append(self.answer_valid)
        if self.image_alignment_valid is not None:
            hard_flags.append(self.image_alignment_valid)
        return all(hard_flags)

    def add_issue(
        self,
        code: str,
        message: str,
        severity: str = "error",
        location: Optional[str] = None,
        **metadata: Any,
    ) -> None:
        self.issues.append(
            VerificationIssue(
                severity=severity,
                code=code,
                message=message,
                location=location,
                metadata=metadata,
            )
        )
        if severity == "error":
            if code.startswith("schema."):
                self.schema_valid = False
            elif code.startswith("geometry."):
                self.geometry_valid = False
            elif code.startswith("quantity."):
                self.quantity_valid = False
            elif code.startswith("derivation."):
                self.derivation_valid = False
            elif code.startswith("answer."):
                self.answer_valid = False
            elif code.startswith("image."):
                self.image_alignment_valid = False

    def to_dict(self) -> JsonDict:
        return {
            "schema_valid": self.schema_valid,
            "geometry_valid": self.geometry_valid,
            "quantity_valid": self.quantity_valid,
            "derivation_valid": self.derivation_valid,
            "image_alignment_valid": self.image_alignment_valid,
            "answer_valid": self.answer_valid,
            "llm_judge_valid": self.llm_judge_valid,
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": self.metrics,
        }


@dataclass
class GeoSymProblemState:
    problem_id: str
    source: GeoSymSource
    input: GeoSymInput
    geometry: GeoSymGeometry
    target: Optional[GeoSymTarget] = None
    proof: GeoSymProof = field(default_factory=GeoSymProof)
    diagnostic: GeoSymDiagnostic = field(default_factory=GeoSymDiagnostic)
    verification: Optional[VerificationReport] = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "problem_id": self.problem_id,
            "source": self.source.to_dict(),
            "input": self.input.to_dict(),
            "geometry": self.geometry.to_dict(),
            "target": self.target.to_dict() if self.target else None,
            "proof": self.proof.to_dict(),
            "diagnostic": self.diagnostic.to_dict(),
            "verification": self.verification.to_dict() if self.verification else None,
            "metadata": self.metadata,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "GeoSymProblemState":
        target_data = data.get("target")
        verification = None
        if isinstance(data.get("verification"), dict):
            report_data = data["verification"]
            verification = VerificationReport(
                schema_valid=report_data.get("schema_valid", True),
                geometry_valid=report_data.get("geometry_valid", True),
                quantity_valid=report_data.get("quantity_valid", True),
                derivation_valid=report_data.get("derivation_valid", True),
                image_alignment_valid=report_data.get("image_alignment_valid"),
                answer_valid=report_data.get("answer_valid"),
                llm_judge_valid=report_data.get("llm_judge_valid"),
                metrics=dict(report_data.get("metrics") or {}),
            )
            for issue in report_data.get("issues") or []:
                verification.issues.append(
                    VerificationIssue(
                        severity=issue.get("severity", "error"),
                        code=issue.get("code", "unknown"),
                        message=issue.get("message", ""),
                        location=issue.get("location"),
                        metadata=dict(issue.get("metadata") or {}),
                    )
                )
        return cls(
            problem_id=data.get("problem_id", ""),
            source=GeoSymSource.from_dict(data.get("source") or {}),
            input=GeoSymInput.from_dict(data.get("input") or {}),
            geometry=GeoSymGeometry.from_dict(data.get("geometry") or {}),
            target=GeoSymTarget.from_dict(target_data) if isinstance(target_data, dict) else None,
            proof=GeoSymProof.from_dict(data.get("proof") or {}),
            diagnostic=GeoSymDiagnostic.from_dict(data.get("diagnostic") or {}),
            verification=verification,
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def from_json(cls, text: str) -> "GeoSymProblemState":
        return cls.from_dict(json.loads(text))
