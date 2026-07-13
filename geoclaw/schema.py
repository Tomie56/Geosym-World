"""Serializable schemas for the GeoClaw interactive environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


ACTION_NAMES = ("zoom", "calculate", "measure", "construct", "verify", "final")
EVIDENCE_LEVELS = ("symbolic_proved", "exact_coordinate", "solver_derived", "multi_instance_verified", "numeric_verified", "unknown", "counterexample_found", "invalid")


@dataclass(frozen=True)
class EvidenceRecord:
    level: str
    claim: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    fact_ids: tuple[str, ...] = ()
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.level not in EVIDENCE_LEVELS:
            raise ValueError(f"Unknown evidence level: {self.level}")

    def to_dict(self) -> Dict[str, Any]:
        return {"level": self.level, "claim": self.claim, "source": self.source, "fact_ids": list(self.fact_ids), "details": self.details}


@dataclass(frozen=True)
class VisualRegion:
    id: str
    region: tuple[float, float, float, float]
    purpose: str = "inspection"
    objects: tuple[str, ...] = ()
    required: bool = False

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VisualRegion":
        region = tuple(float(item) for item in payload["region"])
        if len(region) != 4 or not (0 <= region[0] < region[2] <= 1 and 0 <= region[1] < region[3] <= 1):
            raise ValueError(f"Invalid normalized visual region: {region}")
        return cls(id=str(payload["id"]), region=region, purpose=str(payload.get("purpose", "inspection")), objects=tuple(payload.get("objects", ())), required=bool(payload.get("required", False)))

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "region": list(self.region), "purpose": self.purpose, "objects": list(self.objects), "required": self.required}


@dataclass(frozen=True)
class DifficultyProfile:
    level: str = "high_school"
    category: str = "mixed"
    visual_complexity: int = 0
    construction_depth: int = 0
    theorem_depth: int = 0
    symbolic_growth: int = 0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "DifficultyProfile":
        payload = payload or {}
        return cls(
            level=str(payload.get("level", "high_school")), category=str(payload.get("category", "mixed")),
            visual_complexity=int(payload.get("visual_complexity", 0)), construction_depth=int(payload.get("construction_depth", 0)),
            theorem_depth=int(payload.get("theorem_depth", 0)), symbolic_growth=int(payload.get("symbolic_growth", 0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"level": self.level, "category": self.category, "visual_complexity": self.visual_complexity, "construction_depth": self.construction_depth, "theorem_depth": self.theorem_depth, "symbolic_growth": self.symbolic_growth}


@dataclass(frozen=True)
class AgentAction:
    action: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    thought: str = ""

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AgentAction":
        if not isinstance(payload, dict):
            raise ValueError("Action payload must be a JSON object")
        action = payload.get("action")
        if not isinstance(action, str) or action not in ACTION_NAMES:
            raise ValueError(f"Unknown action: {action}")
        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("Action arguments must be a JSON object")
        thought = payload.get("thought", "")
        if not isinstance(thought, str):
            raise ValueError("Action thought must be a string")
        return cls(action=action, arguments=dict(arguments), thought=thought)

    def to_dict(self) -> Dict[str, Any]:
        return {"thought": self.thought, "action": self.action, "arguments": self.arguments}


@dataclass(frozen=True)
class ToolFeedback:
    step_id: int
    action: str
    status: str
    utility: str
    result: Dict[str, Any] = field(default_factory=dict)
    evidence: tuple[Dict[str, Any], ...] = ()
    errors: tuple[Dict[str, Any], ...] = ()
    budget_used: int = 0
    budget_remaining: int = 0
    image_path: str | None = None
    evidence_strength: str = "unknown"

    @property
    def valid(self) -> bool:
        return self.status == "valid"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "status": self.status,
            "utility": self.utility,
            "result": self.result,
            "evidence": list(self.evidence),
            "errors": list(self.errors),
            "budget": {"used": self.budget_used, "remaining": self.budget_remaining},
            "image_path": self.image_path,
            "evidence_strength": self.evidence_strength,
        }


@dataclass(frozen=True)
class GeoClawObservation:
    task_id: str
    problem_text: str
    current_image: str | None
    crops: tuple[Dict[str, Any], ...]
    visible_objects: tuple[str, ...]
    action_history: tuple[Dict[str, Any], ...]
    verifier_feedback: Dict[str, Any] | None
    budget_used: int
    budget_remaining: int
    terminated: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "problem_text": self.problem_text,
            "current_image": self.current_image,
            "crops": list(self.crops),
            "visible_objects": list(self.visible_objects),
            "action_history": list(self.action_history),
            "verifier_feedback": self.verifier_feedback,
            "budget": {"used": self.budget_used, "remaining": self.budget_remaining},
            "terminated": self.terminated,
        }
