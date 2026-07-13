"""GeoClaw: solver-grounded tool-use environment for geometry agents."""

from .environment import GeoClawEnvironment
from .schema import AgentAction, DifficultyProfile, EvidenceRecord, GeoClawObservation, ToolFeedback, VisualRegion
from .scoring import aggregate_scores, feedback_gain, score_trajectory
from .task import GeoClawTask

__all__ = [
    "AgentAction", "DifficultyProfile", "EvidenceRecord", "GeoClawEnvironment", "GeoClawObservation", "GeoClawTask", "ToolFeedback", "VisualRegion",
    "aggregate_scores", "feedback_gain", "score_trajectory",
]
