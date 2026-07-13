import pytest

from geosym_world_geometry.geoclaw.schema import AgentAction, DifficultyProfile, EvidenceRecord, VisualRegion
from geosym_world_geometry.geoclaw.datasets import pilot_task
from geosym_world_geometry.geoclaw.task import GeoClawTask


def test_action_schema_round_trip():
    action = AgentAction.from_dict({"thought": "inspect", "action": "zoom", "arguments": {"region": [0, 0, 1, 1]}})
    assert action.action == "zoom"
    assert action.to_dict()["thought"] == "inspect"


def test_action_schema_rejects_unknown_action():
    with pytest.raises(ValueError):
        AgentAction.from_dict({"action": "python", "arguments": {}})


def test_action_schema_rejects_non_object_arguments():
    with pytest.raises(ValueError):
        AgentAction.from_dict({"action": "verify", "arguments": []})


def test_evidence_and_visual_region_schemas():
    evidence = EvidenceRecord(level="symbolic_proved", claim={"type": "perpendicular"}, source="solver")
    assert evidence.to_dict()["level"] == "symbolic_proved"
    region = VisualRegion.from_dict({"id": "r1", "region": [0.1, 0.2, 0.5, 0.8], "required": True})
    assert region.required


def test_task_round_trip_preserves_difficulty_and_regions():
    task = pilot_task("visual_length")
    restored = GeoClawTask.from_dict(task.to_dict())
    assert restored.visual_regions[0].id == "target_label"
    assert restored.difficulty.visual_complexity == 3
