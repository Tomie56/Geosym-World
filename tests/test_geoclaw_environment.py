import pytest

from geosym_world_geometry.geoclaw import GeoClawEnvironment
from geosym_world_geometry.geoclaw.datasets import built_in_pilot_tasks, pilot_task


def test_pilot_contains_ten_tasks_across_categories():
    tasks = built_in_pilot_tasks()
    assert len(tasks) == 10
    assert {task.metadata["category"] for task in tasks} == {"visual-hard", "numeric-hard", "construction-hard", "mixed-hard"}


def test_hidden_feedback_observation(tmp_path):
    environment = GeoClawEnvironment(pilot_task("construct_midpoint"), output_dir=tmp_path, expose_feedback=False, render_steps=False)
    observation = environment.step({"action": "calculate", "arguments": {"expression": "2+2"}})
    assert observation.verifier_feedback is None
    assert observation.action_history[-1]["feedback"] == {"status": "hidden"}


def test_invalid_schema_is_logged(tmp_path):
    environment = GeoClawEnvironment(pilot_task("construct_midpoint"), output_dir=tmp_path, render_steps=False)
    environment.step({"action": "unknown"})
    assert environment.trajectory.steps[-1].feedback["status"] == "invalid_schema"


def test_budget_exhaustion_terminates(tmp_path):
    task = pilot_task("verify_right_triangle")
    environment = GeoClawEnvironment(task, output_dir=tmp_path, render_steps=False)
    for _ in range(task.max_steps):
        if environment.terminated:
            break
        environment.step({"action": "calculate", "arguments": {"expression": "1+1"}})
    assert environment.terminated
    assert environment.trajectory.termination["reason"] == "budget_exhausted"


def test_replay_stops_after_final(tmp_path):
    environment = GeoClawEnvironment(pilot_task("construct_midpoint"), output_dir=tmp_path, render_steps=False)
    environment.replay([
        {"action": "final", "arguments": {"answer": "4"}},
        {"action": "calculate", "arguments": {"expression": "1+1"}},
    ])
    assert len(environment.trajectory.steps) == 1


def test_step_after_termination_raises(tmp_path):
    environment = GeoClawEnvironment(pilot_task("construct_midpoint"), output_dir=tmp_path, render_steps=False)
    environment.step({"action": "final", "arguments": {"answer": "4"}})
    with pytest.raises(RuntimeError):
        environment.step({"action": "final", "arguments": {"answer": "4"}})


def test_tool_allowlist_disables_actions(tmp_path):
    environment = GeoClawEnvironment(pilot_task("construct_midpoint"), output_dir=tmp_path, render_steps=False, allowed_tools=("calculate",))
    environment.step({"action": "construct", "arguments": {"operation": "midpoint", "id": "M", "params": {"of": ["A", "B"]}}})
    assert environment.trajectory.steps[-1].feedback["status"] == "unsupported_operation"
    assert environment.allowed_tools == ("calculate", "final")


def test_task_validation_fails_fast():
    task = pilot_task("construct_midpoint")
    assert task.validate() == []
    invalid = type(task)(**{**task.__dict__, "required_tool_sets": (("python",),)})
    assert invalid.validate(execute=False)[0]["code"] == "task.unknown_required_tool"
    wrong_answer = type(task)(**{**task.__dict__, "answer": "999"})
    assert any(issue["code"] == "task.answer_target_mismatch" for issue in wrong_answer.validate())
