from geosym_world_geometry.geoclaw import GeoClawEnvironment
from geosym_world_geometry.geoclaw.datasets import pilot_task
from geosym_world_geometry.geoclaw.scoring import aggregate_scores, feedback_gain, score_trajectory
from geosym_world_geometry.geoclaw.runners import run_batch


def test_tool_selection_and_useful_action_metrics(tmp_path):
    task = pilot_task("construct_midpoint")
    environment = GeoClawEnvironment(task, output_dir=tmp_path, render_steps=False)
    environment.replay([
        {"action": "construct", "arguments": {"operation": "midpoint", "id": "M", "params": {"of": ["A", "B"]}}},
        {"action": "verify", "arguments": {"claim": {"type": "midpoint", "object": "M"}}},
        {"action": "final", "arguments": {"answer": "4"}},
    ])
    metrics = environment.trajectory.metrics
    assert metrics["tool_selection_f1"] == 1.0
    assert metrics["argument_validity"] == 1.0
    assert metrics["verified_step_ratio"] == 1.0
    assert metrics["useful_action_rate"] == 1.0
    assert metrics["construction_validity"] == 1.0
    assert metrics["construction_utility"] == 1.0


def test_recovery_rate_counts_corrected_failure(tmp_path):
    task = pilot_task("construct_midpoint")
    environment = GeoClawEnvironment(task, output_dir=tmp_path, render_steps=False)
    environment.replay([
        {"action": "construct", "arguments": {"operation": "midpoint", "id": "M", "params": {"of": ["A", "Z"]}}},
        {"action": "construct", "arguments": {"operation": "midpoint", "id": "M", "params": {"of": ["A", "B"]}}},
        {"action": "final", "arguments": {"answer": "4"}},
    ])
    metrics = environment.trajectory.metrics
    assert metrics["num_failure_events"] == 1
    assert metrics["num_recovered_events"] == 1
    assert metrics["recovery_rate"] == 1.0


def test_aggregate_and_feedback_gain():
    visible = [{"final_accuracy": 1.0, "tool_selection_f1": 1.0, "argument_validity": 1.0, "invalid_action_rate": 0.0, "verified_step_ratio": 1.0, "useful_action_rate": 1.0, "tool_efficiency": 1.0, "recovery_rate": 1.0}]
    hidden = [{**visible[0], "final_accuracy": 0.0, "recovery_rate": 0.0}]
    assert aggregate_scores(visible)["num_episodes"] == 1
    gain = feedback_gain(visible, hidden)
    assert gain["final_accuracy_gain"] == 1.0
    assert gain["recovery_rate_gain"] == 1.0


def test_batch_runner_writes_manifest(tmp_path):
    tasks = [pilot_task("construct_midpoint"), pilot_task("triangle_area")]
    manifest = run_batch(tasks, lambda task: [{"action": "final", "arguments": {"answer": task.answer}}], tmp_path, render_steps=False)
    assert manifest["num_tasks"] == 2
    assert manifest["aggregate"]["final_accuracy"] == 1.0
    assert manifest["aggregate"]["expression_validity"] is None
    assert (tmp_path / "manifest.json").exists()


def test_expression_and_zoom_metrics(tmp_path):
    task = pilot_task("visual_length")
    environment = GeoClawEnvironment(task, output_dir=tmp_path, render_steps=True)
    fact_id = environment.facts.find("length", "AB")[0].id
    environment.replay([
        {"action": "zoom", "arguments": {"region": [0.05, 0.70, 0.65, 0.98]}},
        {"action": "calculate", "arguments": {"expression": "4", "fact_ids": [fact_id]}},
        {"action": "final", "arguments": {"answer": "4"}},
    ])
    metrics = environment.trajectory.metrics
    assert metrics["expression_validity"] == 1.0
    assert metrics["geometry_expression_support_rate"] == 1.0
    assert metrics["zoom_grounding_iou"] == 1.0
    assert metrics["required_region_coverage_rate"] == 1.0
    assert metrics["calculation_masking_rate"] == 0.0


def test_calculation_masking_and_measurement_audit_metrics(tmp_path):
    task = pilot_task("visual_length")
    environment = GeoClawEnvironment(task, output_dir=tmp_path, render_steps=False)
    environment.replay([
        {"action": "measure", "arguments": {"measurement": "length", "object": "AB", "mode": "image"}},
        {"action": "calculate", "arguments": {"expression": "2+2"}},
        {"action": "final", "arguments": {"answer": "4"}},
    ])
    metrics = environment.trajectory.metrics
    assert metrics["visual_solver_residual_mean"] == 0.0
    assert metrics["calculation_masking_rate"] == 1.0
    assert metrics["measurement_reliance_error"] == 0.0
