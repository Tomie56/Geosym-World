from pathlib import Path

from geosym_world_geometry.geoclaw import GeoClawEnvironment
from geosym_world_geometry.geoclaw.datasets import pilot_task


def make_env(tmp_path, task="construct_midpoint", render=False):
    return GeoClawEnvironment(pilot_task(task), output_dir=tmp_path, render_steps=render)


def feedback(environment):
    return environment.trajectory.steps[-1].feedback


def test_calculator_returns_exact_result(tmp_path):
    environment = make_env(tmp_path)
    environment.step({"action": "calculate", "arguments": {"expression": "sqrt(3**2 + 4**2)"}})
    assert feedback(environment)["status"] == "valid"
    assert feedback(environment)["result"]["exact_value"] == "5"
    assert feedback(environment)["result"]["provenance"]["geometry_supported"] is False


def test_calculator_records_geometry_fact_provenance(tmp_path):
    environment = make_env(tmp_path)
    fact_id = environment.facts.find("length", "AB")[0].id
    environment.step({"action": "calculate", "arguments": {"expression": "4**2", "fact_ids": [fact_id]}})
    provenance = feedback(environment)["result"]["provenance"]
    assert provenance["geometry_supported"] is True
    assert provenance["fact_ids"] == [fact_id]


def test_calculator_rejects_unsafe_expression(tmp_path):
    environment = make_env(tmp_path)
    environment.step({"action": "calculate", "arguments": {"expression": "__import__('os')"}})
    assert feedback(environment)["status"] == "invalid_argument"


def test_construct_is_transactional_on_failure(tmp_path):
    environment = make_env(tmp_path)
    before = environment.resolver.visible_object_ids(environment.state)
    environment.step({"action": "construct", "arguments": {"operation": "midpoint", "id": "M", "params": {"of": ["A", "Z"]}}})
    assert feedback(environment)["status"] == "unknown_object"
    assert environment.resolver.visible_object_ids(environment.state) == before
    assert environment.agent_trace == []


def test_construct_and_verify_midpoint(tmp_path):
    environment = make_env(tmp_path)
    environment.step({"action": "construct", "arguments": {"operation": "midpoint", "id": "M", "params": {"of": ["A", "B"]}}})
    assert "M" in environment.state.points
    assert feedback(environment)["status"] == "valid"
    environment.step({"action": "verify", "arguments": {"claim": {"type": "midpoint", "object": "M"}}})
    assert feedback(environment)["result"]["truth"] == "true"


def test_construct_resolves_point_and_line_aliases(tmp_path):
    environment = make_env(tmp_path)
    environment.step({"action": "construct", "arguments": {"operation": "foot_of_perpendicular", "id": "H", "params": {"point": "point C", "line": "segment BA"}}})
    assert feedback(environment)["status"] == "valid"
    assert "H" in environment.state.points


def test_verify_false_value_and_unknown_claim(tmp_path):
    environment = make_env(tmp_path)
    environment.step({"action": "verify", "arguments": {"claim": {"type": "length", "object": "AB", "value": "5"}}})
    assert feedback(environment)["result"]["truth"] == "false"
    assert feedback(environment)["result"]["evidence_strength"] == "counterexample_found"
    environment.step({"action": "verify", "arguments": {"claim": {"type": "cyclic", "objects": ["A", "B", "C"]}}})
    assert feedback(environment)["status"] == "not_verifiable"


def test_measure_oracle_and_image_modes(tmp_path):
    environment = make_env(tmp_path)
    environment.step({"action": "measure", "arguments": {"measurement": "length", "object": "AB", "mode": "oracle"}})
    assert feedback(environment)["result"]["exact_value"] == "4"
    environment.step({"action": "measure", "arguments": {"measurement": "length", "object": "AB", "mode": "image"}})
    assert feedback(environment)["result"]["exact_value"] is None
    assert feedback(environment)["result"]["numeric_value"] == 4.0
    assert feedback(environment)["result"]["backend"] == "render_scene_pixels"
    assert feedback(environment)["result"]["pixel_value"] > 0
    assert environment.trajectory.steps[-1].audit["visual_solver_residual"] == 0.0


def test_resolver_supports_structural_line_alias(tmp_path):
    environment = make_env(tmp_path)
    resolution = environment.resolver.resolve("segment BA", environment.state, ("segment",))
    assert resolution.status == "resolved"
    assert resolution.object_id == "AB"


def test_measure_and_verify_consume_aliases(tmp_path):
    environment = make_env(tmp_path)
    environment.step({"action": "measure", "arguments": {"measurement": "length", "object": "segment BA", "mode": "oracle"}})
    assert feedback(environment)["result"]["exact_value"] == "4"
    environment.step({"action": "verify", "arguments": {"claim": {"type": "length", "object": "segment BA", "value": "4"}}})
    assert feedback(environment)["result"]["truth"] == "true"


def test_zoom_writes_crop(tmp_path):
    environment = make_env(tmp_path, render=True)
    environment.step({"action": "zoom", "arguments": {"region": [0.1, 0.1, 0.8, 0.8], "scale": 2}})
    crop_path = Path(feedback(environment)["result"]["path"])
    assert crop_path.exists()
    assert environment.crops[0]["id"] == "crop_001"


def test_zoom_scores_annotated_region(tmp_path):
    environment = make_env(tmp_path, task="visual_length", render=True)
    environment.step({"action": "zoom", "arguments": {"region": [0.05, 0.70, 0.65, 0.98]}})
    result = feedback(environment)["result"]
    assert result["best_region_iou"] == 1.0
    assert result["required_region_covered"] is True


def test_final_terminates_and_scores_answer(tmp_path):
    environment = make_env(tmp_path)
    observation = environment.step({"action": "final", "arguments": {"answer": "4"}})
    assert observation.terminated
    assert environment.trajectory.metrics["final_accuracy"] == 1.0


def test_public_trajectory_hides_private_audit(tmp_path):
    environment = make_env(tmp_path)
    environment.step({"action": "measure", "arguments": {"measurement": "length", "object": "AB", "mode": "image"}})
    private_step = environment.trajectory.to_dict(include_private=True)["steps"][0]
    public_step = environment.trajectory.to_dict(include_private=False)["steps"][0]
    assert "audit" in private_step
    assert "audit" not in public_step
