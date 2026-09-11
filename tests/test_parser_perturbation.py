import cv2
import numpy as np

from geosym_world.benchmark import ParserPerturbationBenchmark, build_parser_benchmark_pair
from geosym_world.construction.builder import ConstructionBuilder
from geosym_world.parser import ParserInput
from geosym_world.rendering import state_to_render_scene


def test_parser_perturbation_benchmark_is_deterministic_and_private(tmp_path):
    image_path = tmp_path / "source.png"
    image = np.full((220, 220, 3), 255, dtype=np.uint8)
    cv2.line(image, (25, 180), (190, 180), (0, 0, 0), 3)
    cv2.line(image, (25, 180), (25, 35), (0, 0, 0), 3)
    cv2.line(image, (25, 35), (190, 180), (0, 0, 0), 3)
    cv2.imwrite(str(image_path), image)
    trace = [
        {"op": "add_point", "id": "A", "coord": [0, 0]},
        {"op": "add_point", "id": "B", "coord": [4, 0]},
        {"op": "add_point", "id": "C", "coord": [0, 3]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "AC", "points": ["A", "C"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "add_polygon", "id": "ABC", "vertices": ["A", "B", "C"]},
        {"op": "set_target", "type": "area", "object": "ABC"},
    ]
    state = ConstructionBuilder().execute(trace)
    scene = state_to_render_scene(state, scene_id="perturbation_fixture")
    record = {"id": "perturbation_fixture", "question": "Find the area.", "image": {"formats": {"png": str(image_path)}}}
    public_input, reference = build_parser_benchmark_pair(record, state=state, scene=scene)

    first = ParserPerturbationBenchmark().run(ParserInput.from_dict(public_input), reference, output_dir=tmp_path / "run_a", seed=17)
    second = ParserPerturbationBenchmark().run(ParserInput.from_dict(public_input), reference, output_dir=tmp_path / "run_b", seed=17)

    assert [row["variant"] for row in first["variants"]] == ["clean", "noise", "blur", "low_contrast", "rotate_2deg"]
    assert [row["metrics"] for row in first["variants"]] == [row["metrics"] for row in second["variants"]]
    assert all("private_supervision" not in row["input"] for row in first["variants"])
    assert (tmp_path / "run_a" / "report.json").is_file()
