"""Small deterministic pilot tasks for GeoClaw development."""

from __future__ import annotations

from ..task import GeoClawTask
from ..schema import DifficultyProfile, VisualRegion


def built_in_pilot_tasks() -> list[GeoClawTask]:
    return [
        _triangle_length("visual_length", "The label is small. Inspect the figure and find the length of AB.", ("zoom", "measure")),
        _triangle_length("numeric_length", "Use calculation if needed. Find the length of BC.", ("calculate",), target="BC", answer="5"),
        _midpoint_task(),
        _altitude_task(),
        _parallel_task(),
        _intersection_task(),
        _circle_task(),
        _area_task(),
        _verify_task(),
        _mixed_task(),
    ]


def pilot_task(task_id: str) -> GeoClawTask:
    for task in built_in_pilot_tasks():
        if task.id == task_id:
            return task
    raise KeyError(task_id)


def _triangle_base(target: str, target_type: str) -> list[dict]:
    return [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]},
        {"op": "add_point", "id": "B", "coord": ["4", "0"]},
        {"op": "add_point", "id": "C", "coord": ["0", "3"]},
        {"op": "add_segment", "id": "AB", "points": ["A", "B"]},
        {"op": "add_segment", "id": "AC", "points": ["A", "C"]},
        {"op": "add_segment", "id": "BC", "points": ["B", "C"]},
        {"op": "add_polygon", "id": "ABC", "vertices": ["A", "B", "C"]},
        {"op": "set_target", "type": target_type, "object": target},
    ]


def _triangle_length(task_id: str, question: str, tools: tuple[str, ...], *, target: str = "AB", answer: str = "4") -> GeoClawTask:
    visual = "zoom" in tools
    return GeoClawTask(
        id=task_id, problem_text=question, trace=tuple(_triangle_base(target, "length")), answer=answer,
        required_tool_sets=(tools,), reference_cost=float(len(tools)), max_steps=8,
        visual_regions=(VisualRegion(id="target_label", region=(0.05, 0.70, 0.65, 0.98), purpose="small_label", objects=(target,), required=True),) if visual else (),
        difficulty=DifficultyProfile(level="high_school", category="visual-hard" if visual else "numeric-hard", visual_complexity=3 if visual else 1, construction_depth=0, theorem_depth=1, symbolic_growth=1),
        metadata={"category": "visual-hard" if "zoom" in tools else "numeric-hard"},
    )


def _midpoint_task() -> GeoClawTask:
    trace = _triangle_base("AB", "length")
    return GeoClawTask(
        id="construct_midpoint", problem_text="Construct midpoint M of AB, verify it, then answer AB.", trace=tuple(trace), answer="4",
        required_tool_sets=(("construct", "verify"),), reference_cost=2.5, max_steps=8,
        acceptable_intermediate_facts=({"type": "midpoint", "object": "M"},), metadata={"category": "construction-hard"},
        difficulty=DifficultyProfile(level="high_school", category="construction-hard", visual_complexity=1, construction_depth=1, theorem_depth=1, symbolic_growth=0),
    )


def _altitude_task() -> GeoClawTask:
    return GeoClawTask(
        id="construct_altitude", problem_text="Construct the perpendicular foot H from C to AB, then give the known side AC.", trace=tuple(_triangle_base("AC", "length")), answer="3",
        required_tool_sets=(("construct", "measure"),), reference_cost=2.5, max_steps=8,
        acceptable_intermediate_facts=({"type": "perpendicular", "object": "CH|AB"},), metadata={"category": "construction-hard"},
        difficulty=DifficultyProfile(level="high_school", category="construction-hard", visual_complexity=1, construction_depth=1, theorem_depth=2, symbolic_growth=1),
    )


def _parallel_task() -> GeoClawTask:
    return GeoClawTask(
        id="construct_parallel", problem_text="Through C construct a line parallel to AB, verify the relation, then answer AB.", trace=tuple(_triangle_base("AB", "length")), answer="4",
        required_tool_sets=(("construct", "verify"),), reference_cost=2.5, max_steps=8, difficulty=DifficultyProfile(level="high_school", category="construction-hard", construction_depth=1, theorem_depth=2), metadata={"category": "construction-hard"},
    )


def _intersection_task() -> GeoClawTask:
    trace = [
        {"op": "add_point", "id": "A", "coord": ["0", "0"]}, {"op": "add_point", "id": "B", "coord": ["4", "4"]},
        {"op": "add_point", "id": "C", "coord": ["0", "4"]}, {"op": "add_point", "id": "D", "coord": ["4", "0"]},
        {"op": "add_line", "id": "l1", "points": ["A", "B"]}, {"op": "add_line", "id": "l2", "points": ["C", "D"]},
        {"op": "set_target", "type": "distance", "object": "A|B"},
    ]
    return GeoClawTask(id="construct_intersection", problem_text="Construct intersection X of l1 and l2, then find distance AB.", trace=tuple(trace), answer="4*sqrt(2)", required_tool_sets=(("construct", "measure"),), reference_cost=2.5, max_steps=8, difficulty=DifficultyProfile(level="high_school", category="construction-hard", construction_depth=1, symbolic_growth=2), metadata={"category": "construction-hard"})


def _circle_task() -> GeoClawTask:
    trace = [
        {"op": "add_point", "id": "O", "coord": ["0", "0"]}, {"op": "add_point", "id": "A", "coord": ["3", "0"]},
        {"op": "add_circle", "id": "c", "center": "O", "radius": "3"}, {"op": "set_target", "type": "area", "object": "c"},
    ]
    return GeoClawTask(id="circle_area", problem_text="Measure the radius and calculate the area of circle c.", trace=tuple(trace), answer="9*pi", required_tool_sets=(("measure", "calculate"),), reference_cost=2.0, max_steps=8, difficulty=DifficultyProfile(level="high_school", category="numeric-hard", symbolic_growth=1), metadata={"category": "numeric-hard"})


def _area_task() -> GeoClawTask:
    return GeoClawTask(id="triangle_area", problem_text="Find the area of triangle ABC.", trace=tuple(_triangle_base("ABC", "area")), answer="6", required_tool_sets=(("measure", "calculate"), ("verify",)), reference_cost=2.0, max_steps=8, difficulty=DifficultyProfile(level="high_school", category="numeric-hard", theorem_depth=1, symbolic_growth=1), metadata={"category": "numeric-hard"})


def _verify_task() -> GeoClawTask:
    return GeoClawTask(id="verify_right_triangle", problem_text="Verify that ABC is right and give the area.", trace=tuple(_triangle_base("ABC", "area")), answer="6", required_tool_sets=(("verify",),), reference_cost=1.0, max_steps=6, difficulty=DifficultyProfile(level="high_school", category="mixed-hard", theorem_depth=2), metadata={"category": "mixed-hard"})


def _mixed_task() -> GeoClawTask:
    return GeoClawTask(id="mixed_zoom_construct", problem_text="Inspect the diagram, construct midpoint M of BC, verify it, and give BC.", trace=tuple(_triangle_base("BC", "length")), answer="5", required_tool_sets=(("zoom", "construct", "verify"),), reference_cost=3.5, max_steps=10, visual_regions=(VisualRegion(id="bc_region", region=(0.20, 0.05, 0.95, 0.90), purpose="dense_segment", objects=("B", "C", "BC"), required=True),), difficulty=DifficultyProfile(level="olympiad", category="mixed-hard", visual_complexity=3, construction_depth=1, theorem_depth=2, symbolic_growth=1), metadata={"category": "mixed-hard"})
