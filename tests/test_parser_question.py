from geosym_world.parser import ParserInput, ReconstructionMode, TypedDiagramParserV0
from geosym_world.parser.reconstruction import ParseGraphReconstructor
from geosym_world.parser.question import infer_quantity_assertions, infer_relation_assertions, infer_target_type
from geosym_world.schema import AdmissionStatus, EvidencePrecision, ParseOrigin
from geosym_world.verifier import ParseGraphAdmission


class ParallelSegmentsDetector:
    def detect(self, image_path: str):
        return {
            "image_size": {"width": 200, "height": 200},
            "lines": [
                {"id": "vl_ab", "p1": [20, 50], "p2": [180, 50], "angle_deg": 0.0, "source_ids": ["raw_ab"]},
                {"id": "vl_cd", "p1": [20, 150], "p2": [180, 150], "angle_deg": 0.0, "source_ids": ["raw_cd"]},
            ],
            "circles": [],
            "text_regions": [],
            "topology": {
                "endpoints": [
                    {"id": "vl_ab_p1", "line_id": "vl_ab", "xy": [20, 50]},
                    {"id": "vl_ab_p2", "line_id": "vl_ab", "xy": [180, 50]},
                    {"id": "vl_cd_p1", "line_id": "vl_cd", "xy": [20, 150]},
                    {"id": "vl_cd_p2", "line_id": "vl_cd", "xy": [180, 150]},
                ],
                "line_intersections": [],
                "circle_line_intersections": [],
                "circle_circle_intersections": [],
            },
        }


class EndpointRecognizer:
    def __init__(self, names=("A", "B", "C", "D")):
        positions = {"A": [5, 44, 10, 12], "B": [185, 44, 10, 12], "C": [5, 144, 10, 12], "D": [185, 144, 10, 12]}
        self.labels = tuple(
            {
                "id": f"vt_{name.lower()}",
                "bbox": positions[name],
                "text": name,
                "confidence": 0.97,
                "status": "resolved",
            }
            for name in names
        )

    def recognize(self, image_path, text_regions):
        return list(self.labels)


class MidpointDetector:
    def detect(self, image_path: str):
        return {
            "image_size": {"width": 200, "height": 200},
            "lines": [
                {"id": "vl_ab", "p1": [20, 100], "p2": [180, 100], "angle_deg": 0.0, "source_ids": ["raw_ab"]},
                {"id": "vl_cross", "p1": [100, 60], "p2": [100, 140], "angle_deg": 90.0, "source_ids": ["raw_cross"]},
            ],
            "circles": [],
            "text_regions": [],
            "topology": {
                "endpoints": [
                    {"id": "vl_ab_p1", "line_id": "vl_ab", "xy": [20, 100]},
                    {"id": "vl_ab_p2", "line_id": "vl_ab", "xy": [180, 100]},
                    {"id": "vl_cross_p1", "line_id": "vl_cross", "xy": [100, 60]},
                    {"id": "vl_cross_p2", "line_id": "vl_cross", "xy": [100, 140]},
                ],
                "line_intersections": [{"id": "vx_d", "line_ids": ["vl_ab", "vl_cross"], "xy": [100, 100]}],
                "circle_line_intersections": [],
                "circle_circle_intersections": [],
            },
        }


class MidpointRecognizer:
    def recognize(self, image_path, text_regions):
        return [
            {"id": "vt_a", "bbox": [5, 94, 10, 12], "text": "A", "confidence": 0.97, "status": "resolved"},
            {"id": "vt_b", "bbox": [185, 94, 10, 12], "text": "B", "confidence": 0.97, "status": "resolved"},
            {"id": "vt_d", "bbox": [94, 82, 12, 12], "text": "D", "confidence": 0.97, "status": "resolved"},
        ]


def test_explicit_binary_segment_relations_are_extracted_without_becoming_quantities():
    question = "AB is perpendicular to BC. DE ∥ FG. HI = JK. Find the length of AB."

    relations = infer_relation_assertions(question)

    assert [relation.predicate for relation in relations] == ["perpendicular", "parallel", "equal_length"]
    assert [reference.point_names for reference in relations[0].references] == [("A", "B"), ("B", "C")]
    assert infer_quantity_assertions(question) == ()


def test_target_type_distinguishes_shaded_area_from_generic_area():
    assert infer_target_type("Find the area of the shaded circular segment.").target_type == "shadow_area"
    assert infer_target_type("求阴影部分的面积。").target_type == "shadow_area"
    assert infer_target_type("Find the area of triangle ABC.").target_type == "area"
    assert infer_target_type("Find the ratio of the shaded area to the sector area.").target_type == "shadow_ratio"
    assert infer_target_type("Find the ratio AB:CD.").target_type == "ratio"


def test_midpoint_and_collinear_relations_support_english_and_chinese():
    question = "D is the midpoint of AB. 点 E、F、G 三点共线。"

    relations = infer_relation_assertions(question)

    assert [relation.predicate for relation in relations] == ["midpoint", "collinear"]
    assert [(reference.role, reference.point_names) for reference in relations[0].references] == [
        ("point", ("D",)),
        ("of_0", ("A",)),
        ("of_1", ("B",)),
    ]
    assert [reference.point_names for reference in relations[1].references] == [("E",), ("F",), ("G",)]


def test_chinese_binary_segment_relations_are_explicit_not_incidental_conjunctions():
    relations = infer_relation_assertions("AB垂直于CD。EF与GH长度相等。IJ平行于KL。MN与OP。")

    assert [relation.predicate for relation in relations] == ["perpendicular", "equal_length", "parallel"]


def test_invalid_self_relations_are_ignored():
    assert infer_relation_assertions("AA is parallel to BC. A is the midpoint of AB.") == ()


def test_typed_parser_grounds_and_admits_an_explicit_question_relation(tmp_path):
    result = _parse_parallel_question(tmp_path, EndpointRecognizer())

    relation = next(item for item in result.graph.relations if item.score_kind == "explicit_question_relation")
    assert relation.predicate == "parallel"
    assert relation.origin is ParseOrigin.TEXT
    assert [argument.ref for argument in relation.arguments] == ["segment:vl_ab", "segment:vl_cd"]
    assert len(relation.dependencies) == 4
    assert len([item for item in result.graph.relations if item.predicate == "parallel"]) == 1
    assert result.graph.metadata["suppressed_visual_relation_ids"] == ["relation:parallel:0"]
    assert result.graph.metadata["question_relation_issues"] == []
    assert _stage(result, "question_relations") == {
        "name": "question_relations",
        "status": "complete",
        "output_count": 1,
    }

    admission = ParseGraphAdmission().admit(result.graph)
    decision = next(item for item in admission.decisions if item.candidate_id == relation.id)
    assert decision.decision is AdmissionStatus.ADMITTED
    assert decision.reason == "evidence_policy_satisfied"


def test_typed_parser_binds_proof_target_to_its_question_relation(tmp_path):
    result = _parse_parallel_question(
        tmp_path,
        EndpointRecognizer(),
        question="Prove that AB is parallel to CD.",
    )

    relation = next(item for item in result.graph.relations if item.score_kind == "explicit_question_relation")
    target = result.graph.targets[0]
    assert target.kind == "parallel"
    assert target.mode == "proof"
    assert target.arguments == relation.arguments
    assert target.dependencies == (relation.id, *relation.dependencies)
    assert target.metadata["proof_relation_id"] == relation.id

    admission = ParseGraphAdmission().admit(result.graph)
    assert next(item for item in admission.decisions if item.candidate_id == target.id).decision is AdmissionStatus.ADMITTED
    reconstruction = ParseGraphReconstructor().reconstruct(
        admission,
        mode=ReconstructionMode.PIXEL_WITNESS,
    )
    assert reconstruction.state is not None
    assert reconstruction.state.target.mode == "proof"
    assert reconstruction.state.target.object == "segment:vl_ab|segment:vl_cd"


def test_unresolved_question_relation_keeps_localized_text_without_a_partial_candidate(tmp_path):
    result = _parse_parallel_question(tmp_path, EndpointRecognizer(("A", "B", "C")))

    assert not any(item.score_kind == "explicit_question_relation" for item in result.graph.relations)
    issue = result.graph.metadata["question_relation_issues"][0]
    assert issue["status"] == "reference_unresolved"
    assert issue["references"] == [
        {"role": "object_1", "reference_kind": "segment", "point_names": ["C", "D"]}
    ]
    evidence = next(item for item in result.graph.evidence if item.id == "evidence:question_relation:0")
    assert evidence.precision is EvidencePrecision.LOCALIZED
    assert evidence.raw_content == "AB is parallel to CD"
    assert _stage(result, "question_relations")["status"] == "partial"


def test_typed_parser_grounds_midpoint_roles_to_named_points(tmp_path):
    image_path = tmp_path / "midpoint.png"
    image_path.write_bytes(b"fixture")
    parser = TypedDiagramParserV0(
        detector=MidpointDetector(),
        label_recognizer=MidpointRecognizer(),
        infer_planar_faces=False,
    )
    result = parser.parse(
        ParserInput(id="question_midpoint", image=str(image_path), question="D is the midpoint of AB.")
    )

    relation = next(item for item in result.graph.relations if item.score_kind == "explicit_question_relation")
    point_names = result.graph.point_names()
    assert relation.predicate == "midpoint"
    assert [argument.role for argument in relation.arguments] == ["point", "of_0", "of_1"]
    assert [point_names[argument.ref] for argument in relation.arguments] == ["D", "A", "B"]
    assert len(relation.dependencies) == 3
    admission = ParseGraphAdmission().admit(result.graph)
    assert next(item for item in admission.decisions if item.candidate_id == relation.id).decision is AdmissionStatus.ADMITTED


def _parse_parallel_question(tmp_path, recognizer, question="AB is parallel to CD."):
    image_path = tmp_path / "opaque.png"
    image_path.write_bytes(b"fixture")
    parser = TypedDiagramParserV0(
        detector=ParallelSegmentsDetector(),
        label_recognizer=recognizer,
        infer_planar_faces=False,
    )
    return parser.parse(ParserInput(id="question_relation", image=str(image_path), question=question))


def _stage(result, name):
    return next(stage for stage in result.stages if stage["name"] == name)
