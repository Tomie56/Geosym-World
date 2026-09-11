from geosym_world.parser.v1 import compile_parse_graph
from geosym_world.schema import (
    EvidencePrecision,
    ParseEvidence,
    ParseEvidenceKind,
    ParseGraph,
    ParseObject,
    ParseObjectKind,
    ParseOrigin,
    ParseReference,
    ParseRelation,
)


def test_parse_graph_compiles_without_losing_candidate_provenance():
    graph = ParseGraph(
        id="parse_1",
        source_id="problem_1",
        evidence=(ParseEvidence("pixel_1", "problem_1", ParseEvidenceKind.PIXEL_POINT,
                                EvidencePrecision.LOCALIZED, point=(10.0, 20.0)),),
        objects=(ParseObject("point_a", ParseObjectKind.POINT, ParseOrigin.IMAGE, 0.9,
                             "detector", evidence_refs=("pixel_1",), geometry={"pixel": [10.0, 20.0]}),),
        relations=(ParseRelation("relation_1", "collinear", (ParseReference("point", "point_a"),),
                                 ParseOrigin.IMAGE, 0.8, "detector", evidence_refs=("pixel_1",)),),
    )

    observations, clauses = compile_parse_graph(graph)

    assert observations.observations[0].geometry["point"] == [10.0, 20.0]
    assert clauses.metadata["parse_graph_checksum"] == graph.checksum()
    assert clauses.clauses[0].predicate == "object.point"
    assert clauses.clauses[0].provenance == ("pixel_1",)
    assert clauses.clauses[0].metadata["origin"] == "image"
    assert clauses.clauses[1].arguments == ("point_a",)
