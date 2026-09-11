from geosym_world.parser import Clause, ClauseGraph, solve_admitted_clause_graph


def test_parser_solver_runs_canonical_fact_closure_on_materialized_state():
    graph = ClauseGraph(
        "clauses_1", "problem_1",
        (
            Clause("point_a", "object.point", ("A",), 1.0,
                   value={"exact_coords": [0, 0]}, status="admitted"),
            Clause("point_b", "object.point", ("B",), 1.0,
                   value={"exact_coords": [3, 4]}, status="admitted"),
            Clause("segment_ab", "object.segment", ("AB",), 1.0,
                   value={"p1": "A", "p2": "B"}, status="admitted"),
        ),
    )

    result = solve_admitted_clause_graph(graph)

    assert result.materialization.unsupported_clause_ids == ()
    assert result.materialization.state.segments["segment_ab"].start == "A"
    assert result.facts.find("length", "segment_ab")[0].value == "5"
