from geosym_world.parser import CapabilityLevel, object_capability, parser_capability_report, relation_capability
from geosym_world.schema import ParseObjectKind


def test_capability_report_distinguishes_contract_from_realization_support():
    assert object_capability(ParseObjectKind.POINT).realization is CapabilityLevel.SUPPORTED
    assert object_capability(ParseObjectKind.ARC).representation is CapabilityLevel.SUPPORTED
    assert object_capability(ParseObjectKind.ARC).realization is CapabilityLevel.PARTIAL
    assert relation_capability("ordered_points").realization is CapabilityLevel.SUPPORTED
    assert relation_capability("tangent").realization is CapabilityLevel.DEFERRED

    report = parser_capability_report()
    assert set(report) == {"objects", "relations"}
    assert report["objects"]["point"]["realization"] == "supported"
    assert report["relations"]["point_on_circle"]["representation"] == "supported"
