from geosym_world.parser import TextClaimExtractor


def test_text_claim_extractor_recovers_explicit_constraints():
    claims = TextClaimExtractor().extract(
        "$A B C D$ is a rectangle, AB=16, BC=12, and AC perpendicular to CE."
    )

    assert {(item["type"], tuple(item["objects"])) for item in claims} == {
        ("rectangle", ("ABCD",)),
        ("length", ("AB",)),
        ("length", ("BC",)),
        ("perpendicular", ("AC", "CE")),
    }
