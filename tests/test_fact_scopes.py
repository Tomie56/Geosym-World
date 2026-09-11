from geosym_world.schema import DerivationTracer, FactScope, FactStore, SOURCE_REASONING_SCOPES, validate_derivation_payload


def test_fact_scope_is_serialized_and_part_of_fact_identity():
    facts = FactStore()
    source = facts.add("length", "segment_ab", 4, source="given")
    witness = facts.add("length", "segment_ab", 4, source="derived")

    assert source.scope is FactScope.SOURCE_ASSERTION
    assert witness.scope is FactScope.REALIZATION_FACT
    assert source.id != witness.id
    assert FactStore.from_list(facts.to_list()).to_list() == facts.to_list()


def test_source_reasoning_filter_excludes_witness_and_generation_values():
    facts = FactStore()
    source = facts.add("length", "segment_ab", 4, source="given")
    facts.add("length", "segment_ac", 5, source="derived")
    facts.add("length", "segment_ad", 3, source="generation")

    filtered = facts.for_source_reasoning()

    assert list(filtered) == [source]
    assert facts.find("length", scopes=SOURCE_REASONING_SCOPES) == [source]


def test_source_target_trace_cannot_use_a_realization_target():
    facts = FactStore()
    facts.add("target", "segment_ac", 5, source="derived")

    unrestricted = DerivationTracer().trace_target(facts)
    restricted = DerivationTracer().trace_source_target(facts)

    assert unrestricted["target_fact"] is not None
    assert restricted["target_fact"] is None


def test_derivation_validation_rejects_realization_dependency_in_entailed_fact():
    facts = FactStore()
    source = facts.add("length", "segment_ab", 4, source="given")
    witness = facts.add("length", "segment_ac", 5, source="derived")
    facts.add(
        "target",
        "segment_bc",
        3,
        source="entailed",
        dependencies=(source.id, witness.id),
    )

    derivation = DerivationTracer().trace_target(facts)
    validation = validate_derivation_payload(derivation)

    assert not validation.valid
    assert any(error["code"] == "derivation.scope_leak" for error in validation.errors)
