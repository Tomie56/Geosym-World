"""Single source of truth for current GeoSym builder capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass(frozen=True)
class BuilderActionSpec:
    action: str
    required_fields: List[str]
    description: str
    field_schema: Dict[str, str]
    example: Dict[str, object]

    def to_contract(self) -> Dict[str, object]:
        return {
            "action": self.action,
            "required_fields": list(self.required_fields),
            "field_schema": dict(self.field_schema),
            "example": dict(self.example),
            "description": self.description,
        }


@dataclass(frozen=True)
class GeoSymBuilderCapabilities:
    version: str
    supported_actions: Dict[str, BuilderActionSpec]
    supported_constraint_types: Set[str]
    theorem_relation_types: Set[str]
    unsupported_constructs: List[str]

    def prompt_text(self) -> str:
        actions = "\n".join(
            f"- {name}: requires {', '.join(spec.required_fields) or 'no required fields'}; "
            f"fields={spec.field_schema}; example={spec.example}; {spec.description}"
            for name, spec in self.supported_actions.items()
        )
        relation_mappings = "\n".join(
            f"- {item['relation_type']}: status={item['status']}; "
            f"actions={item['builder_actions']}; core={item.get('core_constraint_action')}; "
            f"note={item['note']}"
            for item in self.relation_mapping_contract()
        )
        unsupported = "\n".join(f"- {item}" for item in self.unsupported_constructs)
        return f"""
Current GeoSym 2D builder capability ({self.version}):
Supported builder instructions:
{actions}

Supported canonical constraint types:
{', '.join(sorted(self.supported_constraint_types))}

Convertible semantic relations:
- right_angle: can be converted to set_perpendicular only when the two incident lines are explicit.

Relation-to-builder mapping:
{relation_mappings}

Unsupported theorem-level constructs for canonical build:
{unsupported}

If a problem mainly needs unsupported theorem-level reasoning, extract those
relations, but keep build_instructions empty or limited to visible scaffolding.
Do not force unsupported constructs into point/segment/length instructions.
""".strip()

    def json_schema_hint(self) -> str:
        examples = ",\n  ".join(
            repr(spec.example).replace("'", '"')
            for spec in self.supported_actions.values()
        )
        return f"""
"build_instructions": [
  {examples}
]
""".strip()

    def action_schema_contract(self) -> List[Dict[str, object]]:
        return [spec.to_contract() for spec in self.supported_actions.values()]

    def relation_mapping_contract(self) -> List[Dict[str, object]]:
        """Global contract for semantic relation types and builder actions."""
        return [
            {
                "relation_type": "length",
                "status": "buildable",
                "buildability_mode": "coordinate_constraint",
                "builder_actions": ["add_point", "add_segment", "set_length"],
                "core_constraint_action": "set_length",
                "note": "requires two named endpoints and an exact value",
            },
            {
                "relation_type": "parallel",
                "status": "buildable",
                "buildability_mode": "coordinate_constraint",
                "builder_actions": ["add_point", "add_segment", "set_parallel"],
                "core_constraint_action": "set_parallel",
                "note": "requires two named line ids",
            },
            {
                "relation_type": "perpendicular",
                "status": "buildable",
                "buildability_mode": "coordinate_constraint",
                "builder_actions": ["add_point", "add_segment", "set_perpendicular"],
                "core_constraint_action": "set_perpendicular",
                "note": "requires two named line ids",
            },
            {
                "relation_type": "right_angle",
                "status": "convertible",
                "buildability_mode": "convertible_coordinate_constraint",
                "builder_actions": ["add_point", "add_segment", "set_perpendicular"],
                "core_constraint_action": "set_perpendicular",
                "note": "convert only when the two incident lines are explicit",
            },
            {
                "relation_type": "target",
                "status": "buildable",
                "buildability_mode": "target",
                "builder_actions": ["set_target"],
                "core_constraint_action": "set_target",
                "note": "declares requested quantity; does not add geometry constraints",
            },
            *[
                {
                    "relation_type": relation_type,
                    "status": "theorem_only",
                    "buildability_mode": "theorem_only",
                    "builder_actions": [],
                    "core_constraint_action": None,
                    "note": "keep as relation/formal graph; current canonical builder cannot encode it as coordinate constraints",
                }
                for relation_type in sorted(self.theorem_relation_types)
            ],
        ]


GEOSYM_2D_BUILDER_CAPABILITIES = GeoSymBuilderCapabilities(
    version="2d-canonical-v0",
    supported_actions={
        "add_point": BuilderActionSpec(
            "add_point",
            ["id"],
            "declare a named point without assigning mathematical coordinates",
            {"id": "point label such as A; do not use pixel coordinates", "from_candidate": "optional script candidate id", "visual_ids": "optional overlay ids"},
            {"action": "add_point", "id": "A", "from_candidate": "sc_len_0", "visual_ids": ["vt_1"]},
        ),
        "add_segment": BuilderActionSpec(
            "add_segment",
            ["points"],
            "declare a visible segment between two named points",
            {"points": "two point labels, e.g. ['A','B']", "from_candidate": "optional script candidate id", "visual_ids": "optional line overlay ids"},
            {"action": "add_segment", "points": ["A", "B"], "from_candidate": "sc_len_0", "visual_ids": ["vl_1"]},
        ),
        "set_length": BuilderActionSpec(
            "set_length",
            ["points", "value"],
            "set the exact length of a named segment",
            {"points": "two point labels, e.g. ['A','B']", "value": "exact symbolic/numeric value from text/solution", "from_candidate": "optional script candidate id"},
            {"action": "set_length", "points": ["A", "B"], "value": "4", "from_candidate": "sc_len_0"},
        ),
        "set_parallel": BuilderActionSpec(
            "set_parallel",
            ["lines"],
            "declare two named lines/segments parallel",
            {"lines": "two line ids such as ['AB','CD']", "from_candidate": "optional script candidate id"},
            {"action": "set_parallel", "lines": ["AB", "CD"], "from_candidate": "sc_parallel_0"},
        ),
        "set_perpendicular": BuilderActionSpec(
            "set_perpendicular",
            ["lines"],
            "declare two named lines/segments perpendicular",
            {"lines": "two incident or named line ids such as ['AB','BC']", "from_candidate": "optional script candidate id"},
            {"action": "set_perpendicular", "lines": ["AB", "BC"], "from_candidate": "sc_right_angle_0"},
        ),
        "set_target": BuilderActionSpec(
            "set_target",
            ["target"],
            "declare the quantity requested by the problem",
            {"target": "object with type/object/value_symbol; object should be a named quantity such as AB", "from_candidate": "optional target candidate id"},
            {"action": "set_target", "target": {"type": "length", "object": "AB", "value_symbol": "x", "from_candidate": "sc_target_0"}},
        ),
    },
    supported_constraint_types={"length", "parallel", "perpendicular"},
    theorem_relation_types={
        "angle",
        "arc_measure",
        "area",
        "chord",
        "circle",
        "congruent",
        "incidence",
        "intersecting_chords",
        "perimeter",
        "ratio",
        "right_triangle",
        "similar",
        "triangle",
    },
    unsupported_constructs=[
        "circle/arc/chord/tangent theorem construction",
        "similarity or congruence proof",
        "angle chasing",
        "right-triangle theorem reasoning unless converted to explicit perpendicular and lengths",
        "area decomposition or shaded region solving",
        "perimeter or ratio solving without explicit reducible length constraints",
        "triangle theorem reasoning unless decomposed into explicit supported constraints",
        "general theorem-level proof traces",
        "pixel-to-math coordinate reconstruction",
    ],
)
