"""
Question generation for GeoSymSolid scenes.
"""

from __future__ import annotations

from typing import Any, Dict, List


class GeoSymSolidQAGenerator:
    """Create solver-backed QA examples from a GeoSymSolid scene."""

    def _qa(self, scene_id: str, suffix: str, task_type: str, question: str,
            answer: Any, target: str, requires: List[str]) -> Dict[str, Any]:
        return {
            "id": f"{scene_id}_q_{suffix}",
            "scene_id": scene_id,
            "task_type": task_type,
            "question": question,
            "answer": answer,
            "target_quantity": target,
            "requires": requires,
        }

    def generate(self, scene: Dict[str, Any], render_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        scene_id = scene["scene_id"]
        template = scene["template_id"]
        q = scene["quantities"]
        qa = []

        if "volume" in q:
            qa.append(self._qa(
                scene_id, "volume", "measurement.volume",
                f"For the {template} solid, what is the exact volume?",
                q["volume"], "volume", ["solid_decomposition", "volume_formula"],
            ))
        if "surface_area" in q:
            qa.append(self._qa(
                scene_id, "surface_area", "measurement.surface_area",
                f"For the {template} solid, what is the total surface area?",
                q["surface_area"], "surface_area", ["face_accounting"],
            ))

        if template == "cuboid_section":
            qa.extend([
                self._qa(scene_id, "space_diag", "measurement.length",
                         "What is the exact length of the space diagonal AG?",
                         q["AG"], "AG", ["3d_distance"]),
                self._qa(scene_id, "section_area", "section.area",
                         "The highlighted diagonal section is A-C-G-E. What is its exact area?",
                         q["section_ACGE_area"], "section_ACGE_area", ["section_identification"]),
            ])
        elif template == "l_shape_composite":
            qa.extend([
                self._qa(scene_id, "notch_face", "measurement.area",
                         "What is the area of the highlighted inner notch face?",
                         q["notch_face_area"], "notch_face_area", ["composite_decomposition"]),
                self._qa(scene_id, "top_l_area", "projection.area",
                         "In the top view, what is the area of the L-shaped footprint?",
                         q["top_L_area"], "top_L_area", ["projection", "area_subtraction"]),
            ])
        elif template == "prism_roof":
            qa.extend([
                self._qa(scene_id, "roof_slant", "measurement.length",
                         "What is the exact slant height of either roof face?",
                         q["roof_slant_height"], "roof_slant_height", ["right_triangle"]),
                self._qa(scene_id, "roof_area", "measurement.area",
                         "What is the combined area of the two highlighted roof faces?",
                         q["roof_area"], "roof_area", ["prism_surface"]),
            ])
        elif template == "cylinder_axis_section":
            qa.extend([
                self._qa(scene_id, "axis_section", "section.area",
                         "The highlighted plane is an axial section of the cylinder. What is its area?",
                         q["axis_section_area"], "axis_section_area", ["cylinder_axis_section"]),
                self._qa(scene_id, "diameter", "measurement.length",
                         "What is the diameter of the cylinder base?",
                         q["diameter"], "diameter", ["radius_to_diameter"]),
            ])
        elif template == "cone_axis_section":
            qa.extend([
                self._qa(scene_id, "slant_height", "measurement.length",
                         "What is the exact slant height of the cone?",
                         q["slant_height"], "slant_height", ["right_triangle"]),
                self._qa(scene_id, "axis_section", "section.area",
                         "What is the area of the highlighted axial triangular section?",
                         q["axis_section_area"], "axis_section_area", ["cone_axis_section"]),
            ])
        elif template == "sphere_great_circle":
            qa.extend([
                self._qa(scene_id, "great_circle", "section.area",
                         "What is the area of a great-circle section of the sphere?",
                         q["great_circle_area"], "great_circle_area", ["sphere_section"]),
                self._qa(scene_id, "diameter", "measurement.length",
                         "What is the sphere diameter?",
                         q["diameter"], "diameter", ["radius_to_diameter"]),
            ])
        elif template == "tangent_sphere_cylinder":
            qa.extend([
                self._qa(scene_id, "total_volume", "composite.volume",
                         "A sphere is tangent to the top of a cylinder along the common axis point. What is the total volume?",
                         q["total_volume"], "total_volume", ["composite_addition"]),
                self._qa(scene_id, "axis_total", "measurement.length",
                         "What is the total length from the cylinder bottom center to the sphere top along the shared axis?",
                         q["axis_total_length"], "axis_total_length", ["axis_alignment", "translation"]),
            ])

        highlighted_faces = scene["solid"].get("highlighted_faces") or []
        first_highlight = highlighted_faces[0] if highlighted_faces else None
        if first_highlight:
            qa.append(self._qa(
                scene_id, "view_visibility", "projection.visibility",
                f"In the perspective view, which face is highlighted?",
                {"face_id": first_highlight, "view": "perspective"},
                first_highlight,
                ["visual_grounding", "highlight_detection"],
            ))

        # Include one machine-checkable visibility statistic from metadata.
        perspective_edges = render_result["metadata"]["perspective"]["edge_visibility"]
        hidden_count = sum(1 for value in perspective_edges.values() if value == "hidden")
        qa.append(self._qa(
            scene_id, "hidden_count", "rendering.hidden_edges",
            "How many hidden dashed edges are recorded in the perspective render metadata?",
            {"value": hidden_count},
            "perspective_hidden_edges",
            ["render_metadata"],
        ))
        return qa
