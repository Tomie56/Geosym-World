"""
Symbolic/CAD geometry kernel for GeoSymSolid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sympy as sp

from .cad import CadValidation, CadValidator
from .symbolic import attach_cad_validation, expr_payload, quantity_payload


Point = Tuple[sp.Expr, sp.Expr, sp.Expr]


def _expr_payload(expr: sp.Expr) -> Dict[str, Any]:
    return expr_payload(expr)


def _quantity(expr: sp.Expr, unit: str = "", role: str = "answer") -> Dict[str, Any]:
    return quantity_payload(expr, unit=unit, role=role)


def _param_payload(**kwargs) -> Dict[str, Any]:
    return {key: _expr_payload(sp.sympify(value)) for key, value in kwargs.items()}


def _poly_area_3d(vertices: List[Point]) -> sp.Expr:
    if len(vertices) < 3:
        return sp.Integer(0)
    p0 = vertices[0]
    area = sp.Integer(0)
    for idx in range(1, len(vertices) - 1):
        p1 = vertices[idx]
        p2 = vertices[idx + 1]
        v1 = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        v2 = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
        cross = (
            v1[1] * v2[2] - v1[2] * v2[1],
            v1[2] * v2[0] - v1[0] * v2[2],
            v1[0] * v2[1] - v1[1] * v2[0],
        )
        area += sp.sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2) / 2
    return sp.simplify(area)


class GeoSymSolidKernel:
    """Generate task-driven, solver-owned solid geometry states."""

    def __init__(self, output_dir: str = "output/geosymsolid"):
        self.output_dir = Path(output_dir)
        self.cad_dir = self.output_dir / "cad"
        self.cad_dir.mkdir(parents=True, exist_ok=True)
        self.cad_validator = CadValidator(str(self.cad_dir))

    def cadquery_available(self) -> bool:
        return self.cad_validator.available()

    def _cad_export(self, scene_id: str, solid_builder) -> CadValidation:
        return self.cad_validator.build_export_validate(scene_id, solid_builder)

    def _attach_shape_validation(self, quantities: Dict[str, Dict[str, Any]],
                                 cad_validation: CadValidation) -> None:
        if not cad_validation.available:
            return
        if "volume" in quantities:
            attach_cad_validation(quantities["volume"], cad_validation.volume)
        if "total_volume" in quantities:
            attach_cad_validation(quantities["total_volume"], cad_validation.volume)
        if "surface_area" in quantities:
            attach_cad_validation(quantities["surface_area"], cad_validation.surface_area)

    def _scene(self, scene_id: str, template_id: str, points: Dict[str, Point],
               edges: List[Tuple[str, str, str]], faces: Dict[str, List[str]],
               highlighted_faces: List[str], quantities: Dict[str, Dict[str, Any]],
               task_targets: List[str], operation_trace: List[Dict[str, Any]],
               formal_state: List[str], cad_validation: CadValidation,
               analytic_surfaces: Optional[List[Dict[str, Any]]] = None,
               parameters: Optional[Dict[str, Any]] = None,
               base_template: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._attach_shape_validation(quantities, cad_validation)
        return {
            "scene_id": scene_id,
            "engine": "GeoSymSolid",
            "template_id": template_id,
            "base_template": base_template or {"id": template_id, "family": template_id},
            "parameters": parameters or {},
            "solid": {
                "type": template_id,
                "points": {
                    name: {
                        "x": _expr_payload(coord[0]),
                        "y": _expr_payload(coord[1]),
                        "z": _expr_payload(coord[2]),
                    }
                    for name, coord in points.items()
                },
                "edges": [{"id": eid, "start": s, "end": e} for eid, s, e in edges],
                "faces": [
                    {"id": fid, "vertices": verts, "highlight": fid in highlighted_faces}
                    for fid, verts in faces.items()
                ],
                "highlighted_faces": highlighted_faces,
                "analytic_surfaces": analytic_surfaces or [],
            },
            "quantities": quantities,
            "task_targets": task_targets,
            "operation_trace": operation_trace,
            "evolution_trace": operation_trace,
            "cad_validation": cad_validation.to_dict(),
            "formal_state": formal_state,
        }

    def create_cuboid_section_scene(self, scene_id: str = "geosymsolid_0000",
                                    a: sp.Expr = sp.Integer(6),
                                    b: sp.Expr = sp.Integer(4),
                                    c: sp.Expr = sp.Integer(3)) -> Dict[str, Any]:
        a, b, c = sp.sympify(a), sp.sympify(b), sp.sympify(c)
        points = {
            "A": (0, 0, 0), "B": (a, 0, 0), "C": (a, b, 0), "D": (0, b, 0),
            "E": (0, 0, c), "F": (a, 0, c), "G": (a, b, c), "H": (0, b, c),
        }
        points = {k: tuple(sp.sympify(v) for v in vals) for k, vals in points.items()}
        edges = [
            ("AB", "A", "B"), ("BC", "B", "C"), ("CD", "C", "D"), ("DA", "D", "A"),
            ("EF", "E", "F"), ("FG", "F", "G"), ("GH", "G", "H"), ("HE", "H", "E"),
            ("AE", "A", "E"), ("BF", "B", "F"), ("CG", "C", "G"), ("DH", "D", "H"),
            ("AC", "A", "C"), ("EG", "E", "G"), ("AG", "A", "G"),
        ]
        faces = {
            "bottom": ["A", "B", "C", "D"],
            "top": ["E", "F", "G", "H"],
            "front": ["A", "B", "F", "E"],
            "right": ["B", "C", "G", "F"],
            "back": ["C", "D", "H", "G"],
            "left": ["D", "A", "E", "H"],
            "section_ACGE": ["A", "C", "G", "E"],
        }
        base_diag = sp.sqrt(a ** 2 + b ** 2)
        quantities = {
            "volume": _quantity(a * b * c, unit="volume"),
            "surface_area": _quantity(2 * (a * b + a * c + b * c), unit="area"),
            "AC": _quantity(base_diag, unit="length"),
            "AG": _quantity(sp.sqrt(a ** 2 + b ** 2 + c ** 2), unit="length"),
            "section_ACGE_area": _quantity(base_diag * c, unit="area"),
            "angle_AG_to_base": {
                **_quantity(sp.atan(c / base_diag), unit="radian"),
                "degrees": float((sp.atan(c / base_diag) * 180 / sp.pi).evalf()),
            },
        }
        cad = self._cad_export(
            scene_id,
            lambda cq: cq.Workplane("XY").box(float(a), float(b), float(c), centered=(False, False, False)),
        )
        return self._scene(
            scene_id, "cuboid_section", points, edges, faces, ["section_ACGE"], quantities,
            ["volume", "AG", "section_ACGE_area", "angle_AG_to_base"],
            [
                {"op": "primitive", "type": "cuboid", "params": {"a": str(a), "b": str(b), "c": str(c)}},
                {"op": "section", "face": "section_ACGE", "through": ["A", "C", "G", "E"]},
            ],
            [
                f"cuboid(A,B,C,D,E,F,G,H,a={a},b={b},c={c})",
                "section(section_ACGE, cuboid, vertices=[A,C,G,E])",
                "perpendicular(edge(AE), face(ABCD))",
            ],
            cad,
            parameters=_param_payload(a=a, b=b, c=c),
            base_template={"id": "cuboid", "family": "polyhedron", "allowed_evolutions": ["section"]},
        )

    def _regular_ngon_points(self, prefix: str, center: Tuple[sp.Expr, sp.Expr, sp.Expr],
                             radius: sp.Expr, z_offset: sp.Expr = 0,
                             n: int = 16) -> Dict[str, Point]:
        cx, cy, cz = center
        points = {}
        for i in range(n):
            angle = 2 * sp.pi * i / n
            points[f"{prefix}{i}"] = (
                sp.simplify(cx + radius * sp.cos(angle)),
                sp.simplify(cy + radius * sp.sin(angle)),
                sp.simplify(cz + z_offset),
            )
        return points

    def create_cylinder_scene(self, scene_id: str, r: sp.Expr = 3, h: sp.Expr = 6,
                              n: int = 16, dx: sp.Expr = 0, dy: sp.Expr = 0) -> Dict[str, Any]:
        r, h, dx, dy = [sp.sympify(v) for v in (r, h, dx, dy)]
        bottom = self._regular_ngon_points("B", (dx, dy, 0), r, 0, n)
        top = self._regular_ngon_points("T", (dx, dy, 0), r, h, n)
        points = {**bottom, **top, "O": (dx, dy, 0), "P": (dx, dy, h)}
        edges = []
        for i in range(n):
            edges.append((f"B{i}B{(i+1)%n}", f"B{i}", f"B{(i+1)%n}"))
            edges.append((f"T{i}T{(i+1)%n}", f"T{i}", f"T{(i+1)%n}"))
            edges.append((f"B{i}T{i}", f"B{i}", f"T{i}"))
        edges.extend([("OP", "O", "P"), ("B0T0_axis_section", "B0", "T0"), (f"B{n//2}T{n//2}_axis_section", f"B{n//2}", f"T{n//2}")])
        faces = {
            "bottom": [f"B{i}" for i in range(n)],
            "top": [f"T{i}" for i in range(n)],
            "axis_section": ["B0", f"B{n//2}", f"T{n//2}", "T0"],
        }
        for i in range(n):
            faces[f"side_{i}"] = [f"B{i}", f"B{(i+1)%n}", f"T{(i+1)%n}", f"T{i}"]
        quantities = {
            "volume": _quantity(sp.pi * r ** 2 * h, unit="volume"),
            "surface_area": _quantity(2 * sp.pi * r * (r + h), unit="area"),
            "axis_section_area": _quantity(2 * r * h, unit="area"),
            "diameter": _quantity(2 * r, unit="length"),
            "height": _quantity(h, unit="length"),
        }
        cad = self._cad_export(
            scene_id,
            lambda cq: cq.Workplane("XY").circle(float(r)).extrude(float(h)).translate((float(dx), float(dy), 0)),
        )
        return self._scene(
            scene_id, "cylinder_axis_section", points, edges, faces, ["axis_section"], quantities,
            ["volume", "surface_area", "axis_section_area"],
            [
                {"op": "primitive", "type": "cylinder", "params": {"r": str(r), "h": str(h)}},
                {"op": "translate", "vector": [str(dx), str(dy), "0"]},
                {"op": "section", "type": "axis_section", "through": ["B0", f"B{n//2}", f"T{n//2}", "T0"]},
            ],
            [f"cylinder(O,P,r={r},h={h})", "axis_section(rectangle, width=2r, height=h)"],
            cad,
            parameters=_param_payload(r=r, h=h, dx=dx, dy=dy),
            base_template={"id": "cylinder", "family": "round_solid", "allowed_evolutions": ["translate", "axis_section"]},
        )

    def create_cone_scene(self, scene_id: str, r: sp.Expr = 3, h: sp.Expr = 6,
                          n: int = 16, dx: sp.Expr = 0, dy: sp.Expr = 0) -> Dict[str, Any]:
        r, h, dx, dy = [sp.sympify(v) for v in (r, h, dx, dy)]
        base = self._regular_ngon_points("B", (dx, dy, 0), r, 0, n)
        points = {**base, "O": (dx, dy, 0), "V": (dx, dy, h)}
        edges = []
        for i in range(n):
            edges.append((f"B{i}B{(i+1)%n}", f"B{i}", f"B{(i+1)%n}"))
            edges.append((f"B{i}V", f"B{i}", "V"))
        edges.extend([("OV", "O", "V"), ("B0V_axis", "B0", "V"), (f"B{n//2}V_axis", f"B{n//2}", "V")])
        faces = {"base": [f"B{i}" for i in range(n)], "axis_section": ["B0", f"B{n//2}", "V"]}
        for i in range(n):
            faces[f"side_{i}"] = [f"B{i}", f"B{(i+1)%n}", "V"]
        slant = sp.sqrt(r ** 2 + h ** 2)
        quantities = {
            "volume": _quantity(sp.pi * r ** 2 * h / 3, unit="volume"),
            "surface_area": _quantity(sp.pi * r * (r + slant), unit="area"),
            "slant_height": _quantity(slant, unit="length"),
            "axis_section_area": _quantity(r * h, unit="area"),
        }
        cad = self._cad_export(
            scene_id,
            lambda cq: cq.Solid.makeCone(float(r), 0, float(h)).translate((float(dx), float(dy), 0)),
        )
        return self._scene(
            scene_id, "cone_axis_section", points, edges, faces, ["axis_section"], quantities,
            ["volume", "slant_height", "axis_section_area"],
            [
                {"op": "primitive", "type": "cone", "params": {"r": str(r), "h": str(h)}},
                {"op": "translate", "vector": [str(dx), str(dy), "0"]},
                {"op": "section", "type": "axis_section", "through": ["B0", f"B{n//2}", "V"]},
            ],
            [f"cone(O,V,r={r},h={h})", "axis_section(isosceles_triangle, base=2r, height=h)"],
            cad,
            parameters=_param_payload(r=r, h=h, dx=dx, dy=dy),
            base_template={"id": "cone", "family": "round_solid", "allowed_evolutions": ["translate", "axis_section"]},
        )

    def create_sphere_scene(self, scene_id: str, r: sp.Expr = 4,
                            dx: sp.Expr = 0, dy: sp.Expr = 0, dz: sp.Expr = 0) -> Dict[str, Any]:
        r, dx, dy, dz = [sp.sympify(v) for v in (r, dx, dy, dz)]
        points = {
            "O": (dx, dy, dz),
            "N": (dx, dy, dz + r),
            "S": (dx, dy, dz - r),
            "E": (dx + r, dy, dz),
            "W": (dx - r, dy, dz),
        }
        edges = [("NS", "N", "S"), ("EW", "E", "W")]
        faces: Dict[str, List[str]] = {}
        quantities = {
            "volume": _quantity(sp.Rational(4, 3) * sp.pi * r ** 3, unit="volume"),
            "surface_area": _quantity(4 * sp.pi * r ** 2, unit="area"),
            "great_circle_area": _quantity(sp.pi * r ** 2, unit="area"),
            "diameter": _quantity(2 * r, unit="length"),
        }
        cad = self._cad_export(
            scene_id,
            lambda cq: cq.Workplane("XY").sphere(float(r)).translate((float(dx), float(dy), float(dz))),
        )
        return self._scene(
            scene_id, "sphere_great_circle", points, edges, faces, [], quantities,
            ["volume", "surface_area", "great_circle_area"],
            [
                {"op": "primitive", "type": "sphere", "params": {"r": str(r)}},
                {"op": "translate", "vector": [str(dx), str(dy), str(dz)]},
                {"op": "section", "type": "great_circle"},
            ],
            [f"sphere(O,r={r})", "great_circle(section, radius=r)"],
            cad,
            analytic_surfaces=[{"id": "sphere_surface", "type": "sphere", "center": "O", "radius": _expr_payload(r)}],
            parameters=_param_payload(r=r, dx=dx, dy=dy, dz=dz),
            base_template={"id": "sphere", "family": "round_solid", "allowed_evolutions": ["translate", "great_circle"]},
        )

    def create_tangent_sphere_cylinder_scene(self, scene_id: str, r: sp.Expr = 3,
                                             h: sp.Expr = 7) -> Dict[str, Any]:
        r, h = sp.sympify(r), sp.sympify(h)
        n = 16
        bottom = self._regular_ngon_points("B", (0, 0, 0), r, 0, n)
        top = self._regular_ngon_points("T", (0, 0, 0), r, h, n)
        points = {**bottom, **top, "O": (0, 0, 0), "P": (0, 0, h)}
        # Sphere tangent to cylinder top at shared axis endpoint P.
        points.update({
            "SO": (0, 0, h + r),
            "SN": (0, 0, h + 2 * r),
            "SS": (0, 0, h),
            "SE": (r, 0, h + r),
            "SW": (-r, 0, h + r),
        })
        edges = []
        for i in range(n):
            edges.append((f"B{i}B{(i+1)%n}", f"B{i}", f"B{(i+1)%n}"))
            edges.append((f"T{i}T{(i+1)%n}", f"T{i}", f"T{(i+1)%n}"))
            edges.append((f"B{i}T{i}", f"B{i}", f"T{i}"))
        edges.append(("OP", "O", "P"))
        edges.extend([("sphere_axis", "SS", "SN"), ("sphere_diameter", "SW", "SE"), ("axis_overlap", "O", "SO")])
        faces = {"bottom": [f"B{i}" for i in range(n)], "top": [f"T{i}" for i in range(n)]}
        for i in range(n):
            faces[f"side_{i}"] = [f"B{i}", f"B{(i+1)%n}", f"T{(i+1)%n}", f"T{i}"]
        faces["shared_tangent_plane_section"] = ["T0", "T4", "T8", "T12"] if "T12" in points else ["T0", "T4", "T8"]
        quantities = {
            "total_volume": _quantity(sp.pi * r ** 2 * h + sp.Rational(4, 3) * sp.pi * r ** 3, unit="volume"),
            "cylinder_volume": _quantity(sp.pi * r ** 2 * h, unit="volume"),
            "sphere_volume": _quantity(sp.Rational(4, 3) * sp.pi * r ** 3, unit="volume"),
            "shared_radius": _quantity(r, unit="length"),
            "axis_total_length": _quantity(h + 2 * r, unit="length"),
        }
        cad = self._cad_export(
            scene_id,
            lambda cq: (
                cq.Workplane("XY").circle(float(r)).extrude(float(h))
                .union(cq.Workplane("XY").sphere(float(r)).translate((0, 0, float(h + r))))
            ),
        )
        return self._scene(
            scene_id, "tangent_sphere_cylinder", points, edges, faces, ["shared_tangent_plane_section"], quantities,
            ["total_volume", "shared_radius", "axis_total_length"],
            [
                {"op": "primitive", "type": "cylinder", "params": {"r": str(r), "h": str(h)}},
                {"op": "primitive", "type": "sphere", "params": {"r": str(r)}},
                {"op": "translate", "target": "sphere", "vector": ["0", "0", str(h + r)]},
                {"op": "tangent", "type": "sphere touches cylinder top at axis endpoint"},
            ],
            [f"tangent(sphere(r={r}), cylinder(r={r},h={h}), point=top_axis_center)"],
            cad,
            analytic_surfaces=[{"id": "sphere_surface", "type": "sphere", "center": "SO", "radius": _expr_payload(r)}],
            parameters=_param_payload(r=r, h=h),
            base_template={"id": "sphere_cylinder_pair", "family": "composite", "allowed_evolutions": ["translate", "tangent", "axis_alignment"]},
        )

    def create_l_shape_scene(self, scene_id: str, a: sp.Expr = 8, b: sp.Expr = 6,
                             h: sp.Expr = 3, cut_x: sp.Expr = 3,
                             cut_y: sp.Expr = 2) -> Dict[str, Any]:
        a, b, h, cut_x, cut_y = [sp.sympify(v) for v in (a, b, h, cut_x, cut_y)]
        # Footprint: (0,0)->(a,0)->(a,b-cut_y)->(cut_x,b-cut_y)->(cut_x,b)->(0,b)
        bottom = {
            "A": (0, 0, 0), "B": (a, 0, 0), "C": (a, b - cut_y, 0),
            "D": (cut_x, b - cut_y, 0), "E": (cut_x, b, 0), "F": (0, b, 0),
        }
        top = {chr(ord(k) + 6): (x, y, h) for k, (x, y, _) in bottom.items()}
        points = {**bottom, **top}
        points = {k: tuple(sp.sympify(v) for v in vals) for k, vals in points.items()}
        pairs = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "A")]
        top_pairs = [(chr(ord(s) + 6), chr(ord(e) + 6)) for s, e in pairs]
        edges = []
        for s, e in pairs + top_pairs:
            edges.append((s + e, s, e))
        for s in bottom:
            t = chr(ord(s) + 6)
            edges.append((s + t, s, t))
        faces = {
            "bottom": ["A", "B", "C", "D", "E", "F"],
            "top": ["G", "H", "I", "J", "K", "L"],
            "front": ["A", "B", "H", "G"],
            "right": ["B", "C", "I", "H"],
            "notch_back": ["C", "D", "J", "I"],
            "notch_side": ["D", "E", "K", "J"],
            "back": ["E", "F", "L", "K"],
            "left": ["F", "A", "G", "L"],
            "highlight_notch": ["D", "E", "K", "J"],
        }
        footprint_area = a * b - (a - cut_x) * cut_y
        perimeter = a + (b - cut_y) + (a - cut_x) + cut_y + cut_x + b
        notch_face_area = cut_y * h
        quantities = {
            "volume": _quantity(footprint_area * h, unit="volume"),
            "surface_area": _quantity(2 * footprint_area + perimeter * h, unit="area"),
            "notch_face_area": _quantity(notch_face_area, unit="area"),
            "top_L_area": _quantity(footprint_area, unit="area"),
            "height": _quantity(h, unit="length"),
        }
        cad = self._cad_export(
            scene_id,
            lambda cq: (
                cq.Workplane("XY").box(float(a), float(b), float(h), centered=(False, False, False))
                .cut(cq.Workplane("XY").box(float(a - cut_x), float(cut_y), float(h + 0.2),
                                            centered=(False, False, False))
                     .translate((float(cut_x), float(b - cut_y), -0.1)))
            ),
        )
        return self._scene(
            scene_id, "l_shape_composite", points, edges, faces, ["highlight_notch"], quantities,
            ["volume", "surface_area", "notch_face_area"],
            [
                {"op": "primitive", "type": "cuboid", "params": {"a": str(a), "b": str(b), "h": str(h)}},
                {"op": "boolean_lite", "type": "rectangular_notch_cut",
                 "params": {"cut_x": str(cut_x), "cut_y": str(cut_y)}},
            ],
            [
                f"l_shape_prism(base=rect({a},{b}) minus rect({a-cut_x},{cut_y}), height={h})",
                "cut(rectangular_notch, top_back_right_corner)",
            ],
            cad,
            parameters=_param_payload(a=a, b=b, h=h, cut_x=cut_x, cut_y=cut_y),
            base_template={"id": "cuboid", "family": "composite_polyhedron", "allowed_evolutions": ["notch_cut"]},
        )

    def create_prism_roof_scene(self, scene_id: str, a: sp.Expr = 8, b: sp.Expr = 4,
                                wall_h: sp.Expr = 3, roof_h: sp.Expr = 2) -> Dict[str, Any]:
        a, b, wall_h, roof_h = [sp.sympify(v) for v in (a, b, wall_h, roof_h)]
        mid = a / 2
        z_top = wall_h
        z_ridge = wall_h + roof_h
        points = {
            "A": (0, 0, 0), "B": (a, 0, 0), "C": (a, b, 0), "D": (0, b, 0),
            "E": (0, 0, z_top), "F": (a, 0, z_top), "G": (a, b, z_top), "H": (0, b, z_top),
            "R": (mid, 0, z_ridge), "S": (mid, b, z_ridge),
        }
        points = {k: tuple(sp.sympify(v) for v in vals) for k, vals in points.items()}
        edges = [
            ("AB", "A", "B"), ("BC", "B", "C"), ("CD", "C", "D"), ("DA", "D", "A"),
            ("AE", "A", "E"), ("BF", "B", "F"), ("CG", "C", "G"), ("DH", "D", "H"),
            ("EF", "E", "F"), ("FG", "F", "G"), ("GH", "G", "H"), ("HE", "H", "E"),
            ("ER", "E", "R"), ("FR", "F", "R"), ("HS", "H", "S"), ("GS", "G", "S"),
            ("RS", "R", "S"),
        ]
        faces = {
            "floor": ["A", "B", "C", "D"],
            "front_wall": ["A", "B", "F", "E"],
            "right_wall": ["B", "C", "G", "F"],
            "back_wall": ["C", "D", "H", "G"],
            "left_wall": ["D", "A", "E", "H"],
            "front_gable": ["E", "F", "R"],
            "back_gable": ["H", "G", "S"],
            "roof_left": ["E", "R", "S", "H"],
            "roof_right": ["F", "G", "S", "R"],
        }
        slant = sp.sqrt((a / 2) ** 2 + roof_h ** 2)
        wall_volume = a * b * wall_h
        roof_volume = a * roof_h * b / 2
        roof_area = 2 * slant * b
        quantities = {
            "volume": _quantity(wall_volume + roof_volume, unit="volume"),
            "roof_slant_height": _quantity(slant, unit="length"),
            "roof_area": _quantity(roof_area, unit="area"),
            "front_gable_area": _quantity(a * roof_h / 2, unit="area"),
        }
        cad = self._cad_export(
            scene_id,
            lambda cq: (
                cq.Workplane("XY").box(float(a), float(b), float(wall_h), centered=(False, False, False))
                .union(cq.Workplane("YZ")
                       .polyline([(0, 0), (0, float(a)), (float(roof_h), float(a) / 2)])
                       .close()
                       .extrude(float(b))
                       .rotate((0, 0, 0), (0, 0, 1), -90)
                       .translate((0, 0, float(wall_h))))
            ),
        )
        return self._scene(
            scene_id, "prism_roof", points, edges, faces, ["roof_left", "roof_right"], quantities,
            ["volume", "roof_slant_height", "roof_area"],
            [
                {"op": "primitive", "type": "cuboid_base", "params": {"a": str(a), "b": str(b), "h": str(wall_h)}},
                {"op": "attach", "type": "triangular_prism_roof", "params": {"roof_h": str(roof_h)}},
            ],
            [
                f"house_prism(base=cuboid({a},{b},{wall_h}), roof_height={roof_h})",
                "attach(triangular_prism_roof, face=top)",
            ],
            cad,
            parameters=_param_payload(a=a, b=b, wall_h=wall_h, roof_h=roof_h),
            base_template={"id": "cuboid_base", "family": "composite_polyhedron", "allowed_evolutions": ["attach_prism_roof"]},
        )
