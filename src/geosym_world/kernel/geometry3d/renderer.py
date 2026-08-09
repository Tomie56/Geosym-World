"""
Rendering backends for GeoSymSolid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import shutil
import subprocess
import textwrap

import cv2
import numpy as np

from .cad import OpenCascadeHLR


Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]


class TextbookSolidRenderer:
    """
    Generic textbook-style renderer for polyhedral GeoSymSolid scenes.

    Hidden edges are classified from projected face coverage and view-depth
    comparison. This is not a full CAD HLR implementation, but it removes the
    previous cuboid-specific edge-name rules and works across the pilot
    templates.
    """

    def __init__(self, output_dir: str = "output/geosymsolid/renders",
                 canvas_size: Tuple[int, int] = (900, 680)):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.canvas_size = canvas_size
        self.margin = 90
        self.hlr = OpenCascadeHLR()

    def _point_values(self, scene: Dict[str, Any]) -> Dict[str, Point3]:
        out = {}
        for name, payload in scene["solid"]["points"].items():
            out[name] = (
                float(payload["x"]["value"]),
                float(payload["y"]["value"]),
                float(payload["z"]["value"]),
            )
        return out

    def _project(self, point: Point3, view: str) -> Point3:
        x, y, z = point
        if view == "front":
            return (x, z, y)
        if view == "side":
            return (y, z, x)
        if view == "top":
            return (x, y, z)
        if view == "perspective":
            return (x - 0.54 * y, z + 0.34 * y, x + y + z)
        raise ValueError(f"unknown view: {view}")

    def _normalizer(self, projected: Dict[str, Point3]):
        width, height = self.canvas_size
        xs = [p[0] for p in projected.values()]
        ys = [p[1] for p in projected.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        scale = min((width - 2 * self.margin) / span_x, (height - 2 * self.margin) / span_y)
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0

        def normalize(point: Point3) -> Tuple[int, int]:
            return (
                int(width / 2 + (point[0] - cx) * scale),
                int(height / 2 - (point[1] - cy) * scale),
            )

        return normalize

    def _normalizer_for_segments(self, segments: List[Tuple[Point2, Point2]]):
        width, height = self.canvas_size
        pts = [p for seg in segments for p in seg]
        if not pts:
            return lambda p: (int(width / 2), int(height / 2))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        scale = min((width - 2 * self.margin) / span_x, (height - 2 * self.margin) / span_y)
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2

        def normalize(point: Point2) -> Tuple[int, int]:
            return (
                int(width / 2 + (point[0] - cx) * scale),
                int(height / 2 - (point[1] - cy) * scale),
            )

        return normalize

    def _cad_builder_from_scene(self, scene: Dict[str, Any]):
        params = {k: float(v["value"]) for k, v in scene.get("parameters", {}).items()}
        template = scene["template_id"]

        def builder(cq):
            if template == "cuboid_section":
                return cq.Workplane("XY").box(params["a"], params["b"], params["c"], centered=(False, False, False))
            if template == "l_shape_composite":
                a, b, h, cut_x, cut_y = params["a"], params["b"], params["h"], params["cut_x"], params["cut_y"]
                return (
                    cq.Workplane("XY").box(a, b, h, centered=(False, False, False))
                    .cut(cq.Workplane("XY").box(a - cut_x, cut_y, h + 0.2, centered=(False, False, False))
                         .translate((cut_x, b - cut_y, -0.1)))
                )
            if template == "prism_roof":
                a, b, wall_h, roof_h = params["a"], params["b"], params["wall_h"], params["roof_h"]
                return (
                    cq.Workplane("XY").box(a, b, wall_h, centered=(False, False, False))
                    .union(cq.Workplane("YZ")
                           .polyline([(0, 0), (0, a), (roof_h, a / 2)])
                           .close()
                           .extrude(b)
                           .rotate((0, 0, 0), (0, 0, 1), -90)
                           .translate((0, 0, wall_h)))
                )
            if template == "cylinder_axis_section":
                return cq.Workplane("XY").circle(params["r"]).extrude(params["h"]).translate((params.get("dx", 0), params.get("dy", 0), 0))
            if template == "cone_axis_section":
                return cq.Solid.makeCone(params["r"], 0, params["h"]).translate((params.get("dx", 0), params.get("dy", 0), 0))
            if template == "sphere_great_circle":
                return cq.Workplane("XY").sphere(params["r"]).translate((params.get("dx", 0), params.get("dy", 0), params.get("dz", 0)))
            if template == "tangent_sphere_cylinder":
                r, h = params["r"], params["h"]
                return (
                    cq.Workplane("XY").circle(r).extrude(h)
                    .union(cq.Workplane("XY").sphere(r).translate((0, 0, h + r)))
                )
            raise ValueError(f"Unsupported HLR template: {template}")

        return builder

    def _render_hlr_view(self, scene: Dict[str, Any], view: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        if view not in {"front", "side", "top"}:
            return None
        result = self.hlr.extract(self._cad_builder_from_scene(scene), view)
        if not result.available or not result.visible:
            return None
        width, height = self.canvas_size
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        all_segments = result.visible + result.hidden
        normalize = self._normalizer_for_segments(all_segments)
        for start, end in result.hidden:
            self._draw_dashed(canvas, normalize(start), normalize(end), (150, 150, 150), 1)
        for start, end in result.visible:
            cv2.line(canvas, normalize(start), normalize(end), (15, 15, 15), 2, cv2.LINE_AA)

        label = {
            "front": "FRONT OCC HLR",
            "side": "SIDE OCC HLR",
            "top": "TOP OCC HLR",
        }[view]
        cv2.putText(canvas, label, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, scene["template_id"], (24, height - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 80, 80), 1, cv2.LINE_AA)
        metadata = {
            "view": view,
            "projection": "orthographic",
            "renderer": "opencascade_hlr",
            "visible_segment_count": len(result.visible),
            "hidden_segment_count": len(result.hidden),
            "hlr": result.to_dict(),
        }
        return canvas, metadata

    def _point_in_polygon(self, point: Point2, polygon: List[Point2], eps: float = 1e-8) -> bool:
        x, y = point
        inside = False
        for i in range(len(polygon)):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % len(polygon)]
            cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
            if abs(cross) <= eps and min(x1, x2) - eps <= x <= max(x1, x2) + eps \
                    and min(y1, y2) - eps <= y <= max(y1, y2) + eps:
                return True
            if (y1 > y) != (y2 > y):
                x_intersect = (x2 - x1) * (y - y1) / (y2 - y1) + x1
                if x <= x_intersect + eps:
                    inside = not inside
        return inside

    def _depth_on_face(self, point: Point2, vertices: List[Point3]) -> Optional[float]:
        rows = []
        values = []
        for vertex in vertices:
            rows.append([vertex[0], vertex[1], 1.0])
            values.append(vertex[2])
            if len(rows) == 3:
                matrix = np.array(rows, dtype=float)
                if abs(np.linalg.det(matrix)) > 1e-9:
                    a, b, c = np.linalg.solve(matrix, np.array(values, dtype=float))
                    return float(a * point[0] + b * point[1] + c)
                rows.pop(1)
                values.pop(1)
        return None

    def _sample_hidden(self, sample: Point3, projected: Dict[str, Point3],
                       faces: List[Dict[str, Any]], edge_vertices: set) -> bool:
        point = (sample[0], sample[1])
        edge_depth = sample[2]

        for face in faces:
            if face.get("highlight"):
                continue
            if face.get("occluder", True) is False:
                continue
            face_vertices = set(face["vertices"])
            if edge_vertices.issubset(face_vertices):
                continue
            projected_vertices = [projected[v] for v in face["vertices"]]
            polygon = [(p[0], p[1]) for p in projected_vertices]
            if len(set(polygon)) < 3:
                continue
            if not self._point_in_polygon(point, polygon):
                continue
            face_depth = self._depth_on_face(point, projected_vertices)
            if face_depth is not None and face_depth > edge_depth + 1e-6:
                return True
        return False

    def _edge_visibility(self, edge: Dict[str, str], projected: Dict[str, Point3],
                         faces: List[Dict[str, Any]]) -> str:
        start = projected[edge["start"]]
        end = projected[edge["end"]]
        edge_vertices = {edge["start"], edge["end"]}
        hidden_votes = 0
        sample_ts = [0.25, 0.5, 0.75]
        for t in sample_ts:
            sample = (
                start[0] * (1 - t) + end[0] * t,
                start[1] * (1 - t) + end[1] * t,
                start[2] * (1 - t) + end[2] * t,
            )
            if self._sample_hidden(sample, projected, faces, edge_vertices):
                hidden_votes += 1
        return "hidden" if hidden_votes >= 2 else "visible"

    def _draw_dashed(self, canvas: np.ndarray, start: Tuple[int, int], end: Tuple[int, int],
                     color: Tuple[int, int, int], width: int = 1) -> None:
        p1 = np.array(start, dtype=float)
        p2 = np.array(end, dtype=float)
        vec = p2 - p1
        length = float(np.linalg.norm(vec))
        if length <= 0:
            return
        direction = vec / length
        distance = 0.0
        while distance < length:
            s = p1 + direction * distance
            e = p1 + direction * min(distance + 10, length)
            cv2.line(canvas, tuple(s.astype(int)), tuple(e.astype(int)), color, width, cv2.LINE_AA)
            distance += 18

    def _draw_analytic_surfaces(self, canvas: np.ndarray, scene: Dict[str, Any],
                                view: str, projected: Dict[str, Point3], pixels: Dict[str, Tuple[int, int]],
                                normalize) -> Dict[str, str]:
        visibility = {}
        for surface in scene["solid"].get("analytic_surfaces", []):
            if surface["type"] != "sphere":
                continue
            center_id = surface["center"]
            radius = float(surface["radius"]["value"])
            center = projected[center_id]
            pixel_center = pixels[center_id]
            px_right = normalize((center[0] + radius, center[1], center[2]))
            px_up = normalize((center[0], center[1] + radius, center[2]))
            rx = max(abs(px_right[0] - pixel_center[0]), 8)
            ry = max(abs(px_up[1] - pixel_center[1]), 8)
            cv2.ellipse(canvas, pixel_center, (rx, ry), 0, 0, 360, (15, 15, 15), 2, cv2.LINE_AA)
            cv2.ellipse(canvas, pixel_center, (rx, max(ry // 3, 5)), 0, 0, 360, (155, 155, 155), 1, cv2.LINE_AA)
            visibility[surface["id"]] = "visible"
        return visibility

    def _render_view(self, scene: Dict[str, Any], view: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        width, height = self.canvas_size
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        points_3d = self._point_values(scene)
        projected = {name: self._project(coord, view) for name, coord in points_3d.items()}
        normalize = self._normalizer(projected)
        pixels = {name: normalize(coord) for name, coord in projected.items()}
        faces = scene["solid"]["faces"]

        # Fill highlighted faces first, far to near.
        highlighted = [face for face in faces if face.get("highlight")]
        for face in sorted(highlighted, key=lambda f: sum(projected[v][2] for v in f["vertices"]) / len(f["vertices"])):
            poly = np.array([pixels[v] for v in face["vertices"]], dtype=np.int32)
            cv2.fillPoly(canvas, [poly], (232, 245, 255))
            cv2.polylines(canvas, [poly], True, (0, 122, 204), 2, cv2.LINE_AA)

        hidden_edges = []
        visible_edges = []
        highlight_edges = set()
        for face in highlighted:
            verts = face["vertices"]
            for idx, start in enumerate(verts):
                end = verts[(idx + 1) % len(verts)]
                highlight_edges.add(tuple(sorted((start, end))))

        edge_visibility = {}
        analytic_visibility = self._draw_analytic_surfaces(canvas, scene, view, projected, pixels, normalize)
        for edge in scene["solid"]["edges"]:
            key = tuple(sorted((edge["start"], edge["end"])))
            visibility = "section" if key in highlight_edges else self._edge_visibility(edge, projected, faces)
            edge_visibility[edge["id"]] = visibility
            item = (edge, pixels[edge["start"]], pixels[edge["end"]], visibility)
            if visibility == "hidden":
                hidden_edges.append(item)
            else:
                visible_edges.append(item)

        for _, start, end, _ in hidden_edges:
            self._draw_dashed(canvas, start, end, (150, 150, 150), 1)
        for _, start, end, visibility in visible_edges:
            color = (0, 122, 204) if visibility == "section" else (15, 15, 15)
            width_px = 3 if visibility == "section" else 2
            cv2.line(canvas, start, end, color, width_px, cv2.LINE_AA)

        occupied = []
        label_offsets = [(7, -7), (7, 16), (-28, -7), (-28, 16), (12, 0), (0, 22)]
        for name, coord in pixels.items():
            cv2.circle(canvas, coord, 4, (20, 20, 20), -1, cv2.LINE_AA)
            chosen = (coord[0] + 7, coord[1] - 7)
            for dx, dy in label_offsets:
                candidate = (coord[0] + dx, coord[1] + dy)
                box = (candidate[0], candidate[1] - 14, candidate[0] + 24, candidate[1] + 4)
                if all(box[2] < ob[0] or box[0] > ob[2] or box[3] < ob[1] or box[1] > ob[3] for ob in occupied):
                    chosen = candidate
                    occupied.append(box)
                    break
            cv2.putText(canvas, name, chosen, cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)

        label = {
            "front": "FRONT ORTHOGRAPHIC",
            "side": "SIDE ORTHOGRAPHIC",
            "top": "TOP ORTHOGRAPHIC",
            "perspective": "AXONOMETRIC / HIGHLIGHT",
        }[view]
        cv2.putText(canvas, label, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, scene["template_id"], (24, height - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 80, 80), 1, cv2.LINE_AA)

        metadata = {
            "view": view,
            "projection": "orthographic" if view != "perspective" else "axonometric",
            "points_2d": {
                name: {"x": int(coord[0]), "y": int(coord[1]), "depth": float(projected[name][2])}
                for name, coord in pixels.items()
            },
            "edge_visibility": edge_visibility,
            "analytic_visibility": analytic_visibility,
        }
        return canvas, metadata

    def render(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        scene_id = scene["scene_id"]
        paths = {}
        metadata = {}
        images = []
        for view in ["front", "side", "top", "perspective"]:
            hlr_render = self._render_hlr_view(scene, view)
            if hlr_render is not None:
                image, meta = hlr_render
            else:
                image, meta = self._render_view(scene, view)
            path = self.output_dir / f"{scene_id}_{view}.png"
            cv2.imwrite(str(path), image)
            paths[view] = str(path)
            metadata[view] = meta
            images.append(image)
        combined = np.vstack([np.hstack([images[0], images[1]]), np.hstack([images[2], images[3]])])
        combined_path = self.output_dir / f"{scene_id}_combined.png"
        cv2.imwrite(str(combined_path), combined)
        paths["combined"] = str(combined_path)
        meta_path = self.output_dir / f"{scene_id}_render_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        paths["metadata"] = str(meta_path)
        return {"backend": "textbook_opencv_hlr", "paths": paths, "metadata": metadata}


class BlenderFreestyleRenderer:
    """Generate and optionally execute a Blender Freestyle render script."""

    def __init__(self, output_dir: str = "output/geosymsolid/blender"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def blender_available(self) -> bool:
        return shutil.which("blender") is not None

    def write_script(self, scene: Dict[str, Any]) -> str:
        scene_id = scene["scene_id"]
        script_path = self.output_dir / f"{scene_id}_freestyle.py"
        stl_path = scene.get("cad_validation", {}).get("stl_path")
        script = f"""
import bpy

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

stl_path = {stl_path!r}
if stl_path:
    bpy.ops.wm.stl_import(filepath=stl_path)

for obj in bpy.context.scene.objects:
    obj.select_set(True)

bpy.context.scene.render.use_freestyle = True
bpy.context.scene.view_settings.view_transform = 'Standard'
bpy.context.scene.render.resolution_x = 1200
bpy.context.scene.render.resolution_y = 900

# Production path:
# - add transparent materials
# - configure orthographic cameras for front/side/top
# - configure Freestyle visible and hidden line sets
# - render PNG/SVG outputs
"""
        script_path.write_text(textwrap.dedent(script).strip() + "\n", encoding="utf-8")
        return str(script_path)

    def render(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        script_path = self.write_script(scene)
        result = {"backend": "blender_freestyle", "available": self.blender_available(), "script_path": script_path, "paths": {}}
        if not result["available"]:
            result["error"] = "blender executable is not available"
            return result
        proc = subprocess.run(["blender", "-b", "--python", script_path], cwd=str(self.output_dir),
                              capture_output=True, text=True, check=False)
        result.update({"returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]})
        return result
