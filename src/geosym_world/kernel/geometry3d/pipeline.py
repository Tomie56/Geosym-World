"""
End-to-end GeoSymSolid sample pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
import sympy as sp

from .kernel import GeoSymSolidKernel
from .qa import GeoSymSolidQAGenerator
from .renderer import BlenderFreestyleRenderer, TextbookSolidRenderer


class GeoSymSolidPipeline:
    """Generate one GeoSymSolid pilot sample with renderings, QA, and GT."""

    def __init__(self, output_dir: str = "output/geosymsolid"):
        self.output_dir = Path(output_dir)
        self.scene_dir = self.output_dir / "scenes"
        self.gt_dir = self.output_dir / "gt"
        self.qa_dir = self.output_dir / "qa"
        self.trace_dir = self.output_dir / "traces"
        self.scene_dir.mkdir(parents=True, exist_ok=True)
        self.gt_dir.mkdir(parents=True, exist_ok=True)
        self.qa_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

        self.kernel = GeoSymSolidKernel(output_dir=str(self.output_dir))
        self.renderer = TextbookSolidRenderer(output_dir=str(self.output_dir / "renders"))
        self.blender_renderer = BlenderFreestyleRenderer(output_dir=str(self.output_dir / "blender"))
        self.qa_generator = GeoSymSolidQAGenerator()

    def run_sample(self, scene_id: str = "geosymsolid_0000") -> Dict[str, Any]:
        return self.run_cuboid_sample(scene_id=scene_id, a=sp.Integer(6), b=sp.Integer(4), c=sp.Integer(3))

    def _write_scene_outputs(self, scene: Dict[str, Any], render_result: Dict[str, Any],
                             blender_result: Dict[str, Any], qa: List[Dict[str, Any]],
                             write_benchmark: bool) -> Dict[str, Any]:
        scene_id = scene["scene_id"]
        scene_path = self.scene_dir / f"{scene_id}_state.json"
        gt_path = self.gt_dir / f"{scene_id}_gt.json"
        qa_path = self.qa_dir / f"{scene_id}_qa.jsonl"
        trace_path = self.trace_dir / f"{scene_id}_trace.json"
        benchmark_path = self.output_dir / "benchmark.jsonl"

        gt = {
            "scene_id": scene_id,
            "template_id": scene["template_id"],
            "quantities": scene["quantities"],
            "task_targets": scene["task_targets"],
            "operation_trace": scene["operation_trace"],
            "formal_state": scene["formal_state"],
            "cad_validation": scene["cad_validation"],
            "render_metadata_path": render_result["paths"]["metadata"],
        }

        with open(scene_path, "w", encoding="utf-8") as f:
            json.dump(scene, f, indent=2, ensure_ascii=False)
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt, f, indent=2, ensure_ascii=False)
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump({
                "scene_id": scene_id,
                "template_id": scene["template_id"],
                "base_template": scene.get("base_template"),
                "parameters": scene.get("parameters"),
                "evolution_trace": scene.get("evolution_trace", []),
                "formal_state": scene.get("formal_state", []),
            }, f, indent=2, ensure_ascii=False)
        with open(qa_path, "w", encoding="utf-8") as f:
            for item in qa:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

        record = {
            "scene_id": scene_id,
            "template_id": scene["template_id"],
            "state_path": str(scene_path),
            "gt_path": str(gt_path),
            "qa_path": str(qa_path),
            "trace_path": str(trace_path),
            "render_paths": render_result["paths"],
            "cad_validation": scene["cad_validation"],
            "blender": blender_result,
            "num_qa": len(qa),
        }
        if write_benchmark:
            with open(benchmark_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")

        return {
            "scene": scene,
            "gt": gt,
            "qa": qa,
            "render": render_result,
            "blender": blender_result,
            "paths": {
                "scene": str(scene_path),
                "gt": str(gt_path),
                "qa": str(qa_path),
                "trace": str(trace_path),
                "benchmark": str(benchmark_path),
                **render_result["paths"],
            },
        }

    def run_cuboid_sample(self, scene_id: str, a: sp.Expr, b: sp.Expr, c: sp.Expr,
                          write_benchmark: bool = True) -> Dict[str, Any]:
        scene = self.kernel.create_cuboid_section_scene(scene_id=scene_id, a=a, b=b, c=c)
        render_result = self.renderer.render(scene)
        blender_result = self.blender_renderer.render(scene)
        qa = self.qa_generator.generate(scene, render_result)
        return self._write_scene_outputs(scene, render_result, blender_result, qa, write_benchmark)

    def run_template_sample(self, scene_id: str, template: str, params: Dict[str, sp.Expr],
                            write_benchmark: bool = True) -> Dict[str, Any]:
        if template == "cuboid_section":
            scene = self.kernel.create_cuboid_section_scene(scene_id=scene_id, **params)
        elif template == "l_shape_composite":
            scene = self.kernel.create_l_shape_scene(scene_id=scene_id, **params)
        elif template == "prism_roof":
            scene = self.kernel.create_prism_roof_scene(scene_id=scene_id, **params)
        elif template == "cylinder_axis_section":
            scene = self.kernel.create_cylinder_scene(scene_id=scene_id, **params)
        elif template == "cone_axis_section":
            scene = self.kernel.create_cone_scene(scene_id=scene_id, **params)
        elif template == "sphere_great_circle":
            scene = self.kernel.create_sphere_scene(scene_id=scene_id, **params)
        elif template == "tangent_sphere_cylinder":
            scene = self.kernel.create_tangent_sphere_cylinder_scene(scene_id=scene_id, **params)
        else:
            raise ValueError(f"Unknown GeoSymSolid template: {template}")
        render_result = self.renderer.render(scene)
        blender_result = self.blender_renderer.render(scene)
        qa = self.qa_generator.generate(scene, render_result)
        return self._write_scene_outputs(scene, render_result, blender_result, qa, write_benchmark)

    def run_batch(self, num_samples: int = 12, scene_prefix: str = "geosymsolid") -> Dict[str, Any]:
        """
        Generate a small deterministic pilot batch.

        Dimensions are intentionally simple but varied enough to produce integer,
        square-root, and fraction-like symbolic expressions.
        """
        template_pool = [
            ("cuboid_section", {"a": sp.Integer(6), "b": sp.Integer(4), "c": sp.Integer(3)}),
            ("l_shape_composite", {"a": sp.Integer(8), "b": sp.Integer(6), "h": sp.Integer(3), "cut_x": sp.Integer(3), "cut_y": sp.Integer(2)}),
            ("prism_roof", {"a": sp.Integer(8), "b": sp.Integer(4), "wall_h": sp.Integer(3), "roof_h": sp.Integer(2)}),
            ("cylinder_axis_section", {"r": sp.Integer(3), "h": sp.Integer(7), "dx": sp.Integer(0), "dy": sp.Integer(0)}),
            ("cone_axis_section", {"r": sp.Integer(3), "h": sp.Integer(6), "dx": sp.Integer(2), "dy": sp.Integer(0)}),
            ("sphere_great_circle", {"r": sp.Integer(4), "dx": sp.Integer(0), "dy": sp.Integer(0), "dz": sp.Integer(4)}),
            ("tangent_sphere_cylinder", {"r": sp.Integer(3), "h": sp.Integer(7)}),
            ("cuboid_section", {"a": sp.Integer(5), "b": sp.Integer(5), "c": sp.Integer(4)}),
            ("l_shape_composite", {"a": sp.Integer(10), "b": sp.Integer(7), "h": sp.Integer(4), "cut_x": sp.Integer(4), "cut_y": sp.Integer(3)}),
            ("prism_roof", {"a": sp.Integer(10), "b": sp.Integer(5), "wall_h": sp.Integer(4), "roof_h": sp.Integer(3)}),
            ("cylinder_axis_section", {"r": sp.Integer(2), "h": sp.Integer(8), "dx": sp.Integer(2), "dy": sp.Integer(1)}),
            ("cone_axis_section", {"r": sp.Integer(4), "h": sp.Integer(5), "dx": sp.Integer(-1), "dy": sp.Integer(1)}),
            ("sphere_great_circle", {"r": sp.Integer(3), "dx": sp.Integer(1), "dy": sp.Integer(2), "dz": sp.Integer(3)}),
            ("cuboid_section", {"a": sp.Integer(8), "b": sp.Integer(3), "c": sp.Integer(4)}),
            ("l_shape_composite", {"a": sp.Integer(9), "b": sp.Integer(5), "h": sp.Integer(5), "cut_x": sp.Integer(5), "cut_y": sp.Integer(2)}),
            ("prism_roof", {"a": sp.Integer(6), "b": sp.Integer(6), "wall_h": sp.Integer(3), "roof_h": sp.Integer(4)}),
        ]
        records: List[Dict[str, Any]] = []
        benchmark_path = self.output_dir / "benchmark.jsonl"

        with open(benchmark_path, "w", encoding="utf-8") as f:
            for index in range(num_samples):
                template, params = template_pool[index % len(template_pool)]
                scene_id = f"{scene_prefix}_{index:04d}"
                result = self.run_template_sample(
                    scene_id=scene_id,
                    template=template,
                    params=params,
                    write_benchmark=False
                )
                record = {
                    "scene_id": scene_id,
                    "template_id": result["scene"]["template_id"],
                    "state_path": result["paths"]["scene"],
                    "gt_path": result["paths"]["gt"],
                    "qa_path": result["paths"]["qa"],
                    "trace_path": result["paths"]["trace"],
                    "render_paths": {
                        key: result["paths"][key]
                        for key in ["front", "side", "top", "perspective", "combined", "metadata"]
                    },
                    "cad_validation": result["scene"]["cad_validation"],
                    "num_qa": len(result["qa"])
                }
                records.append(record)
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")

        return {
            "num_samples": len(records),
            "benchmark": str(benchmark_path),
            "records": records
        }
