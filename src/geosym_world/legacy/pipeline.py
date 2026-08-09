"""Deterministic, parameterized orchestration for the original GeoSym tools."""

from __future__ import annotations

import copy
import json
import multiprocessing
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Sequence


ALL_STAGES = ("template", "builder", "shader", "gt", "qa")


@dataclass(frozen=True)
class LegacyPipelineOptions:
    output_root: Path
    records: int = 1
    seed: int = 42
    workers: int = 1
    stages: tuple[str, ...] = ALL_STAGES
    rounds_distribution: Dict[int, int] | None = None
    shader_attempts: int | None = None
    shader_enabled: bool | None = None
    questions_per_record: int | None = None
    question_types: tuple[str, ...] = ()
    task_timeout_seconds: float | None = 300.0
    max_tasks_per_child: int | None = None
    fail_fast: bool = False
    input_base: Path | None = None
    input_enhanced: Path | None = None
    input_shaded: Path | None = None
    input_gt: Path | None = None

    def __post_init__(self) -> None:
        if self.records < 1:
            raise ValueError("records must be positive")
        if self.workers < 1:
            raise ValueError("workers must be positive")
        if self.task_timeout_seconds is not None and self.task_timeout_seconds <= 0:
            raise ValueError("task_timeout_seconds must be positive or None")
        if self.max_tasks_per_child is not None and self.max_tasks_per_child < 1:
            raise ValueError("max_tasks_per_child must be positive or None")
        if self.max_tasks_per_child is not None and sys.version_info < (3, 11):
            raise ValueError("max_tasks_per_child requires Python 3.11 or newer")
        unknown = set(self.stages).difference(ALL_STAGES)
        if unknown:
            raise ValueError(f"Unknown legacy stages: {sorted(unknown)}")


class LegacyCompatiblePipeline:
    """Run the legacy schema end to end with deterministic process isolation.

    Geometry generation and enhancement use processes because the original
    classes mutate module-level random state. Results retain input order while
    process boundaries prevent cross-sample RNG interference and allow hard
    termination of symbolic calculations that exceed their deadline.
    """

    def __init__(self, config: Dict[str, Any], options: LegacyPipelineOptions):
        self.config = _normalized_config(config, options)
        self.options = options
        self.root = options.output_root.resolve()
        self.paths = _artifact_paths(self.root)
        self.failures: list[Dict[str, Any]] = []
        self.stage_metrics: Dict[str, Any] = {}

    @classmethod
    def from_file(cls, config_path: str | Path, options: LegacyPipelineOptions) -> "LegacyCompatiblePipeline":
        with Path(config_path).open(encoding="utf-8") as file:
            config = json.load(file)
        return cls(config, options)

    def run(self) -> Dict[str, Any]:
        self._prepare_workspace()
        records: list[Dict[str, Any]] = []
        stages = set(self.options.stages)

        if "template" in stages and "builder" in stages and not self.options.input_base:
            records = self._run_fused_generation()
        elif "template" in stages:
            records = self._run_template()
        elif self.options.input_base:
            records = _read_jsonl(self.options.input_base)

        if "builder" in stages and not ("template" in stages and not self.options.input_base):
            if not records:
                records = _read_required(self.options.input_base, "builder requires base records")
            records = self._run_builder(records)
        elif self.options.input_enhanced:
            records = _read_jsonl(self.options.input_enhanced)

        if "shader" in self.options.stages:
            if not records:
                records = _read_required(self.options.input_enhanced, "shader requires enhanced records")
            records = self._run_shader(records)
        elif self.options.input_shaded:
            records = _read_jsonl(self.options.input_shaded)

        if "gt" in self.options.stages:
            if not records:
                records = _read_required(self.options.input_shaded or self.options.input_enhanced, "gt requires geometry records")
            records = self._run_gt(records)
        elif self.options.input_gt:
            records = _read_jsonl(self.options.input_gt)

        qa_rows: list[Dict[str, Any]] = []
        if "qa" in self.options.stages:
            if not records:
                records = _read_required(self.options.input_gt, "qa requires GT records")
            qa_rows = self._run_qa(records)

        artifacts = {name: str(path) for name, path in self.paths.items() if path.exists()}
        artifacts["manifest"] = str(self.paths["manifest"])
        manifest = {
            "engine": "legacy_compatible_optimized",
            "output_root": str(self.root),
            "options": _jsonable_options(self.options),
            "artifacts": artifacts,
            "stage_metrics": self.stage_metrics,
            "num_final_records": len(records),
            "num_qa": len(qa_rows),
            "failures": self.failures,
            "config": self.config,
        }
        _write_json(self.paths["manifest"], manifest)
        return manifest

    def _prepare_workspace(self) -> None:
        for path in self.paths.values():
            if path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)

    def _run_template(self) -> list[Dict[str, Any]]:
        started = time.perf_counter()
        config = self.config["template"]
        payloads = [(config, self.options.seed + 42 * index, index) for index in range(self.options.records)]
        results = self._collect("template", _template_worker, payloads)
        _write_jsonl(self.paths["base"], results)
        self.stage_metrics["template"].update({"seconds": time.perf_counter() - started, "output": str(self.paths["base"])})
        return results

    def _run_fused_generation(self) -> list[Dict[str, Any]]:
        started = time.perf_counter()
        payloads = [
            (
                self.config["template"],
                self.config["builder"],
                self.options.seed + 42 * index,
                self.options.seed + 100_000 + 42 * index,
                index,
            )
            for index in range(self.options.records)
        ]
        groups = self._collect("generation", _generation_worker, payloads)
        generation_metrics = self.stage_metrics["generation"]
        base_records = [group["base"] for group in groups]
        enhanced_records = [record for group in groups for record in group["enhanced"]]
        _write_jsonl(self.paths["base"], base_records)
        _write_jsonl(self.paths["enhanced"], enhanced_records)
        elapsed = time.perf_counter() - started
        self.stage_metrics["template"] = {
            "requested": len(payloads),
            "accepted": len(base_records),
            "rejected": len(payloads) - len(base_records),
            "timed_out": generation_metrics["timed_out"],
            "seconds": elapsed,
            "output": str(self.paths["base"]),
            "fused_with_builder": True,
        }
        self.stage_metrics["builder"] = {
            "requested": len(base_records),
            "accepted": len(enhanced_records),
            "rejected": 0,
            "timed_out": generation_metrics["timed_out"],
            "seconds": elapsed,
            "output": str(self.paths["enhanced"]),
            "fused_with_template": True,
        }
        self.stage_metrics.pop("generation", None)
        return enhanced_records

    def _run_builder(self, base_records: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
        started = time.perf_counter()
        config = self.config["builder"]
        payloads = [
            (record, config, self.options.seed + 100_000 + 42 * index, index)
            for index, record in enumerate(base_records)
        ]
        nested = self._collect("builder", _builder_worker, payloads)
        records = [record for group in nested for record in group]
        _write_jsonl(self.paths["enhanced"], records)
        self.stage_metrics["builder"].update({"seconds": time.perf_counter() - started, "output": str(self.paths["enhanced"])})
        return records

    def _run_shader(self, enhanced_records: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
        started = time.perf_counter()
        _limit_native_threads()
        _write_jsonl(self.paths["enhanced"], enhanced_records)
        if not self.config["shader"].get("enabled", True):
            records = self._render_and_annotate(enhanced_records)
            _write_jsonl(self.paths["shaded"], records)
            self.stage_metrics["shader"] = {
                "accepted": len(records),
                "rejected": 0,
                "matched_regions": 0,
                "seconds": time.perf_counter() - started,
                "output": str(self.paths["shaded"]),
                "shader_enabled": False,
            }
            return records
        random.seed(self.options.seed + 200_000)
        try:
            import numpy as np

            np.random.seed((self.options.seed + 200_000) % (2**32 - 1))
        except ImportError:  # pragma: no cover - legacy shader requires NumPy.
            pass
        from geosym_world.legacy_tools.shader import EnhancedDrawer

        tool_config = {
            "global": {"output_root": str(self.root)},
            "drawer": self.config["drawer"],
            "shader": {**self.config["shader"], "jsonl_path": str(self.paths["shaded"])},
            "annotator": self.config["annotator"],
            "region_extractor": self.config["region_extractor"],
        }
        processor = EnhancedDrawer(tool_config)
        processor.set_enhanced_data(list(enhanced_records))
        processor.process()
        records = _read_jsonl(self.paths["shaded"])
        self.stage_metrics["shader"] = {
            "accepted": len(records),
            "rejected": processor.failure,
            "matched_regions": processor.success,
            "seconds": time.perf_counter() - started,
            "output": str(self.paths["shaded"]),
        }
        return records

    def _render_and_annotate(self, enhanced_records: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
        import cv2

        from geosym_world.legacy_tools.drawer import GeometryDrawer
        from geosym_world.legacy_tools.shader import CoordTransformer, GeometryAnnotator

        drawer = GeometryDrawer(self.config["drawer"])
        image_paths = drawer.batch_draw()
        geometry_params = drawer.get_geometry_params()
        annotator = GeometryAnnotator(self.config["annotator"])
        annotated_dir = self.root / "images" / "annotated"
        annotated_dir.mkdir(parents=True, exist_ok=True)
        records: list[Dict[str, Any]] = []
        for index, (record, raw_path, params) in enumerate(zip(enhanced_records, image_paths, geometry_params)):
            image = cv2.imread(raw_path)
            if image is None:
                self.failures.append({"stage": "shader", "index": index, "error": f"Unable to read {raw_path}"})
                continue
            transformer = CoordTransformer(**params)
            annotated = annotator.draw_annotations(image, record, transformer)
            annotated_path = annotated_dir / f"annotated_raw_{Path(raw_path).name}"
            cv2.imwrite(str(annotated_path), annotated)
            records.append(
                {
                    **record,
                    "raw_path": str(raw_path),
                    "annotated_raw_path": str(annotated_path),
                    "shaded_path": None,
                    "annotated_shaded_path": None,
                    "shader_enabled": False,
                }
            )
        return records

    def _run_gt(self, records: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
        started = time.perf_counter()
        payloads = [(record, index) for index, record in enumerate(records)]
        results = self._collect("gt", _gt_worker, payloads)
        _write_jsonl(self.paths["gt"], results)
        self.stage_metrics["gt"].update({"seconds": time.perf_counter() - started, "output": str(self.paths["gt"])})
        return results

    def _run_qa(self, records: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
        started = time.perf_counter()
        config = self.config["qa"]
        payloads = [(record, config, self.options.seed + 300_000 + index, index) for index, record in enumerate(records)]
        nested = self._collect("qa", _qa_worker, payloads)
        rows = [row for group in nested for row in group]
        _write_jsonl(self.paths["qa"], rows)
        self.stage_metrics["qa"].update({"seconds": time.perf_counter() - started, "output": str(self.paths["qa"])})
        return rows

    def _collect(self, stage: str, worker: Callable[[tuple], Dict[str, Any]], payloads: Iterable[tuple]) -> list[Any]:
        payload_list = list(payloads)
        if self.options.task_timeout_seconds is not None:
            envelopes = _run_isolated_tasks(
                worker,
                payload_list,
                workers=self.options.workers,
                timeout_seconds=self.options.task_timeout_seconds,
            )
        elif self.options.workers == 1:
            _limit_native_threads()
            envelopes = map(worker, payload_list)
        else:
            _limit_native_threads()
            executor_options = {
                "max_workers": self.options.workers,
                "mp_context": multiprocessing.get_context("spawn"),
                "initializer": _limit_native_threads,
            }
            if self.options.max_tasks_per_child is not None:
                executor_options["max_tasks_per_child"] = self.options.max_tasks_per_child
            with ProcessPoolExecutor(
                **executor_options,
            ) as executor:
                envelopes = list(executor.map(worker, payload_list, chunksize=1))
        accepted: list[Any] = []
        rejected = 0
        timed_out = 0
        for envelope in envelopes:
            if envelope["ok"]:
                accepted.append(envelope["value"])
            else:
                rejected += 1
                timed_out += envelope.get("exception_type") == "TaskTimeout"
                failure = {"stage": stage, **envelope}
                self.failures.append(failure)
                if self.options.fail_fast:
                    raise RuntimeError(f"{stage} failed at index {envelope['index']}: {envelope['error']}")
        self.stage_metrics[stage] = {
            "requested": len(payload_list),
            "accepted": len(accepted),
            "rejected": rejected,
            "timed_out": timed_out,
        }
        return accepted


def _run_isolated_tasks(
    worker: Callable[[tuple], Dict[str, Any]],
    payloads: Sequence[tuple],
    *,
    workers: int,
    timeout_seconds: float,
) -> list[Dict[str, Any]]:
    """Run each task in a killable process while preserving input order."""
    if not payloads:
        return []
    _limit_native_threads()
    context = multiprocessing.get_context("spawn")
    pending = iter(enumerate(payloads))
    active: Dict[int, Dict[str, Any]] = {}
    results: list[Dict[str, Any] | None] = [None] * len(payloads)

    def launch() -> bool:
        try:
            position, payload = next(pending)
        except StopIteration:
            return False
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_isolated_worker_entry, args=(sender, worker, payload))
        process.start()
        sender.close()
        active[position] = {
            "process": process,
            "receiver": receiver,
            "started": time.monotonic(),
            "index": _payload_index(payload, position),
        }
        return True

    while len(active) < workers and launch():
        pass

    while active:
        now = time.monotonic()
        completed: list[int] = []
        for position, task in active.items():
            process = task["process"]
            receiver = task["receiver"]
            if receiver.poll():
                try:
                    results[position] = receiver.recv()
                except EOFError:
                    results[position] = _process_exit_error(task["index"], process.exitcode)
                _stop_process(process)
                receiver.close()
                completed.append(position)
            elif not process.is_alive():
                process.join()
                if receiver.poll(0.1):
                    try:
                        results[position] = receiver.recv()
                    except EOFError:
                        results[position] = _process_exit_error(task["index"], process.exitcode)
                else:
                    results[position] = _process_exit_error(task["index"], process.exitcode)
                receiver.close()
                completed.append(position)
            elif now - task["started"] >= timeout_seconds:
                _stop_process(process)
                receiver.close()
                results[position] = {
                    "ok": False,
                    "index": task["index"],
                    "error": f"Task exceeded {timeout_seconds:g} seconds",
                    "exception_type": "TaskTimeout",
                }
                completed.append(position)

        for position in completed:
            active.pop(position)
            launch()
        if active and not completed:
            time.sleep(0.01)

    return [result for result in results if result is not None]


def _isolated_worker_entry(sender, worker: Callable[[tuple], Dict[str, Any]], payload: tuple) -> None:
    try:
        _limit_native_threads()
        sender.send(worker(payload))
    except BaseException as exc:
        sender.send(_error(_payload_index(payload, -1), exc))
    finally:
        sender.close()


def _stop_process(process: multiprocessing.Process) -> None:
    process.join(timeout=0.2)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join()


def _process_exit_error(index: int, exitcode: int | None) -> Dict[str, Any]:
    return {
        "ok": False,
        "index": index,
        "error": f"Worker exited without a result (exit code {exitcode})",
        "exception_type": "WorkerProcessError",
    }


def _payload_index(payload: tuple, fallback: int) -> int:
    return payload[-1] if payload and isinstance(payload[-1], int) else fallback


def _limit_native_threads() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    try:
        import cv2

        cv2.setNumThreads(0)
    except ImportError:
        pass


def _template_worker(payload: tuple[Dict[str, Any], int, int]) -> Dict[str, Any]:
    config, seed, index = payload
    try:
        from geosym_world.legacy_tools.template import TemplateGenerator

        generator = TemplateGenerator(copy.deepcopy(config), seed=seed)
        generator.generate_base_shape()
        generator.generate_derivations()
        record = generator.export_json()
        record.setdefault("base_id", f"base_{index:06d}")
        return _ok(index, record)
    except Exception as exc:
        return _error(index, exc)


def _generation_worker(payload: tuple[Dict[str, Any], Dict[str, Any], int, int, int]) -> Dict[str, Any]:
    template_config, builder_config, template_seed, builder_seed, index = payload
    try:
        from geosym_world.legacy_tools.builder import RandomGeometryBuilder
        from geosym_world.legacy_tools.template import TemplateGenerator

        generator = TemplateGenerator(copy.deepcopy(template_config), seed=template_seed)
        generator.generate_base_shape()
        generator.generate_derivations()
        base = generator.export_json()
        base.setdefault("base_id", f"base_{index:06d}")
        builder = RandomGeometryBuilder(copy.deepcopy(base), base_id=f"base_{index:06d}")
        enhanced = builder.generate_enhancements({**copy.deepcopy(builder_config), "seed": builder_seed})
        return _ok(index, {"base": base, "enhanced": enhanced})
    except Exception as exc:
        return _error(index, exc)


def _builder_worker(payload: tuple[Dict[str, Any], Dict[str, Any], int, int]) -> Dict[str, Any]:
    record, config, seed, index = payload
    try:
        from geosym_world.legacy_tools.builder import RandomGeometryBuilder

        builder = RandomGeometryBuilder(copy.deepcopy(record), base_id=f"base_{index:06d}")
        run_config = {**copy.deepcopy(config), "seed": seed}
        return _ok(index, builder.generate_enhancements(run_config))
    except Exception as exc:
        return _error(index, exc)


def _gt_worker(payload: tuple[Dict[str, Any], int]) -> Dict[str, Any]:
    record, index = payload
    try:
        from geosym_world.legacy_tools.gt import GeometryCalculator

        return _ok(index, GeometryCalculator().calculate_single(record))
    except Exception as exc:
        return _error(index, exc)


def _qa_worker(payload: tuple[Dict[str, Any], Dict[str, Any], int, int]) -> Dict[str, Any]:
    record, config, seed, index = payload
    try:
        from geosym_world.legacy_tools.qa_template import QAGenerator

        random.seed(seed)
        generator = QAGenerator(config)
        rows = generator.generate(
            geo_data=record,
            num_questions=int(config["num_questions_per_record"]),
            question_types=list(config["question_types"]),
            type_weights=config.get("type_weights") or {},
        )
        return _ok(index, rows)
    except Exception as exc:
        return _error(index, exc)


def _normalized_config(config: Dict[str, Any], options: LegacyPipelineOptions) -> Dict[str, Any]:
    normalized = copy.deepcopy(config)
    for section in ("global", "template", "builder", "drawer", "shader", "region_extractor", "annotator", "gt", "qa"):
        normalized.setdefault(section, {})
    root = options.output_root.resolve()
    normalized["global"].update({"output_root": str(root), "main_seed": options.seed, "thread_num": options.workers})
    normalized["template"].update(
        {
            "n": options.records,
            "seed": options.seed + 1,
            "output_dir": str(root / "json" / "base"),
            "output_filename": "base.jsonl",
        }
    )
    normalized["builder"].update({"seed": options.seed + 2, "output_dir": str(root / "json" / "enhanced")})
    if options.rounds_distribution is not None:
        normalized["builder"]["rounds_distribution"] = {str(key): int(value) for key, value in options.rounds_distribution.items()}
    normalized["drawer"].update({"jsonl_path": str(root / "json" / "enhanced" / "enhanced.jsonl"), "output_dir": str(root / "images" / "raw")})
    normalized["shader"]["seed"] = options.seed + 3
    if options.shader_attempts is not None:
        normalized["shader"]["x_attempts"] = max(1, options.shader_attempts)
    if options.shader_enabled is not None:
        normalized["shader"]["enabled"] = options.shader_enabled
    normalized["qa"].update({"seed": options.seed + 4, "output_dir": str(root / "qa")})
    normalized["qa"]["num_questions_per_record"] = int(
        options.questions_per_record
        if options.questions_per_record is not None
        else normalized["qa"].get("num_questions_per_record", normalized["qa"].get("num_questions_per_geo", 1))
    )
    if options.question_types:
        normalized["qa"]["question_types"] = list(options.question_types)
    normalized["qa"].setdefault("question_types", ["length", "angle"])
    return normalized


def _artifact_paths(root: Path) -> Dict[str, Path]:
    return {
        "base": root / "json" / "base" / "base.jsonl",
        "enhanced": root / "json" / "enhanced" / "enhanced.jsonl",
        "shaded": root / "json" / "shaded" / "shaded.jsonl",
        "gt": root / "json" / "final" / "with_gt.jsonl",
        "qa": root / "qa" / "qa.jsonl",
        "manifest": root / "manifest.json",
    }


def _ok(index: int, value: Any) -> Dict[str, Any]:
    return {"ok": True, "index": index, "value": value}


def _error(index: int, exc: Exception) -> Dict[str, Any]:
    return {"ok": False, "index": index, "error": str(exc), "exception_type": type(exc).__name__}


def _read_required(path: Path | None, message: str) -> list[Dict[str, Any]]:
    if path is None:
        raise ValueError(message)
    return _read_jsonl(path)


def _read_jsonl(path: str | Path) -> list[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            json.dump(row, file, ensure_ascii=False)
            file.write("\n")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _jsonable_options(options: LegacyPipelineOptions) -> Dict[str, Any]:
    payload = asdict(options)
    for key, value in payload.items():
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload
