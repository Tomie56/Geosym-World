"""Solver-grounded interactive geometry environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from geosym_world_geometry.builder import ConstructionBuilder
from geosym_world_geometry.rendering import state_to_render_scene, write_png
from geosym_world_geometry.solver import FactClosureSolver

from .feedback import EXECUTION_ERROR, INVALID_SCHEMA, UNSUPPORTED_OPERATION
from .registry import ToolRegistry
from .resolver import ObjectReferenceResolver
from .schema import AgentAction, GeoClawObservation, ToolFeedback
from .scoring import score_trajectory
from .task import GeoClawTask
from .tools import CalculatorTool, ConstructTool, FinalTool, MeasureTool, VerifyTool, ZoomTool
from .trajectory import EpisodeTrajectory, TrajectoryStep
from .utility import classify_utility, fact_keys


class GeoClawEnvironment:
    def __init__(
        self,
        task: GeoClawTask,
        *,
        output_dir: str | Path = "outputs/geoclaw",
        expose_feedback: bool = True,
        render_steps: bool = True,
        allowed_tools: tuple[str, ...] | None = None,
        validate_task: bool = True,
    ):
        if validate_task:
            task.require_valid()
        self.task = task
        self.base_trace = [dict(item) for item in task.trace]
        self.agent_trace: list[Dict[str, Any]] = []
        self.state = ConstructionBuilder().execute(self.base_trace)
        self.facts = FactClosureSolver().solve(self.state)
        self.resolver = ObjectReferenceResolver()
        registry = ToolRegistry([ZoomTool(), CalculatorTool(), MeasureTool(), ConstructTool(), VerifyTool(), FinalTool()])
        self.allowed_tools = tuple(allowed_tools or registry.names())
        unknown_tools = sorted(set(self.allowed_tools) - set(registry.names()))
        if unknown_tools:
            raise ValueError(f"Unknown allowed tools: {unknown_tools}")
        if "final" not in self.allowed_tools:
            self.allowed_tools = (*self.allowed_tools, "final")
        self.registry = registry.restricted(self.allowed_tools)
        self.output_dir = Path(output_dir) / task.id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.expose_feedback = expose_feedback
        self.render_steps = render_steps
        self.crops: list[Dict[str, Any]] = []
        self.information_signatures: set[tuple[str, str]] = set()
        self.terminated = False
        self.final_answer: str | None = None
        self.final_correct = False
        self.current_image = task.original_image
        self.trajectory = EpisodeTrajectory(
            episode_id=f"episode_{task.id}",
            task_id=task.id,
            environment_config={"max_steps": task.max_steps, "expose_feedback": expose_feedback, "render_steps": render_steps, "allowed_tools": list(self.allowed_tools)},
        )
        if self.current_image is None and render_steps:
            self.current_image = self._render_image(0)

    @property
    def step_count(self) -> int:
        return len(self.trajectory.steps)

    def observe(self) -> GeoClawObservation:
        last_feedback = self.trajectory.steps[-1].feedback if self.trajectory.steps and self.expose_feedback else None
        public_history = [
            {"step_id": step.step_id, "action": step.parsed_action, "feedback": step.feedback if self.expose_feedback else {"status": "hidden"}}
            for step in self.trajectory.steps
        ]
        return GeoClawObservation(
            task_id=self.task.id,
            problem_text=self.task.problem_text,
            current_image=str(self.current_image) if self.current_image else None,
            crops=tuple(self.crops),
            visible_objects=self.resolver.visible_object_ids(self.state),
            action_history=tuple(public_history),
            verifier_feedback=last_feedback,
            budget_used=self.step_count,
            budget_remaining=max(0, self.task.max_steps - self.step_count),
            terminated=self.terminated,
        )

    def step(self, payload: Dict[str, Any]) -> GeoClawObservation:
        if self.terminated:
            raise RuntimeError("Episode has terminated")
        if self.step_count >= self.task.max_steps:
            self.terminated = True
            self._finalize("budget_exhausted")
            return self.observe()
        before = self._state_summary()
        before_facts = fact_keys(self.facts)
        parsed = None
        try:
            parsed = AgentAction.from_dict(payload)
            if parsed.action not in self.registry.names():
                tool_result = _error_result(UNSUPPORTED_OPERATION, "action.tool_disabled", f"Tool is disabled: {parsed.action}")
            else:
                tool_result = self.registry.get(parsed.action).execute(self, parsed.arguments)
        except ValueError as exc:
            tool_result = _error_result(INVALID_SCHEMA, "action.invalid_schema", str(exc))
        except Exception as exc:
            tool_result = _error_result(EXECUTION_ERROR, "action.execution_error", str(exc))
        after_facts = fact_keys(self.facts)
        action_name = parsed.action if parsed else str(payload.get("action", "unknown"))
        utility = classify_utility(self, tool_result.status, before_facts, after_facts, action_name, tool_result.result)
        if tool_result.image_changed and self.render_steps:
            self.current_image = self._render_image(self.step_count + 1)
        feedback = ToolFeedback(
            step_id=self.step_count + 1,
            action=action_name,
            status=tool_result.status,
            utility=utility,
            result=tool_result.result,
            evidence=tuple(tool_result.evidence),
            errors=tuple(tool_result.errors),
            budget_used=self.step_count + 1,
            budget_remaining=max(0, self.task.max_steps - self.step_count - 1),
            image_path=str(self.current_image) if self.current_image else None,
            evidence_strength=str(tool_result.result.get("evidence_strength", "unknown")),
        )
        self.trajectory.steps.append(TrajectoryStep(
            step_id=feedback.step_id,
            raw_action=dict(payload),
            parsed_action=parsed.to_dict() if parsed else None,
            feedback=feedback.to_dict(),
            state_before=before,
            state_after=self._state_summary(),
            audit=tool_result.audit,
        ))
        if action_name == "final" and tool_result.status == "valid":
            self._finalize("final_action")
        elif self.step_count >= self.task.max_steps:
            self.terminated = True
            self._finalize("budget_exhausted")
        return self.observe()

    def replay(self, actions: list[Dict[str, Any]]) -> GeoClawObservation:
        observation = self.observe()
        for action in actions:
            if self.terminated:
                break
            observation = self.step(action)
        return observation

    def finish(self, reason: str = "actions_exhausted") -> None:
        if not self.terminated:
            self._finalize(reason)

    def _render_image(self, step_id: int) -> str:
        output = self.output_dir / "steps" / f"step_{step_id:03d}.png"
        scene = state_to_render_scene(self.state)
        write_png(scene, output)
        return str(output)

    def _state_summary(self) -> Dict[str, Any]:
        return {
            "num_objects": len(self.resolver.visible_object_ids(self.state)),
            "num_facts": len(self.facts),
            "agent_trace_length": len(self.agent_trace),
            "target": None if self.state.target is None else {"type": self.state.target.type, "object": self.state.target.object},
        }

    def _finalize(self, reason: str) -> None:
        self.terminated = True
        self.trajectory.final_result = {"answer": self.final_answer, "correct": self.final_correct}
        self.trajectory.termination = {"reason": reason, "num_steps": self.step_count}
        self.trajectory.metrics = score_trajectory(self.task, self.trajectory.steps)


def _error_result(status: str, code: str, message: str):
    from .tools.base import ToolResult
    return ToolResult(status=status, errors=[{"code": code, "message": message}])
