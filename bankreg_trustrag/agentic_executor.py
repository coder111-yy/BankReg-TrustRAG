from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .answer_generator import AnswerGenerationOutcome, AnswerGenerator
from .calculator import CalculationError, Calculator
from .completeness import (
    CompletenessChecker,
    CompletenessResult,
    is_resolved_retrieval_output,
)
from .query_plan import CalculationResult, QueryPlan, RetrievalResult
from .query_planner import PlannerOutcome, QueryPlanner
from .retrieval.index import Hit
from .retrieval_tools import RetrievalTools


Observer = Callable[[str, dict[str, Any]], None]


@dataclass
class AgentState:
    original_query: str
    query_plan: QueryPlan | None = None
    planner_status: str = "not_started"
    planner_error: str | None = None
    planner_diagnostics: tuple[str, ...] = ()
    retrieval_results: dict[str, RetrievalResult] = field(default_factory=dict)
    calculation_results: dict[str, CalculationResult] = field(default_factory=dict)
    resolved_outputs: dict[str, RetrievalResult | CalculationResult] = field(default_factory=dict)
    hits: list[Hit] = field(default_factory=list)
    unresolved_requirements: list[str] = field(default_factory=list)
    clarification: dict[str, Any] | None = None
    execution_error: dict[str, Any] | None = None
    answer_outcome: AnswerGenerationOutcome | None = None
    completeness: CompletenessResult | None = None
    final_answer: str | None = None
    latency: dict[str, int] = field(default_factory=dict)

    def trace(self) -> dict[str, Any]:
        plan = self.query_plan
        return {
            "plan": plan.model_dump() if plan else None,
            "retrieval_tasks": [item.model_dump() for item in (plan.retrieval_tasks if plan else [])],
            "retrieval_results": [item.model_dump() for item in self.retrieval_results.values()],
            "calculation_results": [item.model_dump() for item in self.calculation_results.values()],
            "clarification": self.clarification,
            "execution_error": self.execution_error,
            "resolved_output_count": len(self.resolved_outputs),
            "answer_requirements": [item.model_dump() for item in (plan.answer_requirements if plan else [])],
            "answered_requirements": (
                self.answer_outcome.generated.answered_requirement_ids if self.answer_outcome else []
            ),
            "completeness": {
                "complete": self.completeness.complete,
                "missing_outputs": list(self.completeness.missing_outputs),
                "missing_requirement_ids": list(self.completeness.missing_requirement_ids),
                "reasons": list(self.completeness.reasons),
            } if self.completeness else None,
            "latency": dict(self.latency),
            "planner_status": self.planner_status,
            "planner_error": self.planner_error,
            "planner_diagnostics": list(self.planner_diagnostics),
            "generation_status": self.answer_outcome.status if self.answer_outcome else None,
            "answer_generation": {
                "status": self.answer_outcome.status,
                "attempts": self.answer_outcome.attempts,
                "error": self.answer_outcome.error,
                "raw_answer": self.answer_outcome.generated.answer,
                "output_refs_by_requirement": self.answer_outcome.generated.output_refs_by_requirement,
            } if self.answer_outcome else None,
        }


class BoundedAgentExecutor:
    """Explicit state machine with bounded answer regeneration."""

    def __init__(
        self,
        planner: QueryPlanner,
        retrieval_tools: RetrievalTools,
        calculator: Calculator,
        answer_generator: AnswerGenerator,
        completeness_checker: CompletenessChecker | None = None,
        *,
        max_answer_attempts: int = 2,
        max_steps: int = 8,
    ):
        self.planner = planner
        self.retrieval_tools = retrieval_tools
        self.calculator = calculator
        self.answer_generator = answer_generator
        self.completeness_checker = completeness_checker or CompletenessChecker()
        self.max_answer_attempts = max(1, min(int(max_answer_attempts), 3))
        # Keep compatibility with TrustRAGService, which passes agentic_max_steps.
        # The current executor remains bounded by its existing state-machine logic;
        # storing this value also preserves the public constructor contract.
        self.max_steps = max(2, min(int(max_steps), 12))

    def run(
        self,
        question: str,
        conversation_context: list[dict[str, Any]] | None = None,
        observer: Observer | None = None,
    ) -> AgentState:
        state = AgentState(question)
        planning_started = time.perf_counter()
        _report(observer, "planning", label="正在理解问题并拆解待回答任务")
        planner_outcome: PlannerOutcome = self.planner.plan(question, conversation_context)
        state.latency["planning_ms"] = _elapsed(planning_started)
        state.query_plan = planner_outcome.plan
        state.planner_status = planner_outcome.status
        state.planner_error = planner_outcome.error
        state.planner_diagnostics = planner_outcome.diagnostics
        plan = planner_outcome.plan
        if planner_outcome.status != "ok" or plan.requires_clarification:
            state.clarification = {
                "stage": "planning",
                "reason": plan.clarification_reason or planner_outcome.error or "问题信息不足",
            }
            state.unresolved_requirements = [item.id for item in plan.answer_requirements]
            return state

        retrieval_started = time.perf_counter()
        _report(observer, "tasks_planned", label=f"识别到{len(plan.answer_requirements)}个待回答问题、{len(plan.retrieval_tasks)}个检索任务")
        seen_hits: dict[str, Hit] = {}
        pending_retrievals = list(plan.retrieval_tasks)
        while pending_retrievals:
            progressed = False
            for task in list(pending_retrievals):
                if any(ref not in state.retrieval_results for ref in task.dependencies):
                    continue
                blocked = [
                    ref for ref in task.dependencies
                    if state.retrieval_results[ref].status != "resolved"
                ]
                if blocked:
                    result = RetrievalResult(
                        task_id=task.id,
                        status="blocked",
                        expected_information=task.expected_information,
                        ambiguity_reason=f"前置任务未完成：{', '.join(blocked)}",
                    )
                    state.retrieval_results[task.id] = result
                else:
                    _report(observer, "retrieving_task", label=f"正在检索：{task.expected_information}")
                    execution = self.retrieval_tools.execute(task)
                    if _should_retry_as_text_evidence(plan, task.id, execution.result):
                        # A threshold or enumerated rule may be described in a
                        # Word/PDF paragraph even when the task's expected
                        # answer is numeric. Retry once as text evidence, keep
                        # the original QueryPlan ID, and never substitute
                        # evidence obtained for another task.
                        execution = self.retrieval_tools.execute(_text_evidence_task(task))
                    result = _bind_retrieval_result(task.id, execution.result, execution.hits)
                    state.retrieval_results[task.id] = result
                    if is_resolved_retrieval_output(result):
                        state.resolved_outputs[task.id] = result
                    for hit in execution.hits:
                        seen_hits[hit.evidence_id] = hit
                pending_retrievals.remove(task)
                progressed = True
            if not progressed:
                # QueryPlan validation normally makes this unreachable. Keep a
                # bounded executor-side guard for plans built outside Pydantic.
                for task in pending_retrievals:
                    state.retrieval_results[task.id] = RetrievalResult(
                        task_id=task.id,
                        status="blocked",
                        expected_information=task.expected_information,
                        ambiguity_reason="检索任务依赖无法解析",
                    )
                break
        state.hits = list(seen_hits.values())
        state.latency["retrieval_ms"] = _elapsed(retrieval_started)
        _report(observer, "retrieval_complete", label=f"已完成{len(plan.retrieval_tasks)}个检索任务并取得{len(state.hits)}条证据")

        ambiguous = [item for item in state.retrieval_results.values() if item.status == "ambiguous"]
        if ambiguous:
            state.clarification = {
                "stage": "retrieval",
                "reason": "；".join(item.ambiguity_reason or item.expected_information for item in ambiguous),
                "task_ids": [item.task_id for item in ambiguous],
            }
            state.unresolved_requirements = _requirements_using(plan, {item.task_id for item in ambiguous})
            return state

        calculation_started = time.perf_counter()
        if plan.operations:
            _report(observer, "calculating", label=f"正在执行{len(plan.operations)}项确定性计算")
        pending = list(plan.operations)
        while pending:
            progressed = False
            for operation in list(pending):
                try:
                    result = self.calculator.execute(
                        operation,
                        state.retrieval_results,
                        state.calculation_results,
                    )
                except CalculationError:
                    continue
                state.calculation_results[result.id] = result
                state.resolved_outputs[result.id] = result
                pending.remove(operation)
                progressed = True
            if not progressed:
                break
        state.latency["calculation_ms"] = _elapsed(calculation_started)

        output_check = self.completeness_checker.check_outputs(
            plan,
            state.retrieval_results,
            state.calculation_results,
            state.resolved_outputs,
        )
        state.completeness = output_check
        if not output_check.complete:
            state.unresolved_requirements = list(output_check.missing_requirement_ids)
            state.execution_error = {
                "stage": "output_binding",
                "reason": "已完成执行，但部分结果未能绑定到回答要求",
                "missing_output_count": len(output_check.missing_outputs),
            }
            return state

        generation_started = time.perf_counter()
        missing_requirement_ids: list[str] = []
        for _ in range(self.max_answer_attempts):
            _report(observer, "generating", label="正在根据证据与计算结果生成完整回答")
            outcome = self.answer_generator.generate(
                question,
                plan,
                state.retrieval_results,
                state.calculation_results,
                missing_requirement_ids=missing_requirement_ids,
            )
            state.answer_outcome = outcome
            answer_check = self.completeness_checker.check_answer(plan, outcome.generated)
            state.completeness = answer_check
            if answer_check.complete:
                state.final_answer = outcome.generated.answer
                break
            missing_requirement_ids = list(answer_check.missing_requirement_ids)
        state.latency["generation_ms"] = _elapsed(generation_started)
        if state.final_answer is None:
            state.unresolved_requirements = missing_requirement_ids
            state.execution_error = {
                "stage": "answer_completeness",
                "reason": "最终回答未覆盖全部用户要求",
                "missing_requirement_count": len(missing_requirement_ids),
            }
        return state


def _requirements_using(plan: QueryPlan, missing_refs: set[str]) -> list[str]:
    affected = {
        operation.output_id
        for operation in plan.operations
        if missing_refs.intersection(operation.input_refs())
    } | missing_refs
    changed = True
    while changed:
        changed = False
        for operation in plan.operations:
            if affected.intersection(operation.input_refs()) and operation.output_id not in affected:
                affected.add(operation.output_id)
                changed = True
    return [
        requirement.id
        for requirement in plan.answer_requirements
        if affected.intersection(requirement.required_outputs)
    ]


def _report(observer: Observer | None, stage: str, **details: Any) -> None:
    if observer is not None:
        observer(stage, details)


def _elapsed(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _bind_retrieval_result(
    task_id: str,
    result: RetrievalResult,
    hits: list[Hit],
) -> RetrievalResult:
    """Normalize tool output into the executor's unified output registry."""
    evidence_ids = list(dict.fromkeys([
        *result.evidence_ids,
        *(hit.evidence_id for hit in hits),
    ]))
    updates: dict[str, Any] = {}
    if result.task_id != task_id:
        updates["task_id"] = task_id
    if result.status == "resolved" and evidence_ids != result.evidence_ids:
        updates["evidence_ids"] = evidence_ids
    return result.model_copy(update=updates) if updates else result


def _should_retry_as_text_evidence(
    plan: QueryPlan,
    task_id: str,
    result: RetrievalResult,
) -> bool:
    if result.status != "not_found" or result.evidence_ids:
        return False
    directly_required = any(
        task_id in requirement.required_outputs
        for requirement in plan.answer_requirements
    )
    calculation_input = any(
        task_id in operation.input_refs()
        for operation in plan.operations
    )
    return directly_required and not calculation_input


def _text_evidence_task(task: Any) -> Any:
    constraints = task.semantic_constraints.model_copy(update={
        "indicator": None,
        "parent_indicator": None,
        "row_label": None,
        "column_label": None,
    })
    return task.model_copy(update={
        "expected_value_type": "text",
        "expected_unit": None,
        "semantic_constraints": constraints,
    })
