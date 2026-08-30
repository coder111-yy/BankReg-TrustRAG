from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .answer_generator import AnswerGenerationOutcome, AnswerGenerator
from .calculator import CalculationError, Calculator
from .completeness import CompletenessChecker, CompletenessResult, is_resolved_retrieval_output
from .query_plan import CalculationResult, CalculationTask, QueryPlan, RetrievalResult, RetrievalTask
from .query_planner import (
    PlannerOutcome,
    QueryPlanner,
    calculation_task_from_decision,
    retrieval_task_from_decision,
)
from .retrieval.index import Hit
from .retrieval_tools import RetrievalExecution, RetrievalTools
from .utils import normalize_text


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
    dynamic_retrieval_tasks: list[RetrievalTask] = field(default_factory=list)
    dynamic_calculation_tasks: list[CalculationTask] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    unresolved_requirements: list[str] = field(default_factory=list)
    clarification: dict[str, Any] | None = None
    refusal_reason: str | None = None
    execution_error: dict[str, Any] | None = None
    answer_outcome: AnswerGenerationOutcome | None = None
    completeness: CompletenessResult | None = None
    final_answer: str | None = None
    latency: dict[str, int] = field(default_factory=dict)
    termination_reason: str | None = None

    def trace(self) -> dict[str, Any]:
        plan = self.query_plan
        return {
            "plan": plan.model_dump() if plan else None,
            "retrieval_tasks": [item.model_dump() for item in (plan.retrieval_tasks if plan else [])],
            "dynamic_retrieval_tasks": [item.model_dump() for item in self.dynamic_retrieval_tasks],
            "retrieval_results": [item.model_dump() for item in self.retrieval_results.values()],
            "calculation_tasks": [item.model_dump() for item in (plan.operations if plan else [])],
            "dynamic_calculation_tasks": [item.model_dump() for item in self.dynamic_calculation_tasks],
            "calculation_results": [item.model_dump() for item in self.calculation_results.values()],
            "tool_history": list(self.tool_history),
            "iterations": self.iterations,
            "clarification": self.clarification,
            "refusal_reason": self.refusal_reason,
            "execution_error": self.execution_error,
            "termination_reason": self.termination_reason,
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
    """Adaptive Plan -> Act -> Observe -> Replan agent loop.

    The initial QueryPlan is only a starting hypothesis. Real retrieval results
    are fed back to the planner, which chooses one next action at a time. A
    failed first search therefore causes another search strategy rather than an
    immediate "insufficient evidence" refusal.
    """

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
        self.max_steps = max(2, min(int(max_steps), 12))

    def run(
        self,
        question: str,
        conversation_context: list[dict[str, Any]] | None = None,
        observer: Observer | None = None,
    ) -> AgentState:
        state = AgentState(question)
        planning_started = time.perf_counter()
        _report(observer, "planning", label="正在理解问题并形成初始检索计划")
        planner_outcome: PlannerOutcome = self.planner.plan(question, conversation_context)
        state.latency["planning_ms"] = _elapsed(planning_started)
        state.query_plan = planner_outcome.plan
        state.planner_status = planner_outcome.status
        state.planner_error = planner_outcome.error
        state.planner_diagnostics = planner_outcome.diagnostics
        plan = planner_outcome.plan

        if planner_outcome.status != "ok":
            # Initial structured planning is not a single point of failure.
            # Continue from the recovery seed plan and let next_action choose
            # search/calculate/clarify based on observations.
            state.planner_status = "degraded"
            state.tool_history.append({
                "step": 0,
                "source": "planner",
                "action": "recover",
                "status": planner_outcome.status,
                "summary": "初始查询规划失败，切换为自适应检索恢复模式",
                "error": planner_outcome.error,
            })
            _report(observer, "planner_recovery", label="初始规划未成功，正在切换自适应检索继续处理")
        elif plan.requires_clarification:
            state.clarification = {
                "stage": "planning",
                "reason": plan.clarification_reason or "问题缺少完成任务所需的信息",
                "question": plan.clarification_reason or "请补充需要比较的具体选项或范围。",
            }
            state.termination_reason = "clarification_required"
            return state

        # Execute the initial plan as seed actions. Unlike the old executor,
        # failure here is only an observation; it never directly causes refusal.
        retrieval_started = time.perf_counter()
        _report(
            observer,
            "tasks_planned",
            label=f"已形成初始计划：{len(plan.retrieval_tasks)}个检索任务、{len(plan.operations)}个计算任务",
        )
        seen_hits: dict[str, Hit] = {}
        for task in plan.retrieval_tasks:
            self._execute_retrieval(state, task, seen_hits, observer, source="initial")
        state.hits = list(seen_hits.values())

        # Generic recovery: if an initial task returned no evidence, retry that
        # same information need once with broad retrieval before asking the LLM
        # to invent a new search. This is especially important for multi-file
        # questions: the missing second source is retried directly instead of
        # repeatedly searching an already-resolved first source.
        for task in plan.retrieval_tasks:
            current = state.retrieval_results.get(task.id)
            if current is None or current.status != "not_found":
                continue
            broad_task = task.model_copy(update={"search_mode": "broad"})
            _report(
                observer,
                "agent_replan",
                label=f"初次未找到，正在扩大检索：{task.expected_information}",
            )
            self._execute_retrieval(
                state,
                broad_task,
                seen_hits,
                observer,
                source="initial_broad_retry",
            )

        state.hits = list(seen_hits.values())
        state.latency["retrieval_ms"] = _elapsed(retrieval_started)

        calculation_started = time.perf_counter()
        self._execute_ready_calculations(state, plan.operations, observer, source="initial")
        state.latency["calculation_ms"] = _elapsed(calculation_started)

        initial_completeness = self.completeness_checker.check_outputs(
            plan,
            state.retrieval_results,
            state.calculation_results,
            state.resolved_outputs,
        )
        if initial_completeness.complete and self._generate_answer(
            state, plan, question, observer
        ):
            state.termination_reason = "answered_from_initial_plan"
            return state

        loop_started = time.perf_counter()
        seen_action_signatures: dict[str, int] = {
            _task_signature(task): 1
            for task in plan.retrieval_tasks
        }
        dynamic_search_index = 0
        dynamic_calc_index = 0

        for step in range(1, self.max_steps + 1):
            state.iterations = step
            _report(observer, "agent_observe", label="正在判断当前证据是否足以回答")
            decision_outcome = self.planner.next_action(
                question,
                plan,
                state.retrieval_results,
                state.calculation_results,
                state.tool_history,
                conversation_context,
            )
            if decision_outcome.status != "ok" or decision_outcome.decision is None:
                # If the decision call fails but we already have evidence, try a
                # grounded answer once. This avoids converting a transient
                # planner-format failure into a false evidence refusal.
                if state.hits and self._generate_answer(state, plan, question, observer):
                    state.termination_reason = "answer_after_step_failure"
                    break
                state.execution_error = {
                    "stage": "adaptive_planning",
                    "reason": decision_outcome.error or "智能体下一步决策失败",
                }
                state.termination_reason = "adaptive_planner_failed"
                break

            decision = decision_outcome.decision
            signature = _decision_signature(decision)
            seen_action_signatures[signature] = seen_action_signatures.get(signature, 0) + 1
            if seen_action_signatures[signature] > 1 and decision.action in {"search", "calculate"}:
                state.tool_history.append({
                    "step": step,
                    "action": decision.action,
                    "status": "skipped_duplicate",
                    "summary": "检测到重复动作，要求下一轮更换检索/计算策略",
                    "signature": signature,
                })
                _report(observer, "agent_replan", label="检测到重复动作，正在更换检索策略")
                continue

            if decision.action == "search":
                dynamic_search_index += 1
                task_id = f"agent_r{dynamic_search_index}"
                task = retrieval_task_from_decision(
                    question,
                    decision,
                    task_id,
                    fallback_source_hint=_single_document_hint(plan),
                )
                state.dynamic_retrieval_tasks.append(task)
                _report(observer, "agent_replan", label=decision.summary)
                self._execute_retrieval(state, task, seen_hits, observer, source="adaptive", step=step)
                state.hits = list(seen_hits.values())
                # A new retrieval may make an initial calculation executable.
                self._execute_ready_calculations(state, plan.operations, observer, source="initial_retry")
                continue

            if decision.action == "calculate":
                dynamic_calc_index += 1
                task = calculation_task_from_decision(decision, f"agent_op{dynamic_calc_index}")
                state.dynamic_calculation_tasks.append(task)
                _report(observer, "calculating", label=decision.summary)
                try:
                    result = self.calculator.execute(
                        task,
                        state.retrieval_results,
                        state.calculation_results,
                    )
                except CalculationError as exc:
                    state.tool_history.append({
                        "step": step,
                        "action": "calculate",
                        "status": "failed",
                        "summary": decision.summary,
                        "error": str(exc),
                        "input_refs": task.input_refs(),
                    })
                    continue
                state.calculation_results[result.id] = result
                state.resolved_outputs[result.id] = result
                state.tool_history.append({
                    "step": step,
                    "action": "calculate",
                    "status": "resolved",
                    "summary": decision.summary,
                    "result_id": result.id,
                    "result": result.result,
                    "unit": result.unit,
                    "trace": result.trace,
                })
                continue

            if decision.action == "answer":
                if self._generate_answer(state, plan, question, observer):
                    state.termination_reason = "answered"
                    break
                state.tool_history.append({
                    "step": step,
                    "action": "answer",
                    "status": "incomplete",
                    "summary": "回答草稿未覆盖全部用户要求，继续补充证据或重新组织回答",
                    "missing_requirement_ids": list(state.completeness.missing_requirement_ids) if state.completeness else [],
                })
                continue

            if decision.action == "clarify":
                state.clarification = {
                    "stage": "adaptive_agent",
                    "reason": decision.summary,
                    "question": decision.clarification_question,
                }
                state.termination_reason = "clarification_required"
                break

            if decision.action == "stop":
                state.refusal_reason = decision.summary or "多轮检索后仍未找到足以支持答案的证据"
                state.termination_reason = "insufficient_evidence"
                break

        state.latency["agent_loop_ms"] = _elapsed(loop_started)
        if state.final_answer is None and state.clarification is None and state.execution_error is None and state.refusal_reason is None:
            state.refusal_reason = "已尝试多轮不同检索策略，但当前资料仍不足以支持可靠回答"
            state.termination_reason = "max_steps_insufficient_evidence"
        return state

    def _execute_retrieval(
        self,
        state: AgentState,
        task: RetrievalTask,
        seen_hits: dict[str, Hit],
        observer: Observer | None,
        *,
        source: str,
        step: int | None = None,
    ) -> None:
        _report(observer, "retrieving_task", label=f"正在检索：{task.expected_information}")
        started = time.perf_counter()
        execution: RetrievalExecution = self.retrieval_tools.execute(task)
        result = _bind_retrieval_result(task.id, execution.result, execution.hits)
        state.retrieval_results[task.id] = result
        if is_resolved_retrieval_output(result):
            state.resolved_outputs[task.id] = result
        for hit in execution.hits:
            seen_hits[hit.evidence_id] = hit
        state.tool_history.append({
            "step": step,
            "source": source,
            "action": "search",
            "task_id": task.id,
            "query": task.query,
            "expected_information": task.expected_information,
            "search_mode": task.search_mode,
            "status": result.status,
            "ambiguity_reason": result.ambiguity_reason,
            "evidence_count": len(result.evidence_ids),
            "selected": _selected_summary(result),
            "diagnostics": execution.diagnostics,
            "latency_ms": _elapsed(started),
        })

    def _execute_ready_calculations(
        self,
        state: AgentState,
        operations: list[CalculationTask],
        observer: Observer | None,
        *,
        source: str,
    ) -> None:
        pending = [op for op in operations if op.output_id not in state.calculation_results]
        progressed = True
        while pending and progressed:
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
                state.tool_history.append({
                    "source": source,
                    "action": "calculate",
                    "status": "resolved",
                    "operation_id": operation.id,
                    "result_id": result.id,
                    "result": result.result,
                    "unit": result.unit,
                    "trace": result.trace,
                })
                pending.remove(operation)
                progressed = True
                _report(observer, "calculating", label=f"已完成计算：{result.trace}")

    def _generate_answer(
        self,
        state: AgentState,
        plan: QueryPlan,
        question: str,
        observer: Observer | None,
    ) -> bool:
        generation_started = time.perf_counter()
        available_refs = set(state.retrieval_results) | set(state.calculation_results)
        missing_requirement_ids: list[str] = []
        for _ in range(self.max_answer_attempts):
            _report(observer, "generating", label="正在根据当前证据生成回答")
            outcome = self.answer_generator.generate(
                question,
                plan,
                state.retrieval_results,
                state.calculation_results,
                missing_requirement_ids=missing_requirement_ids,
            )
            state.answer_outcome = outcome
            if outcome.status != "ok":
                state.completeness = CompletenessResult(
                    False,
                    missing_requirement_ids=tuple(item.id for item in plan.answer_requirements),
                    reasons=("回答模型暂时不可用",),
                )
                break
            answer_check = self.completeness_checker.check_answer(
                plan,
                outcome.generated,
                available_refs=available_refs,
                strict_required_outputs=False,
            )
            state.completeness = answer_check
            if answer_check.complete:
                state.final_answer = outcome.generated.answer
                state.latency["generation_ms"] = state.latency.get("generation_ms", 0) + _elapsed(generation_started)
                return True
            missing_requirement_ids = list(answer_check.missing_requirement_ids)
        state.latency["generation_ms"] = state.latency.get("generation_ms", 0) + _elapsed(generation_started)
        state.unresolved_requirements = missing_requirement_ids
        return False



def _single_document_hint(plan: QueryPlan) -> str | None:
    documents = [item for item in plan.entities.documents if item]
    return documents[0] if len(documents) == 1 else None


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


def _decision_signature(decision: Any) -> str:
    if decision.action == "search":
        return "|".join([
            "search",
            normalize_text(decision.query),
            normalize_text(decision.source_hint),
            normalize_text(decision.indicator),
            normalize_text(decision.period),
            normalize_text(decision.row_label),
            normalize_text(decision.column_label),
            str(decision.search_mode),
            str(getattr(decision, "selection", "single")),
            ",".join(sorted(
                normalize_text(value)
                for value in (getattr(decision, "exclude_row_labels", []) or [])
                if normalize_text(value)
            )),
        ])
    if decision.action == "calculate":
        return "|".join([
            "calculate",
            str(decision.operation_type),
            str(decision.output_id),
            ",".join(decision.calculation_input_refs()),
        ])
    return f"{decision.action}|{normalize_text(decision.summary)}"


def _task_signature(task: RetrievalTask) -> str:
    semantic = task.semantic_constraints
    return "|".join([
        "search",
        normalize_text(task.query),
        normalize_text(task.source_scope.document_title),
        normalize_text(semantic.indicator),
        normalize_text(semantic.period),
        normalize_text(semantic.row_label),
        normalize_text(semantic.column_label),
        str(task.search_mode),
        str(getattr(task, "selection", "single")),
        ",".join(sorted(
            normalize_text(value)
            for value in (getattr(task, "exclude_row_labels", []) or [])
            if normalize_text(value)
        )),
    ])


def _selected_summary(result: RetrievalResult) -> dict[str, Any] | None:
    selected = result.selected
    if selected is None:
        return None
    value = selected.value
    if isinstance(value, str) and len(value) > 500:
        value = value[:500] + "…"
    return {
        "value": value,
        "unit": selected.unit,
        "document_title": selected.document_title,
        "sheet_name": selected.sheet_name,
        "cell_address": selected.cell_address,
        "period": selected.period,
        "evidence_ids": list(selected.evidence_ids),
    }
