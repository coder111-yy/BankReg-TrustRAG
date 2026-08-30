from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Set

from .answer_generator import GeneratedAnswer
from .query_plan import CalculationResult, QueryPlan, RetrievalResult


@dataclass(frozen=True)
class CompletenessResult:
    complete: bool
    missing_outputs: tuple[str, ...] = ()
    missing_requirement_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


class CompletenessChecker:
    """Check user-level answer coverage without making task IDs a refusal gate."""

    def check_outputs(
        self,
        plan: QueryPlan,
        retrieval_results: Mapping[str, RetrievalResult],
        calculation_results: Mapping[str, CalculationResult],
        resolved_outputs: Mapping[str, RetrievalResult | CalculationResult] | None = None,
    ) -> CompletenessResult:
        """Legacy diagnostic only.

        The adaptive agent no longer uses this as a hard gate: a later search
        may satisfy an answer requirement using a dynamically-created task ID.
        """
        available = (
            set(resolved_outputs)
            if resolved_outputs is not None
            else {
                task_id
                for task_id, result in retrieval_results.items()
                if is_resolved_retrieval_output(result)
            } | set(calculation_results)
        )
        missing_outputs = tuple(dict.fromkeys(
            output
            for requirement in plan.answer_requirements
            for output in requirement.required_outputs
            if output and output not in available
        ))
        missing_requirements = tuple(
            requirement.id
            for requirement in plan.answer_requirements
            if requirement.required_outputs
            and any(output not in available for output in requirement.required_outputs)
        )
        reasons = tuple("初始计划中的预期输出尚未出现；允许智能体继续检索" for _ in missing_outputs)
        return CompletenessResult(not missing_outputs, missing_outputs, missing_requirements, reasons)

    def check_answer(
        self,
        plan: QueryPlan,
        generated: GeneratedAnswer,
        *,
        available_refs: Set[str] | None = None,
        strict_required_outputs: bool = False,
    ) -> CompletenessResult:
        """Check semantic requirement coverage.

        In adaptive mode, ``required_outputs`` from the initial plan are hints,
        not immutable bindings. A requirement is complete when the Answer Agent
        explicitly marks it answered and grounds it in at least one currently
        available retrieval/calculation reference.
        """
        answered = set(generated.answered_requirement_ids)
        missing_requirements: list[str] = []
        missing_outputs: list[str] = []
        reasons: list[str] = []

        for requirement in plan.answer_requirements:
            used = set(generated.output_refs_by_requirement.get(requirement.id, []))
            if requirement.id not in answered:
                missing_requirements.append(requirement.id)
                reasons.append(f"要求 {requirement.id} 尚未回答")
                continue

            if available_refs is not None:
                valid_used = used & available_refs
                if not valid_used:
                    missing_requirements.append(requirement.id)
                    reasons.append(f"要求 {requirement.id} 没有绑定当前可用证据或计算结果")
                unknown = used - available_refs
                if unknown:
                    missing_outputs.extend(sorted(unknown))
                    reasons.append(f"要求 {requirement.id} 引用了不存在的动态结果")

            if strict_required_outputs:
                absent = [output for output in requirement.required_outputs if output not in used]
                if absent:
                    missing_outputs.extend(absent)
                    if requirement.id not in missing_requirements:
                        missing_requirements.append(requirement.id)
                    reasons.append(f"要求 {requirement.id} 未使用初始计划输出")

        return CompletenessResult(
            not missing_requirements and bool(generated.answer.strip()),
            tuple(dict.fromkeys(missing_outputs)),
            tuple(dict.fromkeys(missing_requirements)),
            tuple(reasons),
        )


def is_resolved_retrieval_output(result: RetrievalResult) -> bool:
    """Treat any evidence-bearing resolved retrieval as a valid output."""
    if result.status != "resolved":
        return False
    evidence_ids = [
        *result.evidence_ids,
        *(result.selected.evidence_ids if result.selected is not None else []),
        *(
            evidence_id
            for candidate in result.candidates
            for evidence_id in candidate.evidence_ids
        ),
    ]
    return bool(list(dict.fromkeys(str(item) for item in evidence_ids if item)))
