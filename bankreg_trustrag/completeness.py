from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .answer_generator import GeneratedAnswer
from .query_plan import CalculationResult, QueryPlan, RetrievalResult


@dataclass(frozen=True)
class CompletenessResult:
    complete: bool
    missing_outputs: tuple[str, ...] = ()
    missing_requirement_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


class CompletenessChecker:
    """Deterministically bind requirements to executable and rendered outputs."""

    def check_outputs(
        self,
        plan: QueryPlan,
        retrieval_results: Mapping[str, RetrievalResult],
        calculation_results: Mapping[str, CalculationResult],
        resolved_outputs: Mapping[str, RetrievalResult | CalculationResult] | None = None,
    ) -> CompletenessResult:
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
            if output not in available
        ))
        missing_requirements = tuple(
            requirement.id
            for requirement in plan.answer_requirements
            if any(output not in available for output in requirement.required_outputs)
        )
        # Internal output identifiers stay in diagnostics, not user-facing
        # clarification text.
        reasons = tuple("执行结果未正确绑定到回答要求" for _ in missing_outputs)
        return CompletenessResult(not missing_outputs, missing_outputs, missing_requirements, reasons)

    def check_answer(self, plan: QueryPlan, generated: GeneratedAnswer) -> CompletenessResult:
        answered = set(generated.answered_requirement_ids)
        missing_requirements: list[str] = []
        missing_outputs: list[str] = []
        reasons: list[str] = []
        for requirement in plan.answer_requirements:
            used = set(generated.output_refs_by_requirement.get(requirement.id, []))
            if requirement.id not in answered:
                missing_requirements.append(requirement.id)
                reasons.append(f"要求 {requirement.id} 未标记为已回答")
            absent = [output for output in requirement.required_outputs if output not in used]
            if absent:
                missing_outputs.extend(absent)
                if requirement.id not in missing_requirements:
                    missing_requirements.append(requirement.id)
                reasons.append(f"要求 {requirement.id} 未使用输出 {', '.join(absent)}")
        return CompletenessResult(
            not missing_requirements and bool(generated.answer.strip()),
            tuple(dict.fromkeys(missing_outputs)),
            tuple(dict.fromkeys(missing_requirements)),
            tuple(reasons),
        )


def is_resolved_retrieval_output(result: RetrievalResult) -> bool:
    """Treat an evidence bundle as a first-class RetrievalTask output.

    Text/PDF/Word facts do not need a scalar ``value``.  A resolved task with
    at least one evidence ID is sufficient even when there is no selected
    table cell.  Candidate-level IDs are accepted for compatible fixtures.
    """
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
