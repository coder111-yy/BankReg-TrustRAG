from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .llm_client import LLMClient
from .query_plan import (
    CalculationTask,
    PlannerOutput,
    QueryPlan,
    RetrievalTask,
)
from .utils import canonical_table_label, normalize_text, reporting_period_details


@dataclass(frozen=True)
class PlannerOutcome:
    status: str
    plan: QueryPlan
    attempts: int = 0
    error: str | None = None
    diagnostics: tuple[str, ...] = ()


class QueryPlanner:
    """Ask the configured model for an executable, schema-validated plan."""

    def __init__(
        self,
        client: LLMClient,
        temperature: float = 0.0,
        max_tokens: int = 2500,
        timeout_seconds: float = 60.0,
    ):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max(512, int(max_tokens))
        self.timeout_seconds = max(1.0, float(timeout_seconds))

    @classmethod
    def from_settings(cls, settings: Any, client: LLMClient | None = None) -> "QueryPlanner":
        return cls(
            client or LLMClient.from_planner_settings(settings),
            temperature=float(getattr(settings, "llm_planner_temperature", 0.0)),
            max_tokens=int(getattr(settings, "llm_planner_max_tokens", 2500)),
            timeout_seconds=float(getattr(settings, "llm_planner_timeout_seconds", 60.0)),
        )

    def plan(self, question: str, conversation_context: list[dict[str, Any]] | None = None) -> PlannerOutcome:
        normalized_question = normalize_text(question)
        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": _planner_input(normalized_question, conversation_context)},
        ]
        result = self.client.structured(
            messages,
            PlannerOutput,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
            prefer_json_schema=False,
            max_attempts=2,
        )
        if result.status == "ok" and isinstance(result.value, PlannerOutput):
            plan = _expand_planner_output(normalized_question, result.value)
            return PlannerOutcome("ok", plan, result.attempts, diagnostics=result.errors)
        return PlannerOutcome(
            result.status,
            _safe_failure_plan(normalized_question, result.error or result.status),
            result.attempts,
            result.error,
            result.errors,
        )


_PLANNER_SYSTEM_PROMPT = """你是查询规划器，不回答问题、不猜数字，只返回JSON。

任务：
1. 把用户要求拆成 answer_requirements。
2. 每个独立取数目标生成一个 retrieval_task，按顺序使用 r1、r2、r3……。
3. 需要计算时生成 sum、subtract、growth_rate、divide、compare、max 或 min 操作，输出按顺序使用 calc1、calc2……。
4. 相差多少使用 subtract 且 absolute=true；增长率必须明确 old_ref 和 new_ref。
5. 每个 answer_requirement.required_outputs 必须绑定最终 retrieval 或 calc 输出。
6. 用户未明确写出单位时省略 expected_unit，禁止猜测单位。
7. 宽表取单个数值时要区分 row_label 和 column_label；例如总计行中的总体数值应同时指定合计列。

只允许五个顶层字段：user_goal、answer_requirements、retrieval_tasks、operations、requires_clarification。
retrieval_task 使用紧凑字段：id、query、expected_information，以及必要时的 indicator、institution、period、source_hint、region、statistical_scope、row_label、column_label、expected_value_type、expected_unit。省略无关字段，不要输出null。
operation 使用：type、output_id、inputs；subtract 另加 absolute，growth_rate 使用 old_ref 和 new_ref。
只返回JSON对象，不要Markdown。"""


def _planner_input(question: str, conversation_context: list[dict[str, Any]] | None) -> str:
    prior_user_messages = [
        normalize_text(item.get("content"))
        for item in (conversation_context or [])
        if item.get("role") == "user" and normalize_text(item.get("content"))
    ][-3:]
    context = "\n".join(f"- {item[:500]}" for item in prior_user_messages) or "无"
    return f"USER_QUESTION:\n{question}\n\nPRIOR_USER_CONTEXT:\n{context}"


def _safe_failure_plan(question: str, reason: str) -> QueryPlan:
    return QueryPlan.model_validate({
        "original_query": question,
        "user_goal": "安全处理当前问题",
        "answer_requirements": [{
            "id": "ar_planning",
            "question": question,
            "required_outputs": ["planning_unavailable"],
        }],
        "entities": {},
        "retrieval_tasks": [],
        "operations": [],
        "requires_multiple_sources": False,
        "requires_table_retrieval": False,
        "requires_calculation": False,
        "requires_clarification": True,
        "clarification_reason": f"查询规划暂时不可用：{reason}",
    })


def _expand_planner_output(question: str, compact: PlannerOutput) -> QueryPlan:
    """Convert the validated compact model into the stable executor contract."""
    calculation_inputs = {
        ref
        for operation in compact.operations
        for ref in operation.input_refs()
    }
    retrieval_tasks: list[RetrievalTask] = []
    indicators: list[str] = []
    institutions: list[str] = []
    periods: list[str] = []
    documents: list[str] = []
    regions: list[str] = []
    units: list[str] = []
    for item in compact.retrieval_tasks:
        period = normalize_text(item.period) or None
        year, month, quarter = _period_scope(period or item.query)
        expected_type = item.expected_value_type or (
            "number" if item.id in calculation_inputs else "text"
        )
        expected_unit = _explicit_expected_unit(question, item.expected_unit)
        column_label = _planner_column_label(item)
        retrieval_tasks.append(RetrievalTask.model_validate({
            "id": item.id,
            "query": item.query,
            "expected_information": item.expected_information,
            "source_scope": {
                "document_title": item.source_hint,
                "year": year,
                "month": month,
                "quarter": quarter,
            },
            "semantic_constraints": {
                "indicator": item.indicator,
                "institution": item.institution,
                "region": item.region,
                "period": period,
                "statistical_scope": item.statistical_scope,
                "row_label": item.row_label,
                "column_label": column_label,
            },
            "expected_value_type": expected_type,
            "expected_unit": expected_unit,
            "dependencies": [],
        }))
        _append_unique(indicators, item.indicator)
        _append_unique(institutions, item.institution)
        _append_unique(periods, period)
        _append_unique(documents, item.source_hint)
        _append_unique(regions, item.region)
        _append_unique(units, expected_unit)

    operations: list[CalculationTask] = []
    prior_outputs: set[str] = set()
    for index, item in enumerate(compact.operations, 1):
        inputs = list(item.inputs)
        if item.type == "subtract" and item.absolute is True and len(inputs) == 2:
            # Absolute difference is commutative. Canonicalize a derived
            # calculation before a raw retrieval so traces are stable across
            # equivalent model outputs: abs(calc1 - r3), not abs(r3 - calc1).
            inputs = [
                ref
                for _, ref in sorted(
                    enumerate(inputs),
                    key=lambda pair: (pair[1] not in prior_outputs, pair[0]),
                )
            ]
        operation = CalculationTask.model_validate({
            "id": f"op{index}",
            "type": item.type,
            "output_id": item.output_id,
            "inputs": inputs,
            "left": item.left,
            "right": item.right,
            "old_ref": item.old_ref,
            "new_ref": item.new_ref,
            "parameters": ({"absolute": item.absolute} if item.absolute is not None else {}),
        })
        operations.append(operation)
        prior_outputs.add(operation.output_id)
    return QueryPlan.model_validate({
        "original_query": question,
        "user_goal": compact.user_goal,
        "answer_requirements": [item.model_dump() for item in compact.answer_requirements],
        "entities": {
            "indicators": indicators,
            "institutions": institutions,
            "periods": periods,
            "documents": documents,
            "regions": regions,
            "units": units,
        },
        "retrieval_tasks": [item.model_dump() for item in retrieval_tasks],
        "operations": [item.model_dump() for item in operations],
        "requires_multiple_sources": len(retrieval_tasks) > 1,
        "requires_table_retrieval": any(
            item.expected_value_type in {"number", "table_cell"}
            or item.semantic_constraints.indicator
            or item.semantic_constraints.row_label
            for item in retrieval_tasks
        ),
        "requires_calculation": bool(operations),
        "requires_clarification": compact.requires_clarification,
        "clarification_reason": (
            "问题缺少完成查询所需的必要维度"
            if compact.requires_clarification else None
        ),
    })


def _period_scope(value: str) -> tuple[int | None, int | None, int | None]:
    _, normalized, _ = reporting_period_details(value)
    normalized = normalized or normalize_text(value)
    month = re.fullmatch(r"(20\d{2})-(0?[1-9]|1[0-2])", normalized)
    if month:
        return int(month.group(1)), int(month.group(2)), None
    quarter = re.fullmatch(r"(20\d{2})-Q([1-4])", normalized, re.IGNORECASE)
    if quarter:
        return int(quarter.group(1)), None, int(quarter.group(2))
    year = re.fullmatch(r"(20\d{2})", normalized)
    return (int(year.group(1)), None, None) if year else (None, None, None)


_EXPLICIT_UNIT_RE = re.compile(r"万亿元|亿元|百万元|万元|万件|百分比|[%％‰]|元|件")


def _explicit_expected_unit(question: str, suggested: str | None) -> str | None:
    """Keep a unit only when it is grounded in the user's own question.

    Planner models may know common reporting units and still guess the wrong
    one for a particular workbook.  Retrieval evidence is authoritative, so
    an omitted user unit must remain unspecified until the cell is selected.
    """
    explicit = list(dict.fromkeys(
        _canonical_unit(match.group(0))
        for match in _EXPLICIT_UNIT_RE.finditer(normalize_text(question))
    ))
    if not explicit:
        return None
    normalized_suggestion = _canonical_unit(suggested)
    if normalized_suggestion and normalized_suggestion in explicit:
        return normalized_suggestion
    return explicit[0] if len(explicit) == 1 else None


def _canonical_unit(value: str | None) -> str:
    normalized = normalize_text(value)
    if normalized in {"百分比", "％"}:
        return "%"
    return normalized


def _planner_column_label(item: Any) -> str | None:
    """Complete an aggregate column only when the plan already proves it.

    A metric-by-dimension workbook can have a row named ``全国合计`` and
    component columns such as property/life/health insurance.  When the
    requested indicator is the metric named by the source itself, the scalar
    aggregate is the ``合计`` column.  Component queries keep their explicit
    column and are never rewritten here.
    """
    explicit = normalize_text(item.column_label) or None
    row = canonical_table_label(item.row_label)
    indicator = canonical_table_label(item.indicator)
    source = canonical_table_label(item.source_hint)
    aggregate_row = row == "合计" or row.endswith("合计")
    source_names_metric = bool(indicator and source and indicator in source)
    # A model sometimes repeats the requested metric in column_label even
    # though the metric names the whole workbook.  That is not a real column
    # dimension; the aggregate row's scalar total is in the 合计 column.
    explicit_repeats_metric = bool(
        explicit and indicator and canonical_table_label(explicit) == indicator
    )
    if aggregate_row and source_names_metric:
        return "合计" if not explicit or explicit_repeats_metric else explicit
    return explicit


def _append_unique(target: list[str], value: str | None) -> None:
    normalized = normalize_text(value)
    if normalized and normalized not in target:
        target.append(normalized)
