from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from .llm_client import LLMClient
from .query_plan import (
    AgentStepDecision,
    CalculationResult,
    CalculationTask,
    PlannerOutput,
    QueryPlan,
    RetrievalResult,
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


@dataclass(frozen=True)
class AgentStepOutcome:
    status: str
    decision: AgentStepDecision | None
    attempts: int = 0
    error: str | None = None
    diagnostics: tuple[str, ...] = ()


class QueryPlanner:
    """LLM planner plus adaptive next-action policy.

    ``plan`` creates only an initial hypothesis. ``next_action`` observes the
    real tool results and decides whether to search again, calculate, answer,
    clarify, or stop. The first plan is never treated as an immutable DAG.
    """

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
        messages = _conversation_messages(
            _PLANNER_SYSTEM_PROMPT,
            normalized_question,
            conversation_context,
        )
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
            try:
                plan = _expand_planner_output(normalized_question, result.value, conversation_context)
            except Exception as exc:  # defensive: the adaptive loop must not crash on an imperfect initial plan
                return PlannerOutcome(
                    "invalid_plan",
                    _safe_failure_plan(normalized_question, str(exc)),
                    result.attempts,
                    str(exc),
                    result.errors,
                )
            return PlannerOutcome("ok", plan, result.attempts, diagnostics=result.errors)
        return PlannerOutcome(
            result.status,
            _safe_failure_plan(normalized_question, result.error or result.status),
            result.attempts,
            result.error,
            result.errors,
        )

    def next_action(
        self,
        question: str,
        plan: QueryPlan,
        retrieval_results: Mapping[str, RetrievalResult],
        calculation_results: Mapping[str, CalculationResult],
        tool_history: list[dict[str, Any]],
        conversation_context: list[dict[str, Any]] | None = None,
    ) -> AgentStepOutcome:
        """Choose exactly one observable next action from current evidence."""
        payload = _agent_step_input(
            question,
            plan,
            retrieval_results,
            calculation_results,
            tool_history,
            conversation_context,
        )
        result = self.client.structured(
            _conversation_messages(
                _AGENT_STEP_SYSTEM_PROMPT,
                question,
                conversation_context,
                runtime_payload=payload,
            ),
            AgentStepDecision,
            temperature=self.temperature,
            max_tokens=min(self.max_tokens, 1800),
            timeout_seconds=self.timeout_seconds,
            prefer_json_schema=False,
            max_attempts=2,
        )
        if result.status == "ok" and isinstance(result.value, AgentStepDecision):
            return AgentStepOutcome("ok", result.value, result.attempts, diagnostics=result.errors)
        return AgentStepOutcome(
            result.status,
            None,
            result.attempts,
            result.error,
            result.errors,
        )


_PLANNER_SYSTEM_PROMPT = """你是金融监管检索智能体的查询规划器。你不回答问题、不猜数字，只返回JSON。

你会收到真实的多轮聊天消息，角色保持为 user / assistant。请先理解“当前最后一条 user 消息”在整段对话中的真实意图，再生成检索计划。

对话理解原则：
1. 对话历史只用于理解指代、补充条件、澄清回答和任务延续，不是监管事实证据。
2. 如果 assistant 上一轮询问“哪个季度/年份/机构/地区”，当前 user 只回答“一季度”“2023年”“大型商业银行”等，应把该回答合并回上一轮未完成任务，而不是把它当成新问题。
3. 如果当前 user 重新完整给出新的文件、工作表、指标或任务目标，应以当前消息为准；旧约束不得机械继承。
4. 不需要依赖“呢、这个、那个”等关键词判断追问；根据完整会话语义判断。
5. 历史 assistant 的回答可以帮助理解“除了全国”“第二个”等指代，但其中事实不能直接作为证据，仍需检索。

规划原则：
6. 把用户真正要求回答的事项拆成 answer_requirements。
7. 为每个独立信息需求生成 retrieval_tasks；不确定字段请省略，不要猜。
8. 需要确定性数学运算时生成 operations。
9. 用户没有明确给出的单位、工作表名、统计口径不要猜成硬约束。
10. source_hint 可以来自当前消息，也可以来自对话中明确延续的数据源。
11. 当前消息明确给出多个《文档名》时，不同 retrieval_task 应绑定各自对应来源。
12. 对“除了X、排除X、不含X”使用 exclude_row_labels。
13. 来源/维度不确定时可使用 search_mode=broad。
14. 无选项最高/最低题，只要数据源本身定义比较集合，就主动检索并使用 selection=max/min。
15. 只有无法通过检索消除、且确实需要用户选择的语义歧义才 requires_clarification=true。
16. required_outputs 只是初始预期，后续智能体可根据真实观察调整。
17. 不输出分析过程，不输出答案。

只允许五个顶层字段：user_goal、answer_requirements、retrieval_tasks、operations、requires_clarification。
retrieval_task 可用字段：id、query、expected_information、indicator、institution、period、source_hint、region、statistical_scope、row_label、column_label、expected_value_type、expected_unit、search_mode、selection、exclude_row_labels。
selection 可取 single/max/min/all。
operation 可用字段：type、output_id、inputs；subtract 可加 absolute；growth_rate 使用 old_ref/new_ref。
只返回JSON对象，不要Markdown。"""


_AGENT_STEP_SYSTEM_PROMPT = """你是证据驱动的金融监管检索智能体。你每次只决定下一步动作，不回答用户问题，不输出思维过程。

你会收到真实的多轮 user/assistant 对话，以及当前查询计划、检索结果、计算结果和最近工具动作。

动作只能是：search / calculate / answer / clarify / stop。

规则：
1. 直接根据完整会话理解当前任务，不使用Python式关键词规则判断是否追问。
2. 历史聊天不是监管证据；最终事实必须来自 CURRENT_RESULTS 或后续检索。
3. 第一次检索失败不能直接 stop；应换表达、换来源候选、放宽为 broad 或补另一个缺失任务。
4. 优先补齐 status!=resolved 的初始任务；已 resolved 的事实不要无意义重复检索。
5. 不重复完全相同的失败搜索。
6. search 的 expected_information 要明确；不确定行列/单位时不要猜硬约束。
7. calculate 只能引用已有 task/result id。
8. answer 仅在证据覆盖全部回答目标时选择。
9. clarify 仅用于继续检索也无法消除、必须由用户决定的歧义。
10. stop 仅用于多轮不同策略后仍缺必要证据。
11. 无选项最高/最低题可以扫描数据源并 selection=max/min。
12. “除了X”使用 exclude_row_labels=[X]。
13. 当前 user 若给出完整新任务，以当前消息覆盖旧条件；若只是“一季度”“除了全国呢？”等补充，则结合前文继续原任务。
14. summary 只写一句高层动作说明，不输出Chain-of-Thought。

只返回符合JSON Schema的对象。"""


def _conversation_messages(
    system_prompt: str,
    current_question: str,
    conversation_context: list[dict[str, Any]] | None,
    *,
    runtime_payload: str | None = None,
) -> list[dict[str, str]]:
    """Build native chat messages; role=user is the HumanMessage role."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]
    messages.extend(
        _chat_history_messages(
            conversation_context,
            current_question=current_question,
        )
    )

    current = normalize_text(current_question)
    if runtime_payload is None:
        messages.append({"role": "user", "content": current})
    else:
        messages.append({
            "role": "user",
            "content": (
                f"{current}\\n\\n"
                "以下是本轮工具执行后的结构化运行状态。它是观察结果，不是新的用户问题：\\n"
                f"{runtime_payload}"
            ),
        })
    return messages


def _chat_history_messages(
    conversation_context: list[dict[str, Any]] | None,
    *,
    current_question: str,
    max_messages: int = 12,
) -> list[dict[str, str]]:
    """Preserve user/assistant roles and remove duplicate failed retries."""
    cleaned: list[dict[str, str]] = []
    current = normalize_text(current_question)

    for item in conversation_context or []:
        role = str(item.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = normalize_text(item.get("content"))
        if not content:
            continue

        # The API may already have persisted the current HumanMessage before
        # execution. Never send the same current question twice.
        if role == "user" and content == current:
            continue

        if (
            cleaned
            and role == "user"
            and cleaned[-1]["role"] == "user"
            and cleaned[-1]["content"] == content
        ):
            continue

        cleaned.append({"role": role, "content": content[:1600]})

    return cleaned[-max(2, int(max_messages)):]


def _agent_step_input(
    question: str,
    plan: QueryPlan,
    retrieval_results: Mapping[str, RetrievalResult],
    calculation_results: Mapping[str, CalculationResult],
    tool_history: list[dict[str, Any]],
    conversation_context: list[dict[str, Any]] | None,
) -> str:
    payload = {
        "INITIAL_QUERY_PLAN": {
            "user_goal": plan.user_goal,
            "answer_requirements": [
                {"id": item.id, "question": item.question}
                for item in plan.answer_requirements
            ],
            "retrieval_tasks": [
                {
                    "id": task.id,
                    "expected_information": task.expected_information,
                    "source_hint": task.source_scope.document_title,
                    "status": (
                        retrieval_results[task.id].status
                        if task.id in retrieval_results else "not_started"
                    ),
                }
                for task in plan.retrieval_tasks
            ],
        },
        "CURRENT_RESULTS": {
            "retrieval": {
                key: _compact_retrieval_result(value)
                for key, value in retrieval_results.items()
            },
            "calculations": {
                key: {
                    "id": value.id,
                    "operation": value.operation,
                    "input_refs": list(value.input_refs),
                    "result": value.result,
                    "unit": value.unit,
                    "trace": value.trace,
                }
                for key, value in calculation_results.items()
            },
        },
        "RECENT_TOOL_HISTORY": tool_history[-10:],
    }
    return json.dumps(payload, ensure_ascii=False, default=str)



def _compact_retrieval_result(result: RetrievalResult) -> dict[str, Any]:
    def compact_candidate(candidate: Any) -> dict[str, Any]:
        if candidate is None:
            return {}
        content = normalize_text(candidate.content)
        value = candidate.value
        if isinstance(value, str) and len(value) > 1000:
            value = value[:1000] + "…"
        if content and len(content) > 1000:
            content = content[:1000] + "…"
        return {
            "value": value,
            "unit": candidate.unit,
            "document_title": candidate.document_title,
            "sheet_name": candidate.sheet_name,
            "cell_address": candidate.cell_address,
            "indicator": candidate.indicator,
            "row_label": candidate.row_label,
            "column_label": candidate.column_label,
            "period": candidate.period,
            "content": content or None,
            "score": candidate.score,
            "evidence_ids": list(candidate.evidence_ids),
        }

    return {
        "task_id": result.task_id,
        "status": result.status,
        "expected_information": result.expected_information,
        "selected": compact_candidate(result.selected),
        "candidates": [compact_candidate(item) for item in result.candidates[:4]],
        "ambiguity_reason": result.ambiguity_reason,
    }



_DOC_TITLE_RE = re.compile(r"《([^》]{2,200})》")
def _current_document_titles(question: str) -> list[str]:
    return list(dict.fromkeys(
        normalize_text(value)
        for value in _DOC_TITLE_RE.findall(normalize_text(question))
        if normalize_text(value)
    ))


def _resolve_initial_task_source_hint(
    item: Any,
    *,
    explicit_documents: list[str],
    inherited_source_hint: str | None,
) -> str | None:
    """Bind an initial retrieval task to the most plausible explicit source.

    Explicit planner source_hint wins. With one named document it is the safe
    default. With multiple documents, never default all tasks to the last
    document; instead use deterministic lexical overlap, and leave the source
    unset when there is no unique match.
    """
    planned = normalize_text(getattr(item, "source_hint", None))
    if planned:
        return planned
    if len(explicit_documents) == 1:
        return explicit_documents[0]
    if len(explicit_documents) > 1:
        task_text = normalize_text(" ".join(str(value or "") for value in [
            getattr(item, "query", None),
            getattr(item, "expected_information", None),
            getattr(item, "indicator", None),
        ]))
        ranked = sorted(
            (
                (_document_task_overlap(task_text, title), index, title)
                for index, title in enumerate(explicit_documents)
            ),
            reverse=True,
        )
        if ranked and ranked[0][0] > 0:
            if len(ranked) == 1 or ranked[0][0] > ranked[1][0]:
                return ranked[0][2]
        return None
    return inherited_source_hint


def _document_task_overlap(task_text: str, document_title: str) -> int:
    """Small deterministic source matcher for explicitly named documents."""
    task = re.sub(r"\s+", "", normalize_text(task_text))
    title = re.sub(r"\s+", "", normalize_text(document_title))
    if not task or not title:
        return 0
    if title in task:
        return 1000 + len(title)

    # Chinese bigrams discriminate titles such as “资本管理办法” vs
    # “恢复和处置计划实施暂行办法” without any domain-specific hardcoding.
    task_bigrams = {task[i:i + 2] for i in range(max(0, len(task) - 1))}
    title_bigrams = {title[i:i + 2] for i in range(max(0, len(title) - 1))}
    generic = {"办法", "实施", "银行", "机构", "商业", "管理", "情况", "数据"}
    shared = (task_bigrams & title_bigrams) - generic
    return sum(len(token) for token in shared)



def _safe_failure_plan(question: str, reason: str) -> QueryPlan:
    """Create a recoverable seed plan after structured-planner failure."""
    return QueryPlan.model_validate({
        "original_query": question,
        "user_goal": "根据当前问题逐步检索证据并完成回答",
        "answer_requirements": [{
            "id": "ar_recovery",
            "question": question,
            "required_outputs": [],
        }],
        "entities": {},
        "retrieval_tasks": [],
        "operations": [],
        "requires_multiple_sources": False,
        "requires_table_retrieval": False,
        "requires_calculation": False,
        "requires_clarification": False,
        "clarification_reason": None,
    })


def retrieval_task_from_decision(
    question: str,
    decision: AgentStepDecision,
    task_id: str,
    *,
    fallback_source_hint: str | None = None,
) -> RetrievalTask:
    period = normalize_text(decision.period) or None
    year, month, quarter = _period_scope(period or decision.query or "")
    return RetrievalTask.model_validate({
        "id": task_id,
        "query": decision.query or decision.expected_information or question,
        "expected_information": decision.expected_information or decision.summary,
        "source_scope": {
            "document_title": normalize_text(decision.source_hint or fallback_source_hint) or None,
            "year": year,
            "month": month,
            "quarter": quarter,
        },
        "semantic_constraints": {
            "indicator": decision.indicator,
            "institution": decision.institution,
            "region": decision.region,
            "period": period,
            "statistical_scope": decision.statistical_scope,
            "row_label": decision.row_label,
            "column_label": decision.column_label,
        },
        "expected_value_type": decision.expected_value_type or "text",
        "expected_unit": _explicit_expected_unit(question, decision.expected_unit),
        "dependencies": [],
        "search_mode": decision.search_mode,
        "selection": decision.selection,
        "exclude_row_labels": list(decision.exclude_row_labels),
    })


def calculation_task_from_decision(decision: AgentStepDecision, task_id: str) -> CalculationTask:
    return CalculationTask.model_validate({
        "id": task_id,
        "type": decision.operation_type,
        "output_id": decision.output_id,
        "inputs": list(decision.inputs),
        "left": decision.left,
        "right": decision.right,
        "old_ref": decision.old_ref,
        "new_ref": decision.new_ref,
        "parameters": ({"absolute": decision.absolute} if decision.absolute is not None else {}),
    })


def _expand_planner_output(
    question: str,
    compact: PlannerOutput,
    conversation_context: list[dict[str, Any]] | None = None,
) -> QueryPlan:
    calculation_inputs = {
        ref
        for operation in compact.operations
        for ref in operation.input_refs()
    }
    retrieval_tasks: list[RetrievalTask] = []
    explicit_documents = _current_document_titles(question)
    inherited_source_hint = None
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
        source_hint = _resolve_initial_task_source_hint(
            item,
            explicit_documents=explicit_documents,
            inherited_source_hint=inherited_source_hint,
        )
        retrieval_tasks.append(RetrievalTask.model_validate({
            "id": item.id,
            "query": item.query,
            "expected_information": item.expected_information,
            "source_scope": {
                "document_title": source_hint,
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
            "search_mode": item.search_mode,
            "selection": item.selection,
            "exclude_row_labels": list(item.exclude_row_labels),
        }))
        _append_unique(indicators, item.indicator)
        _append_unique(institutions, item.institution)
        _append_unique(periods, period)
        _append_unique(documents, source_hint)
        _append_unique(regions, item.region)
        _append_unique(units, expected_unit)

    operations: list[CalculationTask] = []
    prior_outputs: set[str] = set()
    for index, item in enumerate(compact.operations, 1):
        # The first plan is advisory. If a comparison/max/min has been
        # recognized before enough operands are known, skip it for now instead
        # of invalidating the whole plan. The Agent Loop can create the
        # executable CalculationTask after additional retrieval.
        if not _initial_operation_is_executable(item):
            continue
        inputs = list(item.inputs)
        if item.type == "subtract" and item.absolute is True and len(inputs) == 2:
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
            "问题可能缺少完成查询所需的必要维度"
            if compact.requires_clarification else None
        ),
    })


def _initial_operation_is_executable(item: Any) -> bool:
    """Whether the initial planner already knows enough refs to run the op."""
    refs = item.input_refs()
    if item.type == "growth_rate":
        return bool(item.old_ref and item.new_ref)
    if item.type == "subtract":
        return bool(
            (item.left and item.right)
            or (len(item.inputs) == 2 and item.absolute is not None)
        )
    if item.type in {"divide", "compare"}:
        return len(refs) == 2
    return len(refs) >= 2


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


# Multi-character units may be found directly. Single-character 件 is accepted
# only in an explicit unit context, never inside words such as 附件/文件/事件.
_EXPLICIT_UNIT_RE = re.compile(r"万亿元|亿元|百万元|万元|万件|百分比|[%％‰]|元")
_SINGLE_PIECE_UNIT_RE = re.compile(r"(?:\d+(?:\.\d+)?\s*件|单位\s*[:：]\s*件|数量\s*[（(]\s*件\s*[）)])")


def _explicit_expected_unit(question: str, suggested: str | None) -> str | None:
    normalized_question = normalize_text(question)
    explicit = list(dict.fromkeys(
        _canonical_unit(match.group(0))
        for match in _EXPLICIT_UNIT_RE.finditer(normalized_question)
    ))
    if _SINGLE_PIECE_UNIT_RE.search(normalized_question):
        explicit.append("件")
    explicit = list(dict.fromkeys(item for item in explicit if item))
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
    explicit = normalize_text(item.column_label) or None
    row = canonical_table_label(item.row_label)
    indicator = canonical_table_label(item.indicator)
    source = canonical_table_label(item.source_hint)
    aggregate_row = row == "合计" or row.endswith("合计")
    source_names_metric = bool(indicator and source and indicator in source)
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
