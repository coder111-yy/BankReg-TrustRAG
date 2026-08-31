from __future__ import annotations

import re
from difflib import SequenceMatcher
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
        document_catalog: list[dict[str, Any]] | None = None,
    ):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max(512, int(max_tokens))
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        # The planner sees the REAL ingested document catalog and decides which
        # files are needed. Python no longer guesses the source route from the
        # question shape.
        self.document_catalog = list(document_catalog or [])

    @classmethod
    def from_settings(
        cls,
        settings: Any,
        client: LLMClient | None = None,
        *,
        document_catalog: list[dict[str, Any]] | None = None,
    ) -> "QueryPlanner":
        return cls(
            client or LLMClient.from_planner_settings(settings),
            temperature=float(getattr(settings, "llm_planner_temperature", 0.0)),
            max_tokens=int(getattr(settings, "llm_planner_max_tokens", 2500)),
            timeout_seconds=float(getattr(settings, "llm_planner_timeout_seconds", 60.0)),
            document_catalog=document_catalog,
        )

    def plan(self, question: str, conversation_context: list[dict[str, Any]] | None = None) -> PlannerOutcome:
        normalized_question = normalize_text(question)
        planner_input = _planner_input(
            normalized_question,
            conversation_context,
            self.document_catalog,
        )
        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": planner_input},
        ]
        first = self.client.structured(
            messages,
            PlannerOutput,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
            prefer_json_schema=False,
            max_attempts=2,
        )
        if first.status != "ok" or not isinstance(first.value, PlannerOutput):
            return PlannerOutcome(
                first.status,
                _safe_failure_plan(normalized_question, first.error or first.status),
                first.attempts,
                first.error,
                first.errors,
            )

        # A second LLM pass audits the evidence map before execution. This is
        # deliberately model-driven rather than a Python rule such as
        # "multiple choice => one file" or "two years => two files".
        draft = _canonicalize_planner_sources(first.value, self.document_catalog)
        review_messages = [
            {"role": "system", "content": _PLANNER_REVIEW_PROMPT},
            {
                "role": "user",
                "content": _planner_review_input(
                    normalized_question,
                    conversation_context,
                    self.document_catalog,
                    draft,
                ),
            },
        ]
        review = self.client.structured(
            review_messages,
            PlannerOutput,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
            prefer_json_schema=False,
            max_attempts=1,
        )
        compact = review.value if review.status == "ok" and isinstance(review.value, PlannerOutput) else draft
        compact = _canonicalize_planner_sources(compact, self.document_catalog)
        plan = _source_grounded_choice_plan(normalized_question, compact)
        plan = plan or _expand_planner_output(normalized_question, compact)
        diagnostics = tuple(first.errors or ()) + tuple(review.errors or ())
        attempts = int(first.attempts or 0) + int(review.attempts or 0)
        return PlannerOutcome("ok", plan, attempts, diagnostics=diagnostics)


_PLANNER_SYSTEM_PROMPT = """你是“证据检索规划器”。你不回答用户问题，不判断最终答案，只负责决定：为了回答问题，需要去哪些真实文件、检索这些文件中的哪些内容、以及需要哪些确定性计算。只返回JSON。

你会收到 AVAILABLE_DOCUMENTS，它来自当前本地知识库真实入库文件。

规划原则：
1. 先分析最终结论依赖哪些“独立证据前提”，再为每个前提建立 retrieval_task。不要先按题型套固定路线。
2. 由你判断需要检索哪些文件。source_hint 应尽量从 AVAILABLE_DOCUMENTS 中复制真实标题；不要编造不存在的文件名。
3. 由你判断每个文件中要找“哪一部分”。把目标章节、附件、表格、工作表、指标、期间、主体、行列口径等写进 query 和 expected_information；有结构化字段时同时填写 indicator、period、row_label、column_label 等。
4. 不同文件中的独立事实必须拆成不同 retrieval_task；同一文件中若需要彼此独立且位置明显不同的事实，也可以拆成多个任务。
5. 制度规则与统计数据是不同证据前提。例如“依据制度口径并核对2024、2025报表再比较”，通常需要：制度原文任务 + 2024数据任务 + 2025数据任务 + 必要的计算操作。不要让一个制度文件任务承担统计报表取数。
6. 多年份/多季度比较时，每个待比较的原始值必须有自己的可追溯 retrieval_task；差值、增长率、最大/最小等再交给 operation。
7. 对单一来源的纯文本问题，如果一个文件的一组原文足以回答，可以只创建一个 retrieval_task；是否为选择题不改变这一原则。
8. 选择题只是回答格式。不要机械地为A/B/C/D各建任务，也不要机械地把所有选项压成一个任务；应根据“最终判断真正需要哪些证据前提”来规划。
9. 如果题目同时点名/暗示多个来源（例如“附件22并核对2024年、2025年主要监管指标表”），必须在计划中体现这些来源，而不是只保留第一个《》中的文件。
10. 若 AVAILABLE_DOCUMENTS 中存在与用户简称对应的真实标题，source_hint 使用真实标题；query 可保留用户简称以提高召回。
11. 每个 answer_requirement.required_outputs 必须绑定足以完成该要求的最终 retrieval 或 calc 输出。若结论同时依赖规则证据和计算结果，应把两者都绑定。
12. 需要计算时生成 sum、subtract、growth_rate、divide、compare、max 或 min；相差多少使用 subtract；百分点差直接对两个百分比数值做 subtract；增长率明确 old_ref/new_ref。
13. 用户未明确单位时省略 expected_unit，不猜单位。
14. 宽表精确取值要区分 row_label 与 column_label。
15. 只有确实缺少用户无法由知识库补全的必要维度时才 requires_clarification=true。不要因为你还没检索就提前澄清。

只允许五个顶层字段：user_goal、answer_requirements、retrieval_tasks、operations、requires_clarification。
retrieval_task 使用：id、query、expected_information，以及必要时的 indicator、institution、period、source_hint、region、statistical_scope、row_label、column_label、expected_value_type、expected_unit。省略无关字段，不要输出null。
operation 使用：type、output_id、inputs；subtract 可用 inputs+absolute 或 left/right；growth_rate 使用 old_ref/new_ref。
只返回JSON对象，不要Markdown。"""


_PLANNER_REVIEW_PROMPT = """你是“检索计划审查器”。不要回答用户问题。你要审查 DRAFT_PLAN 是否真的覆盖了形成最终答案所需的全部证据前提，并直接输出修正后的 PlannerOutput JSON。

审查重点：
- 是否遗漏用户明确要求核对的文件、年份、季度、工作表或指标；
- 是否把本应来自不同来源的事实错误塞进同一个 retrieval_task；
- source_hint 是否来自 AVAILABLE_DOCUMENTS 的真实标题；
- query / expected_information 是否明确到文件中的目标部分，而不是泛泛搜索整份文件；
- 表格比较是否分别取得每个原始值；
- 差值、增长率、比较等是否建立了 operation；
- answer_requirement.required_outputs 是否同时覆盖规则依据、数据依据和最终计算结果；
- 不按选择题/判断题/开放题套固定流程，只按证据依赖关系规划。

如果 DRAFT_PLAN 已正确，原样输出；否则修复。只输出JSON。"""


def _planner_input(
    question: str,
    conversation_context: list[dict[str, Any]] | None,
    document_catalog: list[dict[str, Any]] | None,
) -> str:
    prior_user_messages = [
        normalize_text(item.get("content"))
        for item in (conversation_context or [])
        if item.get("role") == "user" and normalize_text(item.get("content"))
    ][-3:]
    context = "\n".join(f"- {item[:500]}" for item in prior_user_messages) or "无"
    catalog = _format_document_catalog(document_catalog)
    return (
        f"USER_QUESTION:\n{question}\n\n"
        f"PRIOR_USER_CONTEXT:\n{context}\n\n"
        f"AVAILABLE_DOCUMENTS (真实入库文件标题；请从中选择 source_hint):\n{catalog}"
    )


def _planner_review_input(
    question: str,
    conversation_context: list[dict[str, Any]] | None,
    document_catalog: list[dict[str, Any]] | None,
    draft: PlannerOutput,
) -> str:
    base = _planner_input(question, conversation_context, document_catalog)
    return f"{base}\n\nDRAFT_PLAN:\n{draft.model_dump_json(exclude_none=True)}"


def _format_document_catalog(document_catalog: list[dict[str, Any]] | None) -> str:
    rows: list[str] = []
    seen: set[tuple[str, str]] = set()
    for raw in document_catalog or []:
        doc = dict(raw or {})
        title = normalize_text(doc.get("title") or doc.get("file_name"))
        file_name = normalize_text(doc.get("file_name"))
        doc_type = normalize_text(doc.get("document_type") or doc.get("file_type"))
        if not title and not file_name:
            continue
        key = (title, file_name)
        if key in seen:
            continue
        seen.add(key)
        extras: list[str] = []
        if file_name and file_name != title:
            extras.append(f"file={file_name}")
        if doc_type:
            extras.append(f"type={doc_type}")
        sheets = doc.get("sheet_names") or doc.get("sheets")
        if isinstance(sheets, (list, tuple)) and sheets:
            rendered = ",".join(normalize_text(value) for value in sheets[:8] if normalize_text(value))
            if rendered:
                extras.append(f"sheets={rendered}")
        suffix = f" | {' | '.join(extras)}" if extras else ""
        rows.append(f"- {title or file_name}{suffix}")
    if not rows:
        return "(目录不可用；仅依据用户问题规划 source_hint，不要杜撰具体文件标题)"
    # The current corpus is about five hundred documents. Title-only metadata
    # remains bounded and avoids a Python pre-router silently removing a source.
    return "\n".join(rows[:700])


def _canonicalize_planner_sources(
    compact: PlannerOutput,
    document_catalog: list[dict[str, Any]] | None,
) -> PlannerOutput:
    """Canonicalize an LLM-chosen source to the real catalog title.

    This does not choose a source for the model. It only repairs harmless title
    spelling/prefix differences after the model has already selected one.
    """
    catalog: list[tuple[str, str]] = []
    for raw in document_catalog or []:
        doc = dict(raw or {})
        title = normalize_text(doc.get("title") or doc.get("file_name"))
        file_name = normalize_text(doc.get("file_name"))
        if title:
            catalog.append((title, file_name))
    if not catalog:
        return compact

    repaired = []
    for task in compact.retrieval_tasks:
        hint = normalize_text(task.source_hint)
        if not hint:
            repaired.append(task)
            continue
        hint_key = canonical_table_label(hint)
        exact = [
            title for title, file_name in catalog
            if hint_key in {canonical_table_label(title), canonical_table_label(file_name)}
        ]
        matches = exact
        if not matches:
            matches = [
                title for title, file_name in catalog
                if hint_key and (
                    hint_key in canonical_table_label(title)
                    or canonical_table_label(title) in hint_key
                    or (file_name and hint_key in canonical_table_label(file_name))
                )
            ]
        unique = list(dict.fromkeys(matches))
        repaired.append(task.model_copy(update={"source_hint": unique[0]}) if len(unique) == 1 else task)
    return compact.model_copy(update={"retrieval_tasks": repaired})


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
    """Convert the LLM-reviewed compact plan into the executor contract.

    The model is authoritative about which files and which evidence parts are
    required. Do not rewrite the plan based on Python question-shape heuristics.
    """
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
        has_numeric_coordinate = bool(
            period
            or year
            or month
            or quarter
            or item.row_label
            or item.column_label
            or re.search(r"\.(?:xlsx?|xls)(?:$|\b)", normalize_text(item.source_hint), re.IGNORECASE)
        )
        expected_type = item.expected_value_type or (
            "number" if item.id in calculation_inputs and has_numeric_coordinate else "text"
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

    structured_numeric_refs = {
        task.id
        for task in retrieval_tasks
        if (
            task.source_scope.year
            or task.source_scope.month
            or task.source_scope.quarter
            or task.semantic_constraints.period
            or task.semantic_constraints.row_label
            or task.semantic_constraints.column_label
            or normalize_text(task.source_scope.document_type).lower() in {"excel", "xls", "xlsx"}
        )
    }
    numeric_retrieval_refs = {
        task.id
        for task in retrieval_tasks
        if task.expected_value_type in {"number", "table_cell"} and task.id in structured_numeric_refs
    }
    operations: list[CalculationTask] = []
    prior_outputs: set[str] = set()
    for index, item in enumerate(compact.operations, 1):
        inputs = [
            _resolve_plan_reference(
                ref,
                retrieval_tasks,
                prior_outputs,
            )
            for ref in item.inputs
        ]
        left = _resolve_plan_reference(item.left, retrieval_tasks, prior_outputs) if item.left else None
        right = _resolve_plan_reference(item.right, retrieval_tasks, prior_outputs) if item.right else None
        old_ref = _resolve_plan_reference(item.old_ref, retrieval_tasks, prior_outputs) if item.old_ref else None
        new_ref = _resolve_plan_reference(item.new_ref, retrieval_tasks, prior_outputs) if item.new_ref else None
        # The LLM may include a supporting rule-text task alongside the two
        # numeric operands of compare/divide.  Keep that rule bound to the
        # answer requirement, but never pass it into a binary calculator.
        if item.type in {"compare", "divide"} and len(inputs) != 2:
            structured_operands = [ref for ref in inputs if ref in structured_numeric_refs]
            retrieval_operands = [ref for ref in inputs if ref in numeric_retrieval_refs]
            available_operands = [
                ref for ref in inputs
                if ref in numeric_retrieval_refs or ref in prior_outputs
            ]
            if len(structured_operands) >= 2:
                inputs = structured_operands[:2]
            elif len(retrieval_operands) >= 2:
                inputs = retrieval_operands[:2]
            elif len(available_operands) >= 2:
                inputs = available_operands[:2]
            elif len(inputs) > 2:
                # Supporting evidence is conventionally listed before operands.
                inputs = inputs[-2:]

        absolute = item.absolute
        if item.type == "subtract" and not (left and right) and len(inputs) == 2 and absolute is None:
            absolute = _infer_absolute_subtraction(question)
        if item.type == "subtract" and absolute is True and len(inputs) == 2:
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
            "left": left,
            "right": right,
            "old_ref": old_ref,
            "new_ref": new_ref,
            "parameters": ({"absolute": absolute} if absolute is not None else {}),
        })
        operations.append(operation)
        prior_outputs.add(operation.output_id)

    available_outputs = {
        *(task.id for task in retrieval_tasks),
        *(operation.output_id for operation in operations),
    }
    expanded_requirements = []
    calculation_output_ids = {operation.output_id for operation in operations}
    for requirement in compact.answer_requirements:
        resolved = [
            _resolve_plan_reference(ref, retrieval_tasks, calculation_output_ids)
            for ref in requirement.required_outputs
        ]
        resolved = list(dict.fromkeys(ref for ref in resolved if ref in available_outputs))
        # A model-generated semantic alias must never make an otherwise valid
        # execution plan impossible to bind.  If none of its aliases can be
        # resolved, the Answer Agent may use any completed plan output.
        if not resolved:
            resolved = list(dict.fromkeys([
                *(task.id for task in retrieval_tasks),
                *(operation.output_id for operation in operations),
            ]))
        expanded_requirements.append(requirement.model_copy(update={
            "required_outputs": resolved,
        }))
    return QueryPlan.model_validate({
        "original_query": question,
        "user_goal": compact.user_goal,
        "answer_requirements": [item.model_dump() for item in expanded_requirements],
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


def _infer_absolute_subtraction(question: str) -> bool:
    """Infer only the missing subtraction direction from explicit wording."""
    text = normalize_text(question)
    if re.search(r"从.+到|上升|下降|增加|减少|变化|变动", text):
        return False
    return any(term in text for term in ("相差", "差值", "差额", "绝对差"))


def _resolve_plan_reference(
    reference: str,
    retrieval_tasks: list[RetrievalTask],
    calculation_outputs: set[str],
) -> str:
    """Bind an LLM semantic alias to one executable task/output ID."""
    reference = normalize_text(reference)
    if not reference:
        return reference
    exact_ids = {task.id for task in retrieval_tasks} | set(calculation_outputs)
    if reference in exact_ids:
        return reference

    candidates: dict[str, str] = {
        task.id: " ".join(str(value or "") for value in (
            task.id,
            task.query,
            task.expected_information,
            task.source_scope.document_title,
            task.source_scope.year,
            task.source_scope.month,
            task.source_scope.quarter,
            task.semantic_constraints.indicator,
            task.semantic_constraints.period,
            task.semantic_constraints.row_label,
            task.semantic_constraints.column_label,
        ))
        for task in retrieval_tasks
    }
    candidates.update({output: output for output in calculation_outputs})

    years = set(re.findall(r"(?:19|20)\d{2}", reference))
    if years:
        matches = [key for key, blob in candidates.items() if years <= set(re.findall(r"(?:19|20)\d{2}", blob))]
        if len(matches) == 1:
            return matches[0]

    numbers = set(re.findall(r"(?<!\d)\d{1,3}(?!\d)", reference))
    if numbers:
        matches = [key for key, blob in candidates.items() if numbers <= set(re.findall(r"(?<!\d)\d{1,3}(?!\d)", blob))]
        if len(matches) == 1:
            return matches[0]

    reference_key = re.sub(r"[^0-9a-z]+", "", reference.lower())
    scored = sorted(
        (
            SequenceMatcher(None, reference_key, re.sub(r"[^0-9a-z]+", "", key.lower())).ratio(),
            key,
        )
        for key in candidates
        if reference_key
    )
    if scored and scored[-1][0] >= 0.48:
        return scored[-1][1]
    return reference



_CHOICE_MARKER_RE = re.compile(r"(?<![A-Za-z0-9])([A-H])\s*[.．、)]\s*", re.IGNORECASE)


def _source_grounded_choice_plan(question: str, compact: PlannerOutput) -> QueryPlan | None:
    """Collapse a single-source regulatory MCQ into one evidence task.

    This is a routing normalization, not a hard-coded answer rule. It applies
    only when the user explicitly names exactly one source before the first
    A/B/C... option and the question is textual/regulatory rather than a table
    calculation. The Answer Agent still decides which option is supported.
    """
    normalized = normalize_text(question)
    markers = list(_CHOICE_MARKER_RE.finditer(normalized))
    if len(markers) < 2:
        return None
    stem = normalized[:markers[0].start()]
    if not re.search(r"哪(?:一)?项|下列|表述|正确|错误|符合|不符合", stem):
        return None
    # A title inside 《》 is not proof that the question has only one source.
    # Annual workbooks are often referenced by logical name without brackets,
    # e.g. “依据《办法》附件并核对2024、2025监管指标表”.  Collapsing that
    # question to the quoted rule document erases both table retrieval tasks.
    years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", stem))
    if len(years) >= 2 or re.search(
        r"Excel|工作表|单元格|数值|金额|余额|占比|变化|差值|计算|"
        r"报表|统计表|监管指标|核对|对照|比较|相比|多个?文件|多份(?:文件|材料)",
        stem,
        re.IGNORECASE,
    ):
        return None
    source_titles = [normalize_text(value) for value in re.findall(r"《([^》]{2,160})》", stem)]
    source_titles = [value for value in source_titles if value]
    if len(source_titles) != 1:
        return None
    source_title = source_titles[0]

    options: list[str] = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(normalized)
        text = normalized[marker.end():end].strip(" ；;。")
        if text:
            options.append(f"{marker.group(1).upper()}. {text}")
    if len(options) < 2:
        return None

    requirement_id = compact.answer_requirements[0].id if compact.answer_requirements else "ar1"
    task_id = "r1"
    query = f"{source_title} {stem} {' '.join(options)}"
    expected_information = (
        f"在《{source_title}》中找到能够直接判断这些选项真假的原文依据；"
        "优先召回与选项中的数值阈值、适用区间、主体、义务、例外或名单直接对应的段落"
    )
    task = RetrievalTask.model_validate({
        "id": task_id,
        "query": query,
        "expected_information": expected_information,
        "source_scope": {"document_title": source_title},
        "semantic_constraints": {},
        "expected_value_type": "text",
        "expected_unit": None,
        "dependencies": [],
    })
    requirement_question = compact.answer_requirements[0].question if compact.answer_requirements else question
    return QueryPlan.model_validate({
        "original_query": question,
        "user_goal": compact.user_goal,
        "answer_requirements": [{
            "id": requirement_id,
            "question": requirement_question,
            "required_outputs": [task_id],
        }],
        "entities": {"documents": [source_title]},
        "retrieval_tasks": [task.model_dump()],
        "operations": [],
        "requires_multiple_sources": False,
        "requires_table_retrieval": False,
        "requires_calculation": False,
        "requires_clarification": False,
        "clarification_reason": None,
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


# ``件`` is a valid reporting unit, but it also occurs inside ordinary words
# such as ``附件``、``文件`` and ``事件``.  Match the compound count unit first
# and only accept standalone ``件`` when it is not part of these words.
_EXPLICIT_UNIT_RE = re.compile(
    r"万亿元|亿元|百万元|万元|万件|百分比|[%％‰]|元|(?<![附文事条案部])件"
)


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
