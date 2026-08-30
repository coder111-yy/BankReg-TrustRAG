from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictPlanModel(BaseModel):
    """Strict model used for LLM-generated execution contracts."""

    model_config = ConfigDict(extra="forbid")


class QueryEntities(StrictPlanModel):
    indicators: list[str] = Field(default_factory=list)
    institutions: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    periods: list[str] = Field(default_factory=list)
    documents: list[str] = Field(default_factory=list)
    sheets: list[str] = Field(default_factory=list)
    regions: list[str] = Field(default_factory=list)
    units: list[str] = Field(default_factory=list)


class AnswerRequirement(StrictPlanModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    # Kept for backwards compatibility with the existing planner and traces.
    # In the adaptive agent, these references are hints rather than a hard
    # execution contract. The agent may satisfy the requirement with later,
    # dynamically-created retrieval/calculation outputs.
    required_outputs: list[str] = Field(default_factory=list)


class SubQuestion(StrictPlanModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    purpose: str = ""
    dependencies: list[str] = Field(default_factory=list)


class SourceScope(StrictPlanModel):
    document_title: str | None = None
    document_type: str | None = None
    year: int | None = Field(default=None, ge=1900, le=2200)
    month: int | None = Field(default=None, ge=1, le=12)
    quarter: int | None = Field(default=None, ge=1, le=4)


class SemanticConstraints(StrictPlanModel):
    indicator: str | None = None
    parent_indicator: str | None = None
    institution: str | None = None
    region: str | None = None
    period: str | None = None
    statistical_scope: str | None = None
    row_label: str | None = None
    column_label: str | None = None


ExpectedValueType = Literal["number", "string", "boolean", "text", "table_cell"]
SearchMode = Literal["precise", "broad"]
SelectionMode = Literal["single", "max", "min", "all"]


class PlannerRetrievalTask(StrictPlanModel):
    """Compact LLM-facing retrieval request used by the initial planner."""

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_information: str = Field(min_length=1)
    indicator: str | None = None
    institution: str | None = None
    period: str | None = None
    source_hint: str | None = None
    region: str | None = None
    statistical_scope: str | None = None
    row_label: str | None = None
    column_label: str | None = None
    expected_value_type: ExpectedValueType | None = None
    expected_unit: str | None = None
    search_mode: SearchMode = "precise"
    selection: SelectionMode = "single"
    exclude_row_labels: list[str] = Field(default_factory=list)


class PlannerOperation(StrictPlanModel):
    type: Literal["sum", "subtract", "divide", "compare", "max", "min", "growth_rate", "verify_consistency"]
    output_id: str = Field(min_length=1)
    inputs: list[str] = Field(default_factory=list)
    left: str | None = None
    right: str | None = None
    old_ref: str | None = None
    new_ref: str | None = None
    absolute: bool | None = None

    def input_refs(self) -> list[str]:
        if self.type == "subtract" and self.left and self.right:
            return [self.left, self.right]
        if self.type == "growth_rate" and self.old_ref and self.new_ref:
            return [self.old_ref, self.new_ref]
        return list(self.inputs)

    @model_validator(mode="after")
    def validate_compact_operation(self) -> "PlannerOperation":
        """Keep the initial plan permissive.

        The first plan is only a hypothesis. It may correctly recognize a
        max/min/comparison before all operands have been discovered. Missing
        operands are completed later by the adaptive Agent Loop.
        """
        if self.type == "growth_rate" and bool(self.old_ref) != bool(self.new_ref):
            raise ValueError("growth_rate old_ref/new_ref must be both present or both omitted")
        if self.type == "subtract" and (self.left or self.right) and not (self.left and self.right):
            raise ValueError("subtract left/right must be both present or both omitted")
        return self


class PlannerOutput(StrictPlanModel):
    """Initial plan. It is a hypothesis and may be revised during execution."""

    user_goal: str = Field(min_length=1)
    answer_requirements: list[AnswerRequirement] = Field(min_length=1)
    retrieval_tasks: list[PlannerRetrievalTask] = Field(default_factory=list)
    operations: list[PlannerOperation] = Field(default_factory=list)
    requires_clarification: bool = False


class RetrievalTask(StrictPlanModel):
    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_information: str = Field(min_length=1)
    source_scope: SourceScope = Field(default_factory=SourceScope)
    semantic_constraints: SemanticConstraints = Field(default_factory=SemanticConstraints)
    expected_value_type: ExpectedValueType = "text"
    expected_unit: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    search_mode: SearchMode = "precise"
    selection: SelectionMode = "single"
    exclude_row_labels: list[str] = Field(default_factory=list)


CalculationType = Literal[
    "sum",
    "subtract",
    "divide",
    "compare",
    "max",
    "min",
    "growth_rate",
    "verify_consistency",
    "none",
]


class CalculationTask(StrictPlanModel):
    id: str = Field(min_length=1)
    type: CalculationType
    output_id: str = Field(min_length=1)
    inputs: list[str] = Field(default_factory=list)
    left: str | None = None
    right: str | None = None
    old_ref: str | None = None
    new_ref: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)

    def input_refs(self) -> list[str]:
        if self.type == "subtract" and self.left and self.right:
            return [self.left, self.right]
        if self.type == "growth_rate" and self.old_ref and self.new_ref:
            return [self.old_ref, self.new_ref]
        return list(self.inputs)

    @model_validator(mode="after")
    def validate_operation_contract(self) -> "CalculationTask":
        refs = self.input_refs()
        if self.type == "growth_rate" and not (self.old_ref and self.new_ref):
            raise ValueError("growth_rate requires explicit old_ref and new_ref")
        if self.type == "subtract":
            explicit_direction = bool(self.left and self.right)
            explicit_absolute = len(self.inputs) == 2 and "absolute" in self.parameters
            if not (explicit_direction or explicit_absolute):
                raise ValueError("subtract requires left/right or two inputs with an explicit absolute flag")
        if self.type in {"divide", "compare"} and len(refs) != 2:
            raise ValueError(f"{self.type} requires exactly two input references")
        minimum = 0 if self.type == "none" else 2
        if len(refs) < minimum:
            raise ValueError(f"{self.type} requires at least {minimum} input references")
        return self


class AgentStepDecision(StrictPlanModel):
    """One observable next action chosen by the adaptive agent.

    This deliberately captures only a public action summary, never hidden
    chain-of-thought. The executor feeds the resulting observation back to the
    model and asks for the next action again.
    """

    action: Literal["search", "calculate", "answer", "clarify", "stop"]
    summary: str = Field(min_length=1, max_length=300)

    # Search action fields.
    query: str | None = None
    expected_information: str | None = None
    expected_value_type: ExpectedValueType | None = None
    expected_unit: str | None = None
    source_hint: str | None = None
    indicator: str | None = None
    institution: str | None = None
    period: str | None = None
    region: str | None = None
    statistical_scope: str | None = None
    row_label: str | None = None
    column_label: str | None = None
    search_mode: SearchMode = "precise"
    selection: SelectionMode = "single"
    exclude_row_labels: list[str] = Field(default_factory=list)

    # Calculation action fields.
    operation_type: CalculationType | None = None
    output_id: str | None = None
    inputs: list[str] = Field(default_factory=list)
    left: str | None = None
    right: str | None = None
    old_ref: str | None = None
    new_ref: str | None = None
    absolute: bool | None = None

    # Clarification/stop fields.
    clarification_question: str | None = None

    @model_validator(mode="after")
    def validate_action(self) -> "AgentStepDecision":
        if self.action == "search":
            if not self.query or not self.expected_information:
                raise ValueError("search action requires query and expected_information")
        elif self.action == "calculate":
            if not self.operation_type or not self.output_id:
                raise ValueError("calculate action requires operation_type and output_id")
            refs = self.calculation_input_refs()
            if self.operation_type == "growth_rate" and not (self.old_ref and self.new_ref):
                raise ValueError("growth_rate requires old_ref and new_ref")
            if self.operation_type == "subtract" and not (
                (self.left and self.right) or (len(self.inputs) == 2 and self.absolute is not None)
            ):
                raise ValueError("subtract requires left/right or two inputs plus absolute")
            if self.operation_type != "none" and len(refs) < 2:
                raise ValueError("calculation action requires at least two input references")
        elif self.action == "clarify" and not self.clarification_question:
            raise ValueError("clarify action requires clarification_question")
        return self

    def calculation_input_refs(self) -> list[str]:
        if self.operation_type == "subtract" and self.left and self.right:
            return [self.left, self.right]
        if self.operation_type == "growth_rate" and self.old_ref and self.new_ref:
            return [self.old_ref, self.new_ref]
        return list(self.inputs)


class QueryPlan(StrictPlanModel):
    original_query: str = Field(min_length=1)
    user_goal: str = Field(min_length=1)
    answer_requirements: list[AnswerRequirement] = Field(min_length=1)
    entities: QueryEntities = Field(default_factory=QueryEntities)
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    retrieval_tasks: list[RetrievalTask] = Field(default_factory=list)
    operations: list[CalculationTask] = Field(default_factory=list)
    requires_multiple_sources: bool = False
    requires_table_retrieval: bool = False
    requires_calculation: bool = False
    requires_clarification: bool = False
    clarification_reason: str | None = None

    @model_validator(mode="after")
    def validate_execution_graph(self) -> "QueryPlan":
        # The initial plan is advisory. We only enforce identifier integrity;
        # missing references are allowed because the adaptive loop can repair
        # the plan after observing retrieval results.
        retrieval_ids = [task.id for task in self.retrieval_tasks]
        operation_ids = [task.id for task in self.operations]
        output_ids = [task.output_id for task in self.operations]
        all_ids = retrieval_ids + operation_ids + output_ids
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("retrieval, calculation and output identifiers must be unique")
        if self.requires_clarification and not self.clarification_reason:
            raise ValueError("clarification_reason is required when requires_clarification is true")
        return self


class RetrievalCandidate(StrictPlanModel):
    value: str | None = None
    unit: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    document_id: str | None = None
    document_title: str | None = None
    document_type: str | None = None
    sheet_name: str | None = None
    cell_address: str | None = None
    indicator: str | None = None
    row_label: str | None = None
    column_label: str | None = None
    period: str | None = None
    content: str | None = None
    score: float = 0.0


class RetrievalResult(StrictPlanModel):
    task_id: str
    status: Literal["resolved", "ambiguous", "not_found", "blocked"]
    expected_information: str
    selected: RetrievalCandidate | None = None
    candidates: list[RetrievalCandidate] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    ambiguity_reason: str | None = None


class CalculationInput(StrictPlanModel):
    ref: str
    value: str
    unit: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class CalculationResult(StrictPlanModel):
    id: str
    operation: CalculationType
    input_refs: list[str]
    inputs: list[CalculationInput]
    result: str
    unit: str | None = None
    trace: str
    evidence_ids: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
