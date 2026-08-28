from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


QaType = Literal[
    "regulatory_fact",
    "clause_threshold",
    "business_process",
    "table_lookup",
    "cross_file_judgment",
]


@dataclass
class Document:
    doc_id: str
    title: str
    authority: str | None
    document_no: str | None
    publish_date: str | None
    effective_date: str | None
    expire_date: str | None
    document_type: str
    topic: list[str]
    version: str | None
    status: str
    source_url: str | None
    local_path: str
    sha256: str
    file_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TextEvidence:
    evidence_id: str
    doc_id: str
    content: str
    page: int | None = None
    chapter: str | None = None
    article_no: str | None = None
    paragraph_no: int | None = None
    section: str | None = None
    source_url: str | None = None
    source_location: str | None = None
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TableCellEvidence:
    evidence_id: str
    doc_id: str
    sheet_name: str
    table_name: str | None
    indicator: str | None
    period: str | None
    value: Any
    unit: str | None
    row_header: str | None
    column_header: str | None
    cell_address: str
    context: str
    source_url: str | None = None
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedQuery:
    original_query: str
    qa_type: QaType
    entities: dict[str, Any] = field(default_factory=dict)
    requires_table: bool = False
    requires_multi_hop: bool = False
    rewritten_queries: list[str] = field(default_factory=list)
    # ``qa_type`` remains a compatibility/retrieval label.  The fields below
    # describe what the user wants, how the answer should be returned, and
    # which capabilities the workflow planner must compose.
    intent: str = "lookup"
    answer_format: str = "free_text"
    requirements: dict[str, bool] = field(default_factory=lambda: {
        "retrieval": True,
        "multi_file": False,
        "table": False,
        "calculation": False,
        "comparison": False,
        "multi_hop": False,
        "option_evaluation": False,
    })


@dataclass
class Verification:
    numeric_ok: bool = True
    date_ok: bool = True
    entity_ok: bool = True
    document_no_ok: bool = True
    normative_strength_ok: bool = True
    citation_ok: bool = True
    version_ok: bool = True
    unsupported_claims: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    claim_results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(
            [
                self.numeric_ok,
                self.date_ok,
                self.entity_ok,
                self.document_no_ok,
                self.normative_strength_ok,
                self.citation_ok,
                self.version_ok,
            ]
        ) and not self.unsupported_claims and not self.conflicts

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrustDecision:
    score: float
    decision: Literal["answer", "clarify", "refuse"]
    reasons: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QAResponse:
    answer: str
    qa_type: QaType
    evidence: list[dict[str, Any]]
    verification: dict[str, Any]
    trust: dict[str, Any]
    trace_id: str
    latency_ms: int
    query_plan: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
