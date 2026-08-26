"""Deterministic query agents used by the BankReg-TrustRAG service.

The project book describes a Query Router and specialised agents.  This
module implements the first vertical slice without exposing chain-of-thought:
intent recognition, independent option retrieval, evidence scoring, and a
structured Human-in-the-loop handoff when the evidence is insufficient.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .query import extract_dimension_labels, extract_inline_choices
from .retrieval.index import Hit, HybridIndex
from .schemas import ParsedQuery
from .utils import canonical_table_label, normalize_text, normalized_number, tokens


@dataclass(frozen=True)
class ChoiceAgentResult:
    intent: str
    question: str
    choices: list[str]
    option_hits: dict[int, list[Hit]]
    assessments: list[dict[str, Any]]
    selected_index: int | None
    human_in_loop: dict[str, Any] | None

    @property
    def selected_label(self) -> str | None:
        if self.selected_index is None or self.selected_index >= 4:
            return None
        return "ABCD"[self.selected_index]

    @property
    def all_hits(self) -> list[Hit]:
        unique: dict[str, Hit] = {}
        for hits in self.option_hits.values():
            for hit in hits:
                previous = unique.get(hit.evidence_id)
                if previous is None or hit.fused_score > previous.fused_score:
                    unique[hit.evidence_id] = hit
        return sorted(unique.values(), key=lambda hit: hit.fused_score, reverse=True)

    def to_plan(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "route": "choice_agent",
            "question": self.question,
            "options": self.assessments,
            "selected_option": self.selected_label,
            "human_in_loop": self.human_in_loop or {"status": "not_required"},
        }


def identify_intent(question: str, choices: list[str] | None, qa_type: str) -> dict[str, Any]:
    """Return an auditable intent decision, without generating a conclusion."""
    is_choice = len([item for item in (choices or []) if normalize_text(item)]) >= 2
    return {
        "intent": "multiple_choice" if is_choice else qa_type,
        "qa_type": qa_type,
        "choice_count": len(choices or []),
        "source": "explicit_options" if choices else "inline_option_parser" if extract_inline_choices(question)[1] else "query_router",
    }


def build_agent_workflow(
    parsed: ParsedQuery,
    question: str,
    choices: list[str],
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create an auditable, non-CoT execution plan for the five Agent roles.

    The plan records only routes, inputs and expected artifacts.  It never
    exposes hidden reasoning, but makes Query/Retrieval/Table/Generation/
    Verification responsibilities reproducible from the API response.
    """
    intent = identify_intent(question, choices, parsed.qa_type)
    tasks: list[dict[str, Any]] = [
        {
            "id": "understand",
            "agent": "Query Agent",
            "action": "classify_extract_rewrite",
            "status": "completed",
            "output": {
                "qa_type": parsed.qa_type,
                "entities": parsed.entities,
                "rewritten_queries": parsed.rewritten_queries,
            },
        },
        {
            "id": "retrieve_primary",
            "agent": "Retrieval Agent",
            "action": "hybrid_retrieval_and_rerank",
            "status": "planned",
            "routes": ["bm25", "bge_vector", "metadata"],
            "filters": filters or {},
        },
    ]
    if parsed.qa_type == "table_lookup":
        tasks.append({
            "id": "table_lookup",
            "agent": "Table Agent",
            "action": "locate_indicator_period_cell",
            "status": "planned",
            "inputs": {key: parsed.entities.get(key) for key in ["indicator", "period", "row_label", "column_label"]},
        })
    elif parsed.qa_type == "cross_file_judgment":
        tasks.extend([
            {
                "id": "retrieve_rule",
                "agent": "Retrieval Agent",
                "action": "retrieve_rule_threshold_and_definition",
                "status": "planned",
                "routes": ["bm25", "bge_vector", "metadata"],
            },
            {
                "id": "table_calculation",
                "agent": "Table Agent",
                "action": "locate_value_and_compare_deterministically",
                "status": "planned",
                "inputs": {key: parsed.entities.get(key) for key in ["indicator", "period", "table_name"]},
            },
        ])
    if len(choices) >= 2:
        tasks.append({
            "id": "evaluate_options",
            "agent": "Choice Agent",
            "action": "retrieve_each_option_independently",
            "status": "planned",
            "option_count": len(choices),
        })
    tasks.extend([
        {
            "id": "generate",
            "agent": "Generation Agent",
            "action": "grounded_answer_from_minimal_evidence",
            "status": "planned",
        },
        {
            "id": "verify",
            "agent": "Verification Agent",
            "action": "claim_field_version_and_conflict_checks",
            "status": "planned",
        },
    ])
    return {
        "intent": intent,
        "question_understanding": {
            "qa_type": parsed.qa_type,
            "entities": parsed.entities,
            "requires_table": parsed.requires_table,
            "requires_multi_hop": parsed.requires_multi_hop,
        },
        "tasks": tasks,
    }


def organize_evidence(
    hits: list[Hit],
    minimal_evidence_ids: list[str],
    operations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach minimal evidence to explicit answer roles for audit/display."""
    role_by_id: dict[str, str] = {}
    for operation in operations:
        op_type = operation.get("type")
        # A cross-file operation may later be converted to ``refusal`` when
        # its value is present but a regulatory threshold is absent.  Keep
        # its table/rule evidence roles so the refusal remains traceable.
        if operation.get("table_evidence_ids") or operation.get("rule_evidence_ids"):
            for evidence_id in operation.get("table_evidence_ids", []):
                role_by_id[str(evidence_id)] = "statistical_value"
            for evidence_id in operation.get("rule_evidence_ids", []):
                role_by_id[str(evidence_id)] = "rule_or_formula"
        elif op_type == "table_lookup":
            for evidence_id in operation.get("display_evidence_ids", []):
                role_by_id[str(evidence_id)] = "table_value"
        elif op_type in {"choice_agent", "evidence_choice"}:
            for evidence_id in operation.get("display_evidence_ids", []):
                role_by_id[str(evidence_id)] = "option_support"
    by_id = {hit.evidence_id: hit for hit in hits}
    organized: list[dict[str, Any]] = []
    for evidence_id in dict.fromkeys(minimal_evidence_ids):
        hit = by_id.get(str(evidence_id))
        if hit is None:
            continue
        item = hit.item
        organized.append({
            "evidence_id": hit.evidence_id,
            "role": role_by_id.get(hit.evidence_id, "direct_support"),
            "kind": hit.kind,
            "source_title": item.get("source_title"),
            "source_location": item.get("source_location") or item.get("cell_address"),
            "document_status": item.get("document_status"),
        })
    return organized


def _claim_parts(option: str) -> list[str]:
    value = normalize_text(option).strip(" ：:，,；;。.!！？?\n\t")
    if not value:
        return []
    # Regulatory combination choices commonly contain two statements joined
    # by a semicolon.  Keep commas intact because they often belong to one
    # legal statement or a list of conditions.
    parts = [part.strip(" ：:，,；;。.!！？?\n\t") for part in re.split(r"[；;。！？!?]+", value)]
    return [part for part in parts if part]


def _hit_text(hit: Hit) -> str:
    item = hit.item
    return normalize_text(" ".join(str(item.get(key) or "") for key in [
        "content", "context_window", "context", "indicator", "period", "unit", "row_header", "column_header", "value_text",
    ]))


def _numeric_choice_value(option: str) -> float | None:
    value = normalize_text(option)
    if not re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?\s*[%％]?", value):
        return None
    return normalized_number(value)


def _strict_table_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = value
        if decoded != value or not isinstance(decoded, str):
            return _strict_table_number(decoded)
    if isinstance(value, (int, float)):
        return float(value)
    text = normalize_text(value)
    if not re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?\s*[%％]?", text):
        return None
    return normalized_number(text)


def _comparison_direction(question: str) -> str | None:
    normalized = normalize_text(question)
    if any(term in normalized for term in ("最高", "最大", "最多")):
        return "max"
    if any(term in normalized for term in ("最低", "最小", "最少")):
        return "min"
    return None


def _table_comparison_result(
    question: str,
    choices: list[str],
    option_hits: dict[int, list[Hit]],
) -> dict[str, Any] | None:
    """Compare the same structured column across row-label choices.

    Textual option support cannot answer questions such as "which region has
    the highest health-insurance value": every region label is present in the
    workbook, so all options otherwise tie. Resolve these questions from the
    exact numeric cells, and only when every option has a comparable value.
    """
    direction = _comparison_direction(question)
    if direction is None or len(choices) < 2:
        return None
    _, requested_column = extract_dimension_labels(question)
    column_key = canonical_table_label(requested_column)
    if not column_key:
        return None

    option_candidates: list[list[dict[str, Any]]] = []
    for option_index, option in enumerate(choices[:4]):
        option_key = canonical_table_label(option)
        candidates: list[tuple[Hit, float]] = []
        for hit in option_hits.get(option_index, []):
            if hit.kind != "table":
                continue
            item = hit.item
            row_keys = {
                canonical_table_label(item.get("indicator")),
                canonical_table_label(item.get("row_header")),
            }
            if option_key not in row_keys:
                continue
            column_context = canonical_table_label(" ".join(
                str(item.get(key) or "") for key in ("column_header", "period")
            ))
            if column_key not in column_context:
                continue
            value = _strict_table_number(item.get("value_text"))
            if value is not None:
                candidates.append((hit, value))
        if not candidates:
            return None
        unique: dict[str, dict[str, Any]] = {}
        for hit, value in candidates:
            unique[hit.evidence_id] = {
                "value": value,
                "evidence_id": hit.evidence_id,
                "cell_address": hit.item.get("cell_address"),
            }
        option_candidates.append(sorted(unique.values(), key=_comparison_candidate_sort_key))

    ranges = [
        (min(item["value"] for item in candidates), max(item["value"] for item in candidates))
        for candidates in option_candidates
    ]
    winners: list[int] = []
    for index, (lower, upper) in enumerate(ranges):
        other_ranges = [value_range for other_index, value_range in enumerate(ranges) if other_index != index]
        if not other_ranges:
            continue
        if direction == "max":
            other_boundary = max(value_range[1] for value_range in other_ranges)
            wins = _strictly_separated(lower, other_boundary, direction)
        else:
            other_boundary = min(value_range[0] for value_range in other_ranges)
            wins = _strictly_separated(upper, other_boundary, direction)
        if wins:
            winners.append(index)
    if len(winners) != 1:
        return None

    selected_index = winners[0]
    values: list[dict[str, Any]] = []
    evidence_ids: list[str] = []
    for option_index, candidates in enumerate(option_candidates):
        # Use conservative bounds in the displayed comparison: the winner's
        # worst possible value versus every loser's best possible value.  This
        # lets a repeated row label remain answerable only when every matching
        # row leads to the same result.
        if direction == "max":
            chosen = min(candidates, key=lambda item: item["value"]) if option_index == selected_index else max(candidates, key=lambda item: item["value"])
        else:
            chosen = max(candidates, key=lambda item: item["value"]) if option_index == selected_index else min(candidates, key=lambda item: item["value"])
        values.append({
            "choice_index": option_index,
            "label": "ABCD"[option_index],
            "text": choices[option_index],
            **chosen,
            "candidate_values": candidates,
        })
        evidence_ids.extend(item["evidence_id"] for item in candidates)
    return {
        "selected_index": selected_index,
        "direction": direction,
        "column": requested_column,
        "values": values,
        "evidence_ids": list(dict.fromkeys(evidence_ids)),
    }


def _comparison_candidate_sort_key(item: dict[str, Any]) -> tuple[int, str, float]:
    address = str(item.get("cell_address") or "")
    match = re.search(r"(\d+)$", address)
    return (int(match.group(1)) if match else 10**9, address, float(item["value"]))


def _strictly_separated(left: float, right: float, direction: str) -> bool:
    tolerance = max(1e-9, max(abs(left), abs(right)) * 1e-8)
    return left - right > tolerance if direction == "max" else right - left > tolerance


def _claim_support(claim: str, hit: Hit) -> float:
    evidence = _hit_text(hit)
    normalized_claim = normalize_text(claim)
    if not evidence or not normalized_claim:
        return 0.0
    if normalized_claim in evidence:
        return 1.0
    target = set(tokens(normalized_claim))
    overlap = len(target.intersection(tokens(evidence))) / max(len(target), 1)
    if overlap >= 0.78:
        return 0.9
    if overlap >= 0.58:
        return 0.72
    if overlap >= 0.38:
        return 0.5
    # Scores from the reranker are useful only as a tie-breaker.  They cannot
    # by themselves establish that a claim is supported.
    return min(0.34, 0.18 * max(hit.rerank_score, hit.dense_score) + 0.05 * hit.lexical_score)


def _option_support(option: str, hits: list[Hit]) -> tuple[float, float, list[str]]:
    claims = _claim_parts(option)
    if not claims or not hits:
        return 0.0, 0.0, []
    # Numeric choices must match a structured cell exactly.  This prevents a
    # nearby year or header number from being selected as the answer.
    target = _numeric_choice_value(option)
    if target is not None:
        exact = [
            hit for hit in hits
            if hit.kind == "table" and normalized_number(hit.item.get("value_text")) is not None
            and abs(float(normalized_number(hit.item.get("value_text"))) - target) < max(1e-9, abs(target) * 1e-8)
        ]
        if exact:
            return 1.0, 1.0, [hit.evidence_id for hit in exact[:3]]
    claim_scores: list[float] = []
    evidence_ids: list[str] = []
    for claim in claims:
        ranked = sorted((( _claim_support(claim, hit), hit) for hit in hits), key=lambda item: item[0], reverse=True)
        if ranked:
            score, hit = ranked[0]
            claim_scores.append(score)
            if score >= 0.5:
                evidence_ids.append(hit.evidence_id)
    if not claim_scores:
        return 0.0, 0.0, []
    overall = sum(claim_scores) / len(claim_scores)
    minimum = min(claim_scores)
    return overall, minimum, list(dict.fromkeys(evidence_ids))


def _agent_search(
    index: HybridIndex,
    query: str,
    qa_type: str,
    top_k: int,
    filters: dict[str, Any] | None,
    *,
    rerank: bool,
    dense: bool,
) -> list[Hit]:
    """Use rerank control when available while keeping fixture compatibility."""
    try:
        return index.hybrid_search(query, qa_type, top_k, filters, rerank=rerank, dense=dense)
    except TypeError:
        return index.hybrid_search(query, qa_type, top_k, filters)


def _merge_option_hits(*groups: list[Hit]) -> list[Hit]:
    unique: dict[str, Hit] = {}
    for group in groups:
        for hit in group:
            previous = unique.get(hit.evidence_id)
            if previous is None or hit.fused_score > previous.fused_score:
                unique[hit.evidence_id] = hit
    return sorted(unique.values(), key=lambda hit: hit.fused_score, reverse=True)


def run_choice_agent(
    index: HybridIndex,
    question: str,
    choices: list[str],
    qa_type: str,
    filters: dict[str, Any] | None = None,
    top_k: int = 8,
) -> ChoiceAgentResult:
    """Retrieve each choice independently with one shared BGE rerank pass."""
    option_hits: dict[int, list[Hit]] = {}
    assessments: list[dict[str, Any]] = []
    comparison_direction = _comparison_direction(question)
    _, comparison_column = extract_dimension_labels(question)
    # Rerank the stem once. Option-specific calls below still use independent
    # lexical/vector retrieval, but skip their duplicate CrossEncoder passes.
    # This preserves evidence separation while removing the largest CPU cost.
    shared_hits = _agent_search(index, question, qa_type, max(4, min(top_k, 12)), filters, rerank=True, dense=True)
    for option_index, option in enumerate(choices[:4]):
        label = "ABCD"[option_index]
        # The stem and the option are deliberately searched separately.  A
        # single query containing all options lets common words dominate and
        # was the source of the unrelated answer shown in the screenshot.
        if comparison_direction and comparison_column:
            # Make both table dimensions explicit so structured retrieval
            # filters to the option row before BGE reranking. Without this,
            # an exact row such as 天津 can be pushed out of a small top-k by
            # semantically similar rows, leaving the comparison incomplete.
            option_query = f"{question} 选项{label} “{option}”在“{comparison_column}”口径下"
        else:
            option_query = f"{question} 选项{label} {option}"
        option_specific_hits = _agent_search(index, option_query, qa_type, max(4, min(top_k, 12)), filters, rerank=False, dense=False)
        hits = _merge_option_hits(option_specific_hits, shared_hits)
        option_hits[option_index] = hits
        overall, minimum, evidence_ids = _option_support(option, hits)
        assessments.append({
            "label": label,
            "text": option,
            "score": round(overall, 6),
            "minimum_claim_score": round(minimum, 6),
            "evidence_ids": evidence_ids,
            "retrieved_evidence_ids": [hit.evidence_id for hit in hits[:6]],
        })

    selected_index: int | None = None
    human_in_loop: dict[str, Any] | None = None
    comparison = _table_comparison_result(question, choices, option_hits)
    if comparison is not None:
        selected_index = int(comparison["selected_index"])
        for index, assessment in enumerate(assessments):
            comparison_value = comparison["values"][index]
            is_selected = index == selected_index
            assessment.update({
                "score": 1.0 if is_selected else 0.0,
                "minimum_claim_score": 1.0 if is_selected else 0.0,
                "evidence_ids": comparison["evidence_ids"] if is_selected else [comparison_value["evidence_id"]],
                "table_comparison": {
                    "direction": comparison["direction"],
                    "column": comparison["column"],
                    "value": comparison_value["value"],
                    "cell_address": comparison_value["cell_address"],
                    "evidence_id": comparison_value["evidence_id"],
                    "compared_values": comparison["values"] if is_selected else None,
                },
            })
    ranked = sorted(enumerate(assessments), key=lambda item: item[1]["score"], reverse=True)
    if selected_index is None and ranked:
        best_index, best = ranked[0]
        second_score = ranked[1][1]["score"] if len(ranked) > 1 else 0.0
        margin = float(best["score"]) - float(second_score)
        # A choice is answerable only when every statement in it has evidence
        # and the best option is meaningfully ahead of the alternatives.
        if float(best["score"]) >= 0.60 and float(best["minimum_claim_score"]) >= 0.5 and margin >= 0.08:
            selected_index = best_index
        else:
            human_in_loop = {
                "status": "pending",
                "reason": "选项证据不足或多个选项得分接近，无法自动确定正确答案",
                "prompt": "请人工核对各选项与证据链；确认后再提交最终选项。",
                "margin": round(margin, 6),
            }
    if selected_index is None and human_in_loop is None:
        human_in_loop = {
            "status": "pending",
            "reason": "没有检索到能够支持任一选项的证据",
            "prompt": "请补充文件、条款或人工确认正确选项。",
        }
    if human_in_loop is not None:
        human_in_loop["options"] = assessments
    return ChoiceAgentResult("multiple_choice", question, choices[:4], option_hits, assessments, selected_index, human_in_loop)
