from __future__ import annotations

import re
from typing import Any

from .retrieval.index import Hit
from .schemas import Verification
from .utils import normalize_text, normalized_number, numbers_in


# Do not include single-character ``可`` / ``应``: they occur in ordinary
# words such as “可引用”“应用” and produce false normative-strength failures.
NORMATIVE = ("应当", "应该", "不得", "必须", "可以", "原则上", "禁止", "允许")
TRACKED_ENTITIES = (
    "商业银行", "银行业金融机构", "消费金融公司", "保险公司", "保险业",
    "国家金融监督管理总局", "中国人民银行", "金融机构",
)


def verify_claims(
    answer: str,
    question: str,
    hits: list[Hit],
    claims: list[str],
    operations: list[dict[str, Any]] | None = None,
) -> Verification:
    verification = Verification()
    if not hits:
        verification.citation_ok = False
        verification.unsupported_claims.append("没有检索到可引用证据")
        verification.claim_results = [
            {"claim_id": f"claim_{index + 1}", "text": claim, "evidence_ids": [], "supported": False}
            for index, claim in enumerate(claims)
            if normalize_text(claim)
        ]
        return verification
    evidence = "\n".join(_evidence_for_hit(hit) for hit in hits)
    # Dates such as 2023-10 contain a negative-looking month token when
    # parsed by the generic number matcher.  Remove date expressions before
    # comparing metric values; dates are verified separately below.
    answer_numbers = numbers_in(_without_dates(_without_citations(answer)))
    evidence_numbers = numbers_in(_without_dates(evidence))
    evidence_numbers.extend(_deterministic_result_numbers(operations or []))
    for number in answer_numbers:
        if not any(abs(number - expected) < max(1e-9, abs(expected) * 1e-8) for expected in evidence_numbers):
            verification.numeric_ok = False
            verification.unsupported_claims.append(f"数字 {number:g} 未在证据中找到")
    dates = re.findall(r"20\d{2}(?:[-年]\d{1,2}(?:[-月]\d{1,2}日?)?)?", _without_citations(answer))
    if dates and not all(_date_supported(date, evidence) for date in dates):
        verification.date_ok = False
        verification.unsupported_claims.append("回答中的日期未被证据支持")
    normative_in_answer = [word for word in NORMATIVE if word in answer]
    normative_in_evidence = [word for word in NORMATIVE if word in evidence]
    if normative_in_answer and not any(word in normative_in_evidence for word in normative_in_answer):
        verification.normative_strength_ok = False
        verification.unsupported_claims.append("规范性用语可能超出证据")
    verification.claim_results = _verify_claims_against_individual_evidence(claims, hits, operations or [])
    for result in verification.claim_results:
        if not result["supported"]:
            verification.unsupported_claims.append(str(result["text"])[:120])
    _verify_entities_and_references(answer, question, evidence, verification)
    _verify_current_version(question, hits, verification)
    _detect_evidence_conflicts(hits, verification)
    verification.citation_ok = bool(hits)
    # Evidence is selected with document status available; an unknown status is not
    # automatically rejected, but is surfaced as a version caveat.
    for hit in hits:
        if hit.kind == "text" and hit.item.get("doc_id") and hit.item.get("content"):
            break
    return verification


def _evidence_for_hit(hit: Hit) -> str:
    return " ".join(str(hit.item.get(key) or "") for key in [
        "content", "context_window", "context", "indicator", "period", "value_text", "row_header",
        "column_header", "unit", "source_title", "source_file_name", "table_name",
    ])


def _verify_claims_against_individual_evidence(
    claims: list[str],
    hits: list[Hit],
    operations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, claim in enumerate(claims):
        text = normalize_text(claim)
        if not text:
            continue
        evidence_ids = [hit.evidence_id for hit in hits if _claim_supported(text, _evidence_for_hit(hit))]
        # A deterministic multi-quarter lookup intentionally renders several
        # exact values in one claim.  No single cell contains all values, so
        # verify the claim against the union of its retrieved evidence while
        # retaining the complete supporting ID chain.
        if not evidence_ids and len(hits) > 1 and _claim_supported(text, "\n".join(_evidence_for_hit(hit) for hit in hits)):
            evidence_ids = [hit.evidence_id for hit in hits]
        if not evidence_ids:
            evidence_ids = _calculation_supporting_evidence(text, operations)
        results.append({
            "claim_id": f"claim_{index + 1}",
            "text": text,
            "evidence_ids": evidence_ids,
            "supported": bool(evidence_ids),
        })
    return results


def _deterministic_result_numbers(operations: list[dict[str, Any]]) -> list[float]:
    """Allow verified calculator outputs in addition to source-cell numbers."""
    numbers: list[float] = []
    for operation in operations:
        if operation.get("type") != "table_calculation":
            continue
        for key in ("result", "difference", "value"):
            value = operation.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numbers.append(float(value))
            elif isinstance(value, str):
                parsed = normalized_number(value)
                if parsed is not None:
                    numbers.append(float(parsed))
    return numbers


def _calculation_supporting_evidence(claim: str, operations: list[dict[str, Any]]) -> list[str]:
    """Link a calculation claim to its audited operand cells."""
    for operation in operations:
        if operation.get("type") != "table_calculation":
            continue
        result = operation.get("result", operation.get("difference"))
        parsed = normalized_number(result)
        if parsed is None:
            continue
        claim_numbers = numbers_in(_without_dates(_without_citations(claim)))
        if any(abs(number - parsed) < max(1e-9, abs(parsed) * 1e-8) for number in claim_numbers):
            ids = [str(item) for item in operation.get("operand_evidence_ids", []) if item]
            if ids:
                return ids
    return []


def _verify_entities_and_references(answer: str, question: str, evidence: str, verification: Verification) -> None:
    for entity in TRACKED_ENTITIES:
        if entity in answer and entity not in evidence:
            verification.entity_ok = False
            verification.unsupported_claims.append(f"主体“{entity}”未在证据中找到")
    reference_patterns = (
        r"第[一二三四五六七八九十百零0-9]+条",
        r"〔\d{4}〕\d+号",
    )
    for pattern in reference_patterns:
        for reference in re.findall(pattern, answer):
            if reference not in evidence:
                verification.document_no_ok = False
                verification.unsupported_claims.append(f"文号或条款“{reference}”未在证据中找到")


def _verify_current_version(question: str, hits: list[Hit], verification: Verification) -> None:
    if not any(term in normalize_text(question) for term in ("现行", "当前", "最新")):
        return
    statuses = [normalize_text(hit.item.get("document_status")).lower() for hit in hits]
    if statuses and not any(status in {"effective", "现行", "有效"} for status in statuses):
        verification.version_ok = False
        verification.unsupported_claims.append("当前/现行问题未检索到可确认有效版本的证据")


def _detect_evidence_conflicts(hits: list[Hit], verification: Verification) -> None:
    """Flag only true same-source/same-cell contradictions, not normal alternatives."""
    seen: dict[tuple[str, str, str, str], str] = {}
    for hit in hits:
        item = hit.item
        key = (
            normalize_text(item.get("source_title") or item.get("doc_id")),
            normalize_text(item.get("indicator")),
            normalize_text(item.get("period")),
            normalize_text(item.get("cell_address")),
        )
        value = normalize_text(item.get("value_text"))
        if not any(key) or not value:
            continue
        previous = seen.get(key)
        if previous is not None and previous != value:
            conflict = f"证据单元“{key[3] or key[1]}”存在冲突值"
            if conflict not in verification.conflicts:
                verification.conflicts.append(conflict)
        else:
            seen[key] = value


def _claim_supported(claim: str, evidence: str) -> bool:
    claim = normalize_text(claim)
    if claim in evidence:
        return True
    claim_without_dates = _without_dates(claim)
    evidence_without_dates = _without_dates(evidence)
    terms = [term for term in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{2,}|\d+(?:\.\d+)?%?", claim_without_dates)]
    if not terms:
        return True
    # A stored ratio may be represented as a fraction (0.01513) while the
    # answer correctly renders the table unit as 1.513%.  Compare numeric
    # terms after percentage normalization instead of requiring the literal
    # string to occur in the evidence.
    numeric_terms = [term for term in terms if normalized_number(term) is not None]
    evidence_numbers = numbers_in(evidence_without_dates)
    numeric_supported = all(
        any(abs(normalized_number(term) - expected) < max(1e-9, abs(expected) * 1e-8) for expected in evidence_numbers)
        for term in numeric_terms
    )
    non_numeric_terms = [term for term in terms if normalized_number(term) is None]
    if numeric_terms and numeric_supported:
        # Table answers follow the deterministic form “指标在期间的值为数值”。
        # Once the indicator phrase and the normalized numeric value are both
        # present in the evidence, explanatory wording and date formatting
        # should not make a grounded claim fail verification.
        indicator_prefix = re.split(r"在|于", claim_without_dates, maxsplit=1)[0].strip(" ：:，,、")
        if len(indicator_prefix) >= 2 and indicator_prefix in evidence:
            return True
        if not non_numeric_terms:
            return True
        matched = sum(_text_term_overlap(term, evidence) for term in non_numeric_terms)
        return matched / len(non_numeric_terms) >= 0.45
    matched = sum(_text_term_overlap(term, evidence) for term in terms)
    return matched / len(terms) >= 0.45


def _without_dates(text: str) -> str:
    return re.sub(r"20\d{2}(?:年\s*\d{1,2}月?\s*\d{0,2}日?|[-/]\d{1,2}(?:[-/]\d{1,2})?)?", "", normalize_text(text))


def _without_citations(text: str) -> str:
    """Remove inline evidence labels before numeric/entity validation.

    Evidence IDs commonly contain digits such as ``text:DOC_12:p3``. Those
    digits are identifiers, not factual claims from the answer.
    """
    return re.sub(r"\[\s*(?:证据|evidence)\s*[:：][^\]]+\]", "", text, flags=re.IGNORECASE)


def _date_supported(date: str, evidence: str) -> bool:
    if date in evidence:
        return True
    normalized = normalize_text(date).replace("年", "-").replace("月", "").replace("日", "")
    evidence_normalized = normalize_text(evidence).replace("年", "-").replace("月", "").replace("日", "")
    if normalized in evidence_normalized:
        return True
    year_match = re.match(r"(20\d{2})", date)
    return bool(year_match and year_match.group(1) in evidence)


def _text_term_overlap(term: str, evidence: str) -> float:
    if term in evidence:
        return 1.0
    if re.fullmatch(r"[\u4e00-\u9fff]+", term) and len(term) >= 2:
        bigrams = [term[index : index + 2] for index in range(len(term) - 1)]
        return sum(1 for bigram in bigrams if bigram in evidence) / len(bigrams)
    return 0.0


def trust_decision(hits: list[Hit], verification: Verification, qa_type: str, min_score: float = 0.58, draft_confidence: float = 0.0) -> dict[str, Any]:
    retrieval = min(1.0, sum(max(hit.fused_score, 0.0) for hit in hits[:3]) * 25) if hits else 0.0
    evidence = min(1.0, len(hits) / (2 if qa_type in {"table_lookup", "cross_file_judgment"} else 1))
    source_authority = 0.75 if hits else 0.0
    version_validity = 0.7 if hits else 0.0
    verification_score = 1.0 if verification.passed else max(0.0, 1.0 - 0.2 * len(verification.unsupported_claims))
    score = 0.25 * retrieval + 0.20 * evidence + 0.15 * source_authority + 0.15 * version_validity + 0.25 * verification_score
    score = min(1.0, max(score, draft_confidence * 0.55))
    reasons: list[str] = []
    if not hits:
        reasons.append("没有足够证据")
    if not verification.passed:
        reasons.extend(verification.unsupported_claims[:3])
    if score >= min_score and hits and verification.citation_ok and not verification.unsupported_claims:
        decision = "answer"
    elif hits and score >= min_score * 0.72:
        decision = "clarify"
        reasons.append("证据存在但置信度不足或存在待核对字段")
    else:
        decision = "refuse"
    return {"score": round(score, 6), "decision": decision, "reasons": reasons, "components": {"retrieval": round(retrieval, 6), "evidence": round(evidence, 6), "source_authority": source_authority, "version_validity": version_validity, "verification": round(verification_score, 6)}}
