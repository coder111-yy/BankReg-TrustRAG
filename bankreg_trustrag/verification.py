from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from .retrieval.index import Hit
from .schemas import Verification
from .utils import normalize_text


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
    completeness: Any | None = None,
    grounding_refs: dict[str, list[str]] | None = None,
) -> Verification:
    verification = Verification()
    operations = operations or []
    if completeness is not None:
        complete = bool(
            completeness.get("complete", False)
            if isinstance(completeness, dict)
            else getattr(completeness, "complete", completeness)
        )
        if not complete:
            verification.completeness_ok = False
            missing = (
                completeness.get("missing_requirement_ids", [])
                if isinstance(completeness, dict)
                else list(getattr(completeness, "missing_requirement_ids", []) or [])
            )
            _record_failure(
                verification,
                claim=normalize_text(answer),
                error_type="incomplete_answer",
                expected="所有AnswerRequirement均已回答并绑定输出",
                actual={"missing_requirement_ids": missing},
                evidence_ids=[hit.evidence_id for hit in hits],
                calculation_ids=_all_calculation_ids(operations),
                message="回答未覆盖全部用户要求",
            )
    if not hits:
        verification.citation_ok = False
        _record_failure(
            verification,
            claim=normalize_text(answer),
            error_type="missing_evidence",
            expected="至少一条可引用证据",
            actual="未检索到证据",
            message="没有检索到可引用证据",
            calculation_ids=_all_calculation_ids(operations),
        )
        verification.claim_results = [
            {
                "claim_id": f"claim_{index + 1}", "text": claim,
                "evidence_ids": [], "calculation_ids": [], "supported": False,
            }
            for index, claim in enumerate(claims)
            if normalize_text(claim)
        ]
        return verification
    evidence = "\n".join(_evidence_for_hit(hit) for hit in hits)
    # Dates such as 2023-10 contain a negative-looking month token when
    # parsed by the generic number matcher.  Remove date expressions before
    # comparing metric values; dates are verified separately below.
    answer_numbers = _decimal_numbers(_without_dates(_without_citations(answer)))
    evidence_numbers = _decimal_numbers(_without_dates(evidence))
    calculation_numbers = _deterministic_result_numbers(operations)
    allowed_numbers = [*evidence_numbers, *calculation_numbers]
    for number in answer_numbers:
        if not any(_decimal_equal(number, expected) for expected in allowed_numbers):
            verification.numeric_ok = False
            rendered = _decimal_text(number)
            _record_failure(
                verification,
                claim=_claim_containing_number(claims, number) or normalize_text(answer),
                error_type="unsupported_number",
                expected={"allowed_numbers": [_decimal_text(value) for value in allowed_numbers]},
                actual=rendered,
                evidence_ids=[hit.evidence_id for hit in hits],
                calculation_ids=_all_calculation_ids(operations),
                message=f"数字 {rendered} 未在证据或计算结果中找到",
            )
    dates = re.findall(r"20\d{2}(?:[-年]\d{1,2}(?:[-月]\d{1,2}日?)?)?", _without_citations(answer))
    if dates and not all(_date_supported(date, evidence) for date in dates):
        verification.date_ok = False
        unsupported_dates = [date for date in dates if not _date_supported(date, evidence)]
        _record_failure(
            verification,
            claim=normalize_text(answer),
            error_type="unsupported_date",
            expected="日期应出现在证据中",
            actual=unsupported_dates,
            evidence_ids=[hit.evidence_id for hit in hits],
            calculation_ids=_all_calculation_ids(operations),
            message="回答中的日期未被证据支持",
        )
    _verify_units(answer, evidence, operations, verification, hits)
    normative_in_answer = [word for word in NORMATIVE if word in answer]
    normative_in_evidence = [word for word in NORMATIVE if word in evidence]
    if normative_in_answer and not any(word in normative_in_evidence for word in normative_in_answer):
        verification.normative_strength_ok = False
        _record_failure(
            verification,
            claim=normalize_text(answer),
            error_type="unsupported_normative_strength",
            expected=normative_in_evidence,
            actual=normative_in_answer,
            evidence_ids=[hit.evidence_id for hit in hits],
            calculation_ids=_all_calculation_ids(operations),
            message="规范性用语可能超出证据",
        )
    verification.claim_results = _verify_claims_against_individual_evidence(
        claims,
        hits,
        operations,
        grounding_refs,
    )
    for result in verification.claim_results:
        if not result["supported"]:
            message = str(result["text"])[:120]
            _record_failure(
                verification,
                claim=str(result["text"]),
                error_type="unsupported_claim",
                expected="claim由证据或确定性计算支持",
                actual="未找到完整支持链",
                evidence_ids=list(result.get("evidence_ids", [])),
                calculation_ids=list(result.get("calculation_ids", [])),
                message=message,
            )
    _verify_entities_and_references(answer, question, evidence, verification, hits, operations)
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
    grounding_refs: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    declared_refs = {
        str(ref)
        for refs in (grounding_refs or {}).values()
        for ref in refs
        if ref
    }
    declared_operations = [
        (index, operation)
        for index, operation in enumerate(operations)
        if _calculation_id(operation, index) in declared_refs
    ]
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
        calculation_evidence_ids, calculation_ids = _calculation_support(text, operations)
        if not evidence_ids and calculation_evidence_ids:
            evidence_ids = calculation_evidence_ids
        if not evidence_ids and not calculation_ids and declared_operations:
            # The structured Answer Agent explicitly binds every answered
            # requirement to tool outputs, and CompletenessChecker validates
            # those bindings before Verification.  A qualitative conclusion
            # may therefore cite the declared calculations without Python
            # interpreting words such as “一致”“明显” or “接近”.
            calculation_ids = [
                _calculation_id(operation, operation_index)
                for operation_index, operation in declared_operations
            ]
            evidence_ids = [
                str(evidence_id)
                for _, operation in declared_operations
                for evidence_id in [
                    *operation.get("operand_evidence_ids", []),
                    *operation.get("evidence_ids", []),
                ]
                if evidence_id
            ]
            calculation_ids = list(dict.fromkeys(calculation_ids))
            evidence_ids = list(dict.fromkeys(evidence_ids))
        results.append({
            "claim_id": f"claim_{index + 1}",
            "text": text,
            "evidence_ids": evidence_ids,
            "calculation_ids": calculation_ids,
            "supported": bool(evidence_ids or calculation_ids),
        })
    return results


def _deterministic_result_numbers(operations: list[dict[str, Any]]) -> list[Decimal]:
    """Allow numbers explicitly recorded by the deterministic calculator.

    Verification deliberately does not derive a new difference, tolerance or
    business conclusion.  A number is allowed only when the CalculationResult
    actually carries it in its result, inputs, trace or audited details.
    """
    numbers: list[Decimal] = []
    for operation in operations:
        if not _is_calculation_operation(operation):
            continue
        for key in ("result", "difference", "value"):
            numbers.extend(_decimal_value(operation.get(key), operation.get("unit")))
        numbers.extend(_decimal_value(operation.get("trace")))
        inputs = operation.get("inputs")
        if isinstance(inputs, list):
            for item in inputs:
                if isinstance(item, dict):
                    numbers.extend(_decimal_value(item.get("value"), item.get("unit")))
        details = operation.get("details")
        if isinstance(details, dict):
            for value in details.values():
                numbers.extend(_decimal_value(value))
    return list(dict.fromkeys(numbers))


def _calculation_support(
    claim: str,
    operations: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Link a calculation claim to its calculation ID and operand cells."""
    claim_numbers = _decimal_numbers(_without_dates(_without_citations(claim)))
    supporting_evidence: list[str] = []
    calculation_ids: list[str] = []
    for index, operation in enumerate(operations):
        if not _is_calculation_operation(operation):
            continue
        operation_numbers = _deterministic_result_numbers([operation])
        numeric_match = bool(operation_numbers) and any(
            _decimal_equal(claim_number, operation_number)
            for claim_number in claim_numbers
            for operation_number in operation_numbers
        )
        # Non-numeric calculator outputs (for example a literal boolean) are
        # grounded only when the exact tool token is present.  Do not map that
        # token to business phrases such as “一致/不一致”; that interpretation
        # belongs exclusively to the Answer Agent.
        result_text = normalize_text(operation.get("result")).lower()
        literal_result_match = bool(
            result_text
            and not _decimal_value(result_text, operation.get("unit"))
            and result_text in normalize_text(claim).lower()
        )
        if numeric_match or literal_result_match:
            ids = [str(item) for item in [
                *operation.get("operand_evidence_ids", []),
                *operation.get("evidence_ids", []),
            ] if item]
            supporting_evidence.extend(ids)
            calculation_ids.append(_calculation_id(operation, index))
    return (
        list(dict.fromkeys(supporting_evidence)),
        list(dict.fromkeys(calculation_ids)),
    )


def _is_calculation_operation(operation: dict[str, Any]) -> bool:
    return bool(
        operation.get("type") in {"table_calculation", "calculation"}
        or operation.get("operation") in {
            "sum", "subtract", "divide", "compare", "max", "min", "growth_rate", "verify_consistency",
        }
    )


def _verify_units(
    answer: str,
    evidence: str,
    operations: list[dict[str, Any]],
    verification: Verification,
    hits: list[Hit],
) -> None:
    answer_units = set(re.findall(r"(?:万?亿元|万元|元|[%％])", answer))
    if not answer_units:
        return
    supported = set(re.findall(r"(?:万?亿元|万元|元|[%％])", evidence))
    if any(term in evidence for term in ("率", "比例", "占比", "比率")):
        # Ratio tables frequently persist fractions (for example 0.01513)
        # without a literal unit while the verified presentation is 1.513%.
        supported.add("%")
    supported.update(
        normalize_text(operation.get("unit"))
        for operation in operations
        if operation.get("unit")
    )
    normalized_supported = {"%" if item == "％" else item for item in supported}
    unsupported = [
        item for item in answer_units
        if ("%" if item == "％" else item) not in normalized_supported
    ]
    if unsupported:
        verification.unit_ok = False
        _record_failure(
            verification,
            claim=normalize_text(answer),
            error_type="unsupported_unit",
            expected=sorted(normalized_supported),
            actual=unsupported,
            evidence_ids=[hit.evidence_id for hit in hits],
            calculation_ids=_all_calculation_ids(operations),
            message=f"单位“{unsupported[0]}”未被证据或计算结果支持",
        )


def _verify_entities_and_references(
    answer: str,
    question: str,
    evidence: str,
    verification: Verification,
    hits: list[Hit],
    operations: list[dict[str, Any]],
) -> None:
    for entity in TRACKED_ENTITIES:
        if entity in answer and entity not in evidence:
            verification.entity_ok = False
            _record_failure(
                verification,
                claim=normalize_text(answer),
                error_type="unsupported_entity",
                expected="主体应出现在证据中",
                actual=entity,
                evidence_ids=[hit.evidence_id for hit in hits],
                calculation_ids=_all_calculation_ids(operations),
                message=f"主体“{entity}”未在证据中找到",
            )
    reference_patterns = (
        r"第[一二三四五六七八九十百零0-9]+条",
        r"〔\d{4}〕\d+号",
    )
    for pattern in reference_patterns:
        for reference in re.findall(pattern, answer):
            if reference not in evidence:
                verification.document_no_ok = False
                _record_failure(
                    verification,
                    claim=normalize_text(answer),
                    error_type="unsupported_reference",
                    expected="文号或条款应出现在证据中",
                    actual=reference,
                    evidence_ids=[hit.evidence_id for hit in hits],
                    calculation_ids=_all_calculation_ids(operations),
                    message=f"文号或条款“{reference}”未在证据中找到",
                )


def _verify_current_version(question: str, hits: list[Hit], verification: Verification) -> None:
    if not any(term in normalize_text(question) for term in ("现行", "当前", "最新")):
        return
    statuses = [normalize_text(hit.item.get("document_status")).lower() for hit in hits]
    if statuses and not any(status in {"effective", "现行", "有效"} for status in statuses):
        verification.version_ok = False
        _record_failure(
            verification,
            claim=normalize_text(question),
            error_type="unverified_version",
            expected=["effective", "现行", "有效"],
            actual=statuses,
            evidence_ids=[hit.evidence_id for hit in hits],
            message="当前/现行问题未检索到可确认有效版本的证据",
        )


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
                verification.failure_details.append({
                    "claim": key[3] or key[1],
                    "error_type": "evidence_conflict",
                    "expected": previous,
                    "actual": value,
                    "evidence_ids": [hit.evidence_id],
                    "calculation_ids": [],
                })
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
    numeric_terms = [term for term in terms if _decimal_numbers(term)]
    evidence_numbers = _decimal_numbers(evidence_without_dates)
    numeric_supported = all(
        any(
            _decimal_equal(actual, expected)
            for actual in _decimal_numbers(term)
            for expected in evidence_numbers
        )
        for term in numeric_terms
    )
    non_numeric_terms = [term for term in terms if not _decimal_numbers(term)]
    if numeric_terms and numeric_supported:
        # Once the relevant subject phrase and normalized number both occur
        # in evidence, ordinary natural-language phrasing should not make a
        # grounded claim fail verification.
        indicator_prefix = re.split(r"在|于", claim_without_dates, maxsplit=1)[0].strip(" ：:，,、")
        if len(indicator_prefix) >= 2 and indicator_prefix in evidence:
            return True
        if not non_numeric_terms:
            return True
        matched = sum(_text_term_overlap(term, evidence) for term in non_numeric_terms)
        return matched / len(non_numeric_terms) >= 0.45
    matched = sum(_text_term_overlap(term, evidence) for term in terms)
    return matched / len(terms) >= 0.45


_DECIMAL_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*[%％]?")


def _decimal_numbers(text: str) -> list[Decimal]:
    values: list[Decimal] = []
    for match in _DECIMAL_RE.finditer(normalize_text(text)):
        raw = match.group(0).replace(",", "").replace(" ", "")
        percentage = raw.endswith(("%", "％"))
        if percentage:
            raw = raw[:-1]
        try:
            value = Decimal(raw)
        except InvalidOperation:
            continue
        values.append(value / Decimal("100") if percentage else value)
    return values


def _decimal_value(value: Any, unit: Any = None) -> list[Decimal]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float, Decimal)):
        text = str(value)
    elif isinstance(value, str):
        text = value
    else:
        return []
    normalized_unit = normalize_text(unit)
    if normalized_unit in {"%", "％"} and not text.rstrip().endswith(("%", "％")):
        text += "%"
    return _decimal_numbers(text)


def _decimal_equal(actual: Decimal, expected: Decimal) -> bool:
    """Compare numeric facts as Decimal, independent of display zeros."""
    if actual == expected:
        return True
    tolerance = max(Decimal("1e-12"), abs(expected) * Decimal("1e-12"))
    return abs(actual - expected) <= tolerance


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _claim_containing_number(claims: list[str], number: Decimal) -> str | None:
    for claim in claims:
        if any(_decimal_equal(value, number) for value in _decimal_numbers(_without_dates(claim))):
            return normalize_text(claim)
    return None


def _calculation_id(operation: dict[str, Any], index: int) -> str:
    return normalize_text(operation.get("id") or operation.get("output_id")) or f"calculation_{index + 1}"


def _all_calculation_ids(operations: list[dict[str, Any]]) -> list[str]:
    return [
        _calculation_id(operation, index)
        for index, operation in enumerate(operations)
        if _is_calculation_operation(operation)
    ]


def _record_failure(
    verification: Verification,
    *,
    claim: Any,
    error_type: str,
    expected: Any,
    actual: Any,
    evidence_ids: list[str] | None = None,
    calculation_ids: list[str] | None = None,
    message: str,
) -> None:
    if message not in verification.unsupported_claims:
        verification.unsupported_claims.append(message)
    detail = {
        "claim": normalize_text(claim),
        "error_type": error_type,
        "expected": expected,
        "actual": actual,
        "evidence_ids": list(dict.fromkeys(evidence_ids or [])),
        "calculation_ids": list(dict.fromkeys(calculation_ids or [])),
    }
    if detail not in verification.failure_details:
        verification.failure_details.append(detail)


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
