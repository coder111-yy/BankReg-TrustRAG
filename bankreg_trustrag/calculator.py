from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from typing import Mapping

from .query_plan import (
    CalculationInput,
    CalculationResult,
    CalculationTask,
    RetrievalResult,
)
from .utils import normalize_text


class CalculationError(ValueError):
    pass


class Calculator:
    """Execute audited financial calculations with Decimal only."""

    def execute(
        self,
        task: CalculationTask,
        retrieval_results: Mapping[str, RetrievalResult],
        calculation_results: Mapping[str, CalculationResult],
    ) -> CalculationResult:
        refs = task.input_refs()
        inputs = [self._resolve(ref, retrieval_results, calculation_results) for ref in refs]
        decimals = [_decimal(item.value) for item in inputs]
        evidence_ids = list(dict.fromkeys(
            evidence_id for item in inputs for evidence_id in item.evidence_ids
        ))
        unit = _compatible_unit(inputs)
        details: dict[str, object] = {}

        with localcontext() as context:
            context.prec = 40
            if task.type == "sum":
                result = sum(decimals, Decimal("0"))
                symbol = " + "
            elif task.type == "subtract":
                result = decimals[0] - decimals[1]
                if bool(task.parameters.get("absolute", False)):
                    result = abs(result)
                    details["absolute"] = True
                    symbol = " - "
                else:
                    symbol = " - "
            elif task.type == "divide":
                if decimals[1] == 0:
                    raise CalculationError("division by zero")
                result = decimals[0] / decimals[1]
                unit = normalize_text(task.parameters.get("unit")) or None
                symbol = " / "
            elif task.type == "growth_rate":
                old_value, new_value = decimals
                if old_value == 0:
                    raise CalculationError("growth_rate old value cannot be zero")
                raw_rate = (new_value - old_value) / old_value
                unrounded_percentage = raw_rate * Decimal("100")
                decimal_places = max(0, min(int(task.parameters.get("decimal_places", 2)), 8))
                result = unrounded_percentage.quantize(
                    Decimal("1").scaleb(-decimal_places),
                    rounding=ROUND_HALF_UP,
                )
                unit = "%"
                details["raw_rate"] = _decimal_text(raw_rate)
                details["unrounded_percentage"] = _decimal_text(unrounded_percentage)
                details["decimal_places"] = decimal_places
                symbol = " growth_rate "
            elif task.type in {"max", "min"}:
                result = max(decimals) if task.type == "max" else min(decimals)
                selected_index = decimals.index(result)
                details["selected_ref"] = refs[selected_index]
                symbol = f" {task.type} "
            elif task.type == "compare":
                operator = str(task.parameters.get("operator") or "==")
                comparison = _compare(decimals[0], decimals[1], operator)
                result_text = "true" if comparison else "false"
                return CalculationResult(
                    id=task.output_id,
                    operation=task.type,
                    input_refs=refs,
                    inputs=inputs,
                    result=result_text,
                    unit=None,
                    trace=f"{_decimal_text(decimals[0])} {operator} {_decimal_text(decimals[1])} = {result_text}",
                    evidence_ids=evidence_ids,
                    details={"operator": operator},
                )
            elif task.type == "verify_consistency":
                tolerance = _decimal(task.parameters.get("tolerance", "0"))
                spread = max(decimals) - min(decimals)
                consistent = spread <= tolerance
                result_text = "true" if consistent else "false"
                return CalculationResult(
                    id=task.output_id,
                    operation=task.type,
                    input_refs=refs,
                    inputs=inputs,
                    result=result_text,
                    unit=None,
                    trace=f"spread {_decimal_text(spread)} <= tolerance {_decimal_text(tolerance)} = {result_text}",
                    evidence_ids=evidence_ids,
                    details={"spread": _decimal_text(spread), "tolerance": _decimal_text(tolerance)},
                )
            elif task.type == "none":
                result = decimals[0] if decimals else Decimal("0")
                symbol = ""
            else:  # pragma: no cover - protected by the Pydantic literal
                raise CalculationError(f"unsupported operation: {task.type}")

        result_text = _decimal_text(result)
        if task.type == "growth_rate":
            trace = (
                f"({_decimal_text(decimals[1])} - {_decimal_text(decimals[0])}) / "
                f"{_decimal_text(decimals[0])} = {result_text}%"
            )
        elif task.type in {"max", "min"}:
            trace = f"{task.type}({', '.join(_decimal_text(item) for item in decimals)}) = {result_text}"
        elif task.type == "none":
            trace = result_text
        else:
            trace = f"{symbol.join(_decimal_text(item) for item in decimals)} = {result_text}"
        return CalculationResult(
            id=task.output_id,
            operation=task.type,
            input_refs=refs,
            inputs=inputs,
            result=result_text,
            unit=unit,
            trace=trace,
            evidence_ids=evidence_ids,
            details=details,
        )

    @staticmethod
    def _resolve(
        ref: str,
        retrieval_results: Mapping[str, RetrievalResult],
        calculation_results: Mapping[str, CalculationResult],
    ) -> CalculationInput:
        if ref in calculation_results:
            result = calculation_results[ref]
            return CalculationInput(
                ref=ref,
                value=result.result,
                unit=result.unit,
                evidence_ids=result.evidence_ids,
            )
        retrieval = retrieval_results.get(ref)
        if retrieval is None or retrieval.status != "resolved" or retrieval.selected is None:
            raise CalculationError(f"input {ref} is not resolved")
        if retrieval.selected.value is None:
            raise CalculationError(f"input {ref} has no value")
        return CalculationInput(
            ref=ref,
            value=retrieval.selected.value,
            unit=retrieval.selected.unit,
            evidence_ids=retrieval.selected.evidence_ids,
        )


def _decimal(value: object) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise CalculationError(f"not a numeric value: {value}")
    text = normalize_text(value).strip().strip('"').replace(",", "")
    text = re.sub(r"\s*(?:亿元|万元|元|亿|万|%|％)$", "", text)
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        raise CalculationError(f"not a numeric value: {value}") from exc


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _compatible_unit(inputs: list[CalculationInput]) -> str | None:
    units = {normalize_text(item.unit) for item in inputs if normalize_text(item.unit)}
    if len(units) > 1:
        raise CalculationError(f"incompatible units: {sorted(units)}")
    return next(iter(units), None)


def _compare(left: Decimal, right: Decimal, operator: str) -> bool:
    operations = {
        "==": left == right,
        "!=": left != right,
        ">": left > right,
        ">=": left >= right,
        "<": left < right,
        "<=": left <= right,
    }
    if operator not in operations:
        raise CalculationError(f"unsupported comparison operator: {operator}")
    return operations[operator]
