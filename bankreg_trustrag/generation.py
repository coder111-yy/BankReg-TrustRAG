from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from .retrieval.index import Hit
from .schemas import ParsedQuery
from .utils import normalize_text


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "none"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 45.0
    max_tokens: int = 800
    temperature: float = 0.0
    max_context_chars: int = 12000

    @property
    def enabled(self) -> bool:
        return bool(
            self.provider.lower() not in {"", "none", "disabled"}
            and self.model
            and self.base_url
        )


@dataclass(frozen=True)
class GenerationResult:
    status: str
    answer: str | None = None
    context_evidence_ids: tuple[str, ...] = ()
    error: str | None = None


class GroundedGenerator:
    """Call an OpenAI-compatible chat endpoint with retrieved evidence only.

    The provider is opt-in.  With the default ``none`` provider, the service
    keeps its deterministic, locally verifiable answer path.  The generator
    never sends local paths or database records; only the selected evidence
    text and public source labels are placed in the prompt.
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    @classmethod
    def from_settings(cls, settings: Any) -> "GroundedGenerator":
        return cls(
            LLMConfig(
                provider=str(getattr(settings, "llm_provider", "none") or "none"),
                model=getattr(settings, "llm_model", None),
                base_url=getattr(settings, "llm_base_url", None),
                api_key=getattr(settings, "llm_api_key", None),
                timeout_seconds=float(getattr(settings, "llm_timeout_seconds", 45.0)),
                max_tokens=int(getattr(settings, "llm_max_tokens", 800)),
                temperature=float(getattr(settings, "llm_temperature", 0.0)),
                max_context_chars=int(getattr(settings, "llm_max_context_chars", 12000)),
            )
        )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def status(self) -> dict[str, Any]:
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "enabled": self.enabled,
        }

    def generate(
        self,
        question: str,
        parsed: ParsedQuery,
        hits: list[Hit],
        operations: list[dict[str, Any]],
    ) -> GenerationResult:
        context, evidence_ids = self.build_context(hits)
        if not self.enabled:
            return GenerationResult("disabled", context_evidence_ids=tuple(evidence_ids))
        if not context:
            return GenerationResult("error", context_evidence_ids=tuple(evidence_ids), error="empty_evidence_context")

        system_prompt = (
            "你是 BankReg-TrustRAG 的 Evidence-Grounded Answer Agent。只能根据 USER_QUESTION、"
            "EVIDENCE_CONTEXT 和 DETERMINISTIC_FACTS 回答；问题类型和意图字段只描述任务，不能作为事实来源。"
            "不得补充证据中没有的数字、日期、机构、文号、条款或规范性事实；不得自行检索、重新计算或修改工具数字。"
            "规范性强度必须忠实于证据，多个来源冲突时如实报告，证据不足时明确说明无法可靠判断。"
            "由你根据用户真正的问题自由决定回答句式、结构、段落、表格、结论顺序和详略，不套用固定回答模板。"
            "一致、接近、明显、增长或下降等业务语义由你根据证据和确定性结果作有依据的判断，Python工具不替你措辞；"
            "不得把内部true/false原样输出给用户。"
            "不要输出检索过程、Chain-of-Thought、信任分、耗时、文件路径、Evidence ID 或内部任务标记。"
        )
        user_prompt = "\n".join(
            [
                f"USER_QUESTION: {normalize_text(question)}",
                f"QUESTION_TYPE: {parsed.qa_type}",
                f"INTENT: {parsed.intent}",
                f"ANSWER_FORMAT: {parsed.answer_format}",
                f"REQUIREMENTS: {json.dumps(parsed.requirements, ensure_ascii=False, default=str)}",
                f"QUERY_ENTITIES: {json.dumps(parsed.entities, ensure_ascii=False, default=str)}",
                f"DETERMINISTIC_FACTS: {json.dumps(_safe_operations(operations), ensure_ascii=False, default=str)}",
                "EVIDENCE_CONTEXT:",
                context,
            ]
        )
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        try:
            response = httpx.post(
                _chat_completions_url(str(self.config.base_url)),
                headers=headers,
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            answer = clean_user_answer(_response_text(body))
            if not answer:
                return GenerationResult("error", context_evidence_ids=tuple(evidence_ids), error="empty_model_answer")
            return GenerationResult("generated", answer=answer, context_evidence_ids=tuple(evidence_ids))
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            # Do not return exception text or request details to the user; the
            # deterministic fallback remains the auditable answer path.
            return GenerationResult(
                "error",
                context_evidence_ids=tuple(evidence_ids),
                error=type(exc).__name__,
            )

    def build_context(self, hits: list[Hit]) -> tuple[str, list[str]]:
        blocks: list[str] = []
        evidence_ids: list[str] = []
        used_chars = 0
        for index, hit in enumerate(hits[:8], 1):
            item = hit.item
            evidence_id = str(item.get("evidence_id") or hit.evidence_id)
            source = str(item.get("source_title") or item.get("source_file_name") or "未标注来源")
            location = _evidence_location(item, hit.kind)
            content = _evidence_content(item, hit.kind)
            # Evidence IDs remain in the structured response, but are not put
            # into the model prompt.  This prevents an internal trace token
            # from leaking into the user-facing answer.
            block = f"[证据 {index}]\nsource={source}\nlocation={location}\ncontent={content}"
            remaining = self.config.max_context_chars - used_chars
            if remaining <= 0:
                break
            block = block[:remaining]
            blocks.append(block)
            evidence_ids.append(evidence_id)
            used_chars += len(block)
        return "\n\n".join(blocks), evidence_ids


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/chat/completions") else f"{base}/chat/completions"


def _response_text(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") if isinstance(message, dict) else ""
    if isinstance(content, list):
        content = "".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    return str(content or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def clean_user_answer(answer: str) -> str:
    """Remove internal trace syntax before text reaches the answer panel.

    Evidence IDs remain available in the structured response and the evidence
    ledger.  They are implementation identifiers, so displaying them inline
    makes a concise answer harder to read without adding audit value.
    """
    text = str(answer or "")
    text = re.sub(
        r"\s*\[\s*(?:证据|evidence)\s*[:：][^\]\n]+\]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*(?:证据|evidence)\s*id\s*[:：]\s*(?:cell|text):[^\s，。；;]+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*\[?\s*(?:证据|evidence)\s*\d+\s*\]?", "", text, flags=re.IGNORECASE)
    # Collapse layout whitespace but keep Chinese punctuation as authored by
    # the model (NFKC would turn full-width punctuation into ASCII).
    lines = [re.sub(r"\s+", " ", line).strip(" -•") for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _evidence_location(item: dict[str, Any], kind: str) -> str:
    if kind == "table" or item.get("cell_address"):
        return " · ".join(str(value) for value in [item.get("sheet_name") or item.get("table_name") or "工作表", item.get("cell_address") or "未标注单元格"])
    return " · ".join(
        str(value)
        for value in [
            f"第{item['page']}页" if item.get("page") else None,
            f"段落{item['paragraph_no']}" if item.get("paragraph_no") else None,
        ]
        if value
    ) or "未标注段落"


def _evidence_content(item: dict[str, Any], kind: str) -> str:
    if kind == "table" or item.get("cell_address"):
        values = [
            item.get("indicator"), item.get("period"), item.get("row_header"),
            item.get("column_header"), item.get("value_text"), item.get("unit"), item.get("context"),
        ]
    else:
        values = [item.get("content"), item.get("context_window"), item.get("context")]
    return normalize_text(" | ".join(str(value) for value in values if value))


def _safe_operations(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {
        "type", "value", "values", "unit", "period", "formula", "threshold",
        "comparator", "confidence", "intent", "answer_format", "method",
        "selected_option", "selected_text", "comparison_summary", "data_source",
        "rule_source", "operation", "input_refs", "inputs", "result", "trace",
        "details", "difference",
    }
    safe_operations: list[dict[str, Any]] = []
    for operation in operations:
        safe = {
            key: _safe_operation_value(value)
            for key, value in operation.items()
            if key in allowed
        }
        safe_operations.append(safe)
    return safe_operations


def _safe_operation_value(value: Any) -> Any:
    """Remove internal evidence identifiers from facts sent to the LLM."""
    if isinstance(value, list):
        return [_safe_operation_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _safe_operation_value(item)
            for key, item in value.items()
            if key not in {"evidence_id", "evidence_ids", "cell", "cell_address"}
        }
    return value


def split_grounded_claims(answer: str) -> list[str]:
    """Split model prose for existing claim verification, dropping citations."""
    claims: list[str] = []
    for part in answer.replace("\r", "\n").split("\n"):
        for sentence in part.split("。"):
            cleaned = sentence.strip()
            cleaned = re.sub(r"\[\s*(?:证据|evidence)\s*[:：][^\]]+\]", "", cleaned, flags=re.IGNORECASE).strip(" 。；;\t")
            if cleaned:
                claims.append(cleaned + ("。" if sentence.strip().endswith("。") else ""))
    return claims
