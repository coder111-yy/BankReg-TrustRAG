from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError


ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class LLMClientConfig:
    provider: str = "none"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 45.0
    max_tokens: int = 1600
    max_retries: int = 2

    @property
    def enabled(self) -> bool:
        return bool(
            self.provider.lower() not in {"", "none", "disabled"}
            and self.model
            and self.base_url
        )


@dataclass(frozen=True)
class StructuredLLMResult:
    status: str
    value: BaseModel | None = None
    attempts: int = 0
    error: str | None = None
    errors: tuple[str, ...] = ()


class LLMClient:
    """Small OpenAI-compatible client shared by planning and generation."""

    def __init__(self, config: LLMClientConfig):
        self.config = config
        self._json_schema_supported: bool | None = None

    @classmethod
    def from_settings(cls, settings: Any) -> "LLMClient":
        return cls(LLMClientConfig(
            provider=str(getattr(settings, "llm_provider", "none") or "none"),
            model=getattr(settings, "llm_model", None),
            base_url=getattr(settings, "llm_base_url", None),
            api_key=getattr(settings, "llm_api_key", None),
            timeout_seconds=float(getattr(settings, "llm_timeout_seconds", 45.0)),
            max_tokens=int(getattr(settings, "llm_max_tokens", 1600)),
            max_retries=max(0, int(getattr(settings, "llm_max_retries", 2))),
        ))

    @classmethod
    def from_planner_settings(cls, settings: Any) -> "LLMClient":
        """Build a dedicated planner client, preferring a light model config."""
        return cls(LLMClientConfig(
            provider=str(getattr(settings, "llm_provider", "none") or "none"),
            model=(
                getattr(settings, "planner_model", None)
                or getattr(settings, "llm_model", None)
            ),
            base_url=(
                getattr(settings, "planner_base_url", None)
                or getattr(settings, "llm_base_url", None)
            ),
            api_key=(
                getattr(settings, "planner_api_key", None)
                or getattr(settings, "llm_api_key", None)
            ),
            timeout_seconds=float(getattr(settings, "llm_planner_timeout_seconds", 60.0)),
            max_tokens=int(getattr(settings, "llm_planner_max_tokens", 2500)),
            # Planner is deliberately bounded to one generation plus one JSON
            # repair request, independent of the answer-generator retry count.
            max_retries=1,
        ))

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def structured(
        self,
        messages: list[dict[str, str]],
        output_model: type[ModelT],
        *,
        temperature: float,
        max_tokens: int | None = None,
        timeout_seconds: float | None = None,
        prefer_json_schema: bool = True,
        max_attempts: int | None = None,
    ) -> StructuredLLMResult:
        if not self.enabled:
            return StructuredLLMResult("disabled", error="llm_disabled")

        validation_error: str | None = None
        errors: list[str] = []
        schema_text = json.dumps(output_model.model_json_schema(), ensure_ascii=False)
        total_attempts = max(1, int(max_attempts or (self.config.max_retries + 1)))
        for attempt in range(1, total_attempts + 1):
            use_native_schema = (
                prefer_json_schema
                and attempt == 1
                and self._json_schema_supported is not False
            )
            request_messages = list(messages)
            if validation_error or not use_native_schema:
                request_messages.append({
                    "role": "user",
                    "content": (
                        "请只返回一个符合JSON Schema的JSON对象，"
                        "不要使用Markdown代码块、不要增加Schema之外的字段。"
                        f"校验错误：{(validation_error or 'provider_json_object_compatibility_mode')[:500]}\n"
                        f"必须严格符合的JSON Schema：{schema_text}"
                    ),
                })
            response_format: dict[str, Any]
            if use_native_schema:
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": output_model.__name__,
                        "strict": True,
                        "schema": output_model.model_json_schema(),
                    },
                }
            else:
                # Some local OpenAI-compatible servers implement json_object
                # but not json_schema. Pydantic remains the authority here.
                response_format = {"type": "json_object"}
            try:
                text = self._post(
                    request_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    timeout_seconds=timeout_seconds,
                    disable_thinking=_is_deepseek(self.config),
                )
                payload = _decode_structured_payload(text)
                value = output_model.model_validate(payload)
                if use_native_schema:
                    self._json_schema_supported = True
                return StructuredLLMResult("ok", value=value, attempts=attempt, errors=tuple(errors))
            except json.JSONDecodeError as exc:
                validation_error = f"invalid_json:{exc.msg}"
            except ValidationError as exc:
                validation_error = f"schema_validation:{exc}"
            except httpx.HTTPStatusError as exc:
                if use_native_schema and exc.response.status_code in {400, 404, 415, 422}:
                    self._json_schema_supported = False
                validation_error = type(exc).__name__
            except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
                validation_error = f"{type(exc).__name__}:{str(exc)[:200]}"
            errors.append(validation_error)
        return StructuredLLMResult(
            "error",
            attempts=total_attempts,
            error=validation_error or "structured_output_failed",
            errors=tuple(errors),
        )

    def _post(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int | None,
        response_format: dict[str, Any] | None = None,
        timeout_seconds: float | None = None,
        disable_thinking: bool = False,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "temperature": temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "messages": messages,
        }
        if response_format:
            payload["response_format"] = response_format
        # Structured planning needs a short final JSON object, not a hidden
        # reasoning trace.  DeepSeek-compatible endpoints support this switch;
        # do not send the provider-specific field to unrelated OpenAI servers.
        if disable_thinking and _is_deepseek(self.config):
            payload["thinking"] = {"type": "disabled"}
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        response = httpx.post(
            _chat_completions_url(str(self.config.base_url)),
            headers=headers,
            json=payload,
            timeout=timeout_seconds or self.config.timeout_seconds,
        )
        response.raise_for_status()
        return _response_text(response.json())


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base if base.endswith("/chat/completions") else f"{base}/chat/completions"


def _is_deepseek(config: LLMClientConfig) -> bool:
    provider = config.provider.strip().lower()
    base_url = (config.base_url or "").lower()
    return provider == "deepseek" or "api.deepseek.com" in base_url


def _response_text(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("missing_choices")
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        content = "".join(
            str(part.get("text") or part.get("content") or "")
            for part in content
            if isinstance(part, dict)
        )
    # A few OpenAI-compatible gateways return completion-style ``text`` even
    # when the request uses the chat endpoint.  Accept it without ever using
    # ``reasoning_content`` as an answer source.
    if not content:
        content = choice.get("text") or choice.get("delta", {}).get("content")
    text = str(content or "").strip()
    if not text:
        raise ValueError("empty_model_content")
    return text


def _decode_structured_payload(text: str) -> Any:
    """Decode common model-wrapped JSON without weakening schema validation.

    OpenAI-compatible providers generally honour ``response_format=json_object``,
    but some model deployments still wrap the object in Markdown or emit a
    Python-dict spelling (single quotes/``True``/``None``).  The old direct
    ``json.loads`` call treated those harmless transport variations as a
    planner failure.  We extract one balanced object and use
    ``ast.literal_eval`` only as a safe compatibility parser; the caller still
    validates the resulting value against the strict Pydantic model.
    """
    candidates = _structured_payload_candidates(text)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
        try:
            return ast.literal_eval(candidate)
        except (SyntaxError, ValueError, MemoryError) as exc:
            last_error = exc
        # A few providers omit quotes around simple object keys. Repair only
        # that syntactic form; values are never rewritten or inferred.
        repaired = re.sub(
            r"([{,]\s*)([A-Za-z_][A-Za-z0-9_-]*)\s*:",
            r'\1"\2":',
            candidate,
        )
        if repaired != candidate:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError as exc:
                last_error = exc
            try:
                return ast.literal_eval(repaired)
            except (SyntaxError, ValueError, MemoryError) as exc:
                last_error = exc
    if isinstance(last_error, json.JSONDecodeError):
        raise last_error
    raise json.JSONDecodeError("Expecting JSON object", text, 0)


def _structured_payload_candidates(text: str) -> list[str]:
    cleaned = text.strip()
    candidates: list[str] = []
    if cleaned:
        candidates.append(cleaned)
    # Remove only a surrounding Markdown fence; balanced extraction below also
    # handles prose such as "Here is the JSON:" and <think> wrappers.
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
    if fenced and fenced not in candidates:
        candidates.append(fenced)
    extracted = _extract_balanced_object(cleaned)
    if extracted and extracted not in candidates:
        candidates.append(extracted)
    return candidates


def _extract_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None
