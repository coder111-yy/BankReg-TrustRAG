from __future__ import annotations

import time
import uuid
import re
from pathlib import Path
from typing import Any, Callable

from .agent import build_agent_workflow, identify_intent, organize_evidence, run_choice_agent
from .config import Settings
from .generation import GroundedGenerator, split_grounded_claims
from .query import classify_question_scope, extract_inline_choices, parse_query
from .reasoning import AnswerDraft, reason
from .retrieval.index import HybridIndex
from .retrieval.bge import BGEConfig, BGEPipeline
from .schemas import QAResponse
from .storage import Store
from .verification import trust_decision, verify_claims


class TrustRAGService:
    def __init__(self, settings: Settings, store: Store | None = None):
        self.settings = settings
        self.store = store or Store(settings.db_path)
        self.generator = GroundedGenerator.from_settings(settings)
        if self.store.document_count() == 0 and (settings.artifact_dir / "documents.jsonl").exists():
            self.store.load_jsonl(settings.artifact_dir)
        self.semantic = BGEPipeline(
            BGEConfig(
                mode=settings.bge_mode,
                embedding_model=settings.bge_embedding_model,
                reranker_model=settings.bge_reranker_model,
                cache_dir=settings.bge_cache_dir,
                vector_dir=settings.bge_vector_dir,
                device=settings.bge_device,
                batch_size=settings.bge_batch_size,
                max_length=settings.bge_max_length,
                local_files_only=settings.bge_local_files_only,
                rerank_top_k=settings.bge_rerank_top_k,
            )
        )
        self.index = HybridIndex.from_store(self.store, semantic=self.semantic, vector_dir=settings.bge_vector_dir)

    def reload(self) -> None:
        self.store.load_jsonl(self.settings.artifact_dir)
        self.index = HybridIndex.from_store(self.store, semantic=self.semantic, vector_dir=self.settings.bge_vector_dir)

    def ask(
        self,
        question: str,
        choices: list[str] | None = None,
        qa_type: str | None = None,
        filters: dict[str, Any] | None = None,
        conversation_context: list[dict[str, Any]] | None = None,
        observer: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> QAResponse:
        started = time.perf_counter()
        def report(stage: str, **details: Any) -> None:
            if observer is not None:
                observer(stage, {"elapsed_ms": int((time.perf_counter() - started) * 1000), **details})

        report("understanding", label="正在理解问题与识别查询类型")
        question_for_retrieval, inline_choices = extract_inline_choices(question)
        effective_choices = [str(value).strip() for value in (choices or inline_choices) if str(value).strip()]
        question_for_analysis = _question_with_conversation_context(question_for_retrieval, conversation_context)
        parsed = parse_query(question_for_analysis, effective_choices)
        # Keep the actual user message as the auditable original query.  The
        # contextual form is used only to resolve follow-up references during
        # routing/retrieval; it is never promoted to regulatory evidence.
        parsed.original_query = question
        if qa_type:
            parsed.qa_type = qa_type  # type: ignore[assignment]
        # A multiple-choice stem may be generic while the regulatory domain
        # anchor appears only inside an option (for example, “商业银行不得…”).
        scope = classify_question_scope(" ".join([question_for_analysis, *effective_choices]))
        if not scope["in_scope"]:
            report("scope_guard", label="问题不在银行监管知识库范围内")
            return self._refuse_out_of_scope(
                question,
                question_for_retrieval,
                effective_choices,
                parsed,
                scope,
                started,
            )
        filters = {**(filters or {})}
        if parsed.entities.get("filenames"):
            filters["file_name"] = parsed.entities["filenames"]
        if parsed.entities.get("table_name"):
            # A question can name a workbook's logical table without giving
            # its generated filename.  Use the logical title as a strict
            # document scope so an insurance table cannot outrank the bank
            # regulatory table with the same indicator and period.
            filters["title"] = [parsed.entities["table_name"]]
        if parsed.entities.get("title_hints"):
            filters["title"] = list(dict.fromkeys([
                *(filters.get("title") or []),
                *parsed.entities["title_hints"],
            ]))
            # Evaluation rows sometimes provide an attachment short name,
            # while ingestion stores the generated filename with a numeric
            # prefix.  If that filename has no exact match, the quoted source
            # title is the safer fallback instead of returning zero hits.
            requested_files = filters.get("file_name") or []
            if requested_files:
                known_names = [str(doc.get("file_name") or "") for doc in getattr(self.index, "documents", [])]
                if known_names and not any(str(value).lower() in name.lower() for value in requested_files for name in known_names):
                    filters.pop("file_name", None)
        agent_workflow = build_agent_workflow(parsed, question_for_retrieval, effective_choices, filters)
        retrieval_query = question_for_analysis
        retrieval_anchors = [
            parsed.entities.get("indicator"),
            parsed.entities.get("table_name"),
            parsed.entities.get("period_normalized"),
            parsed.entities.get("quarter"),
            parsed.entities.get("row_label"),
            parsed.entities.get("column_label"),
        ]
        retrieval_query += " " + " ".join(str(anchor) for anchor in retrieval_anchors if anchor)
        # Options are intentionally not concatenated into this base query.  The
        # choice agent below retrieves each option independently.
        retrieval_k = max(self.settings.top_k, 32) if parsed.requires_table else self.settings.top_k
        report("retrieving", label="正在检索本地制度与统计资料", routes=["bm25", "metadata", "bge_vector"])
        if parsed.qa_type == "cross_file_judgment":
            # Cross-file questions have two independent scopes.  The named
            # workbook constrains the structured/statistical hop, but must not
            # constrain the regulatory-definition hop.
            table_hits = self.index.hybrid_search(retrieval_query, "table_lookup", retrieval_k, filters)
            rule_filters = {
                key: value
                for key, value in filters.items()
                if key not in {"title", "file_name"}
            }
            rule_query = " ".join(
                str(value)
                for value in [
                    parsed.entities.get("indicator"),
                    "监管制度 监管要求 监管阈值 计算公式 指标解释",
                    parsed.entities.get("period"),
                ]
                if value
            )
            rule_hits = self.index.hybrid_search(rule_query, "cross_file_judgment", max(retrieval_k, 16), rule_filters)
            # Formula definitions may be represented as structured Excel
            # cells rather than text paragraphs (for example, 2025 schedule
            # sheet ``指标解释!C6``).  Add a high-recall, exact-term table hop
            # so the definition is not lost to the cross-file reranker.
            year_match = re.search(r"20\d{2}", str(parsed.entities.get("period") or question))
            formula_hits = self.index.search_formula_evidence(
                str(parsed.entities.get("indicator") or ""),
                max(retrieval_k, 8),
                rule_filters,
                year_match.group(0) if year_match else None,
            )
            hits = _merge_hits(rule_hits, formula_hits, table_hits)
        else:
            hits = self.index.hybrid_search(retrieval_query, parsed.qa_type, retrieval_k, filters)
        choice_result = None
        if len(effective_choices) >= 2:
            report("choice_retrieval", label="正在分别核对每个选项的证据")
            choice_result = run_choice_agent(
                self.index,
                retrieval_query,
                effective_choices,
                parsed.qa_type,
                filters,
                max(self.settings.top_k, 8),
            )
            hits = _merge_hits(hits, choice_result.all_hits)
        _enrich_hits(self.index, hits)
        report(
            "evidence_selected",
            label="已完成重排并筛选最小证据集",
            documents=_retrieved_document_labels(hits),
            evidence_count=len(hits),
        )
        missing_year = _requested_year_missing(parsed.entities.get("years", []), hits)
        if missing_year:
            draft = AnswerDraft(
                f"知识库中没有找到与{missing_year}年相匹配的监管或统计资料，无法可靠回答。请核对年份，或提供具体文件名称。",
                [],
                [{"type": "refusal", "source": None, "reason": f"知识库缺少{missing_year}年证据"}],
            )
        else:
            report("reasoning", label="正在依据证据执行规则匹配与确定性计算")
            if choice_result is not None:
                draft = _choice_answer_draft(choice_result)
            else:
                draft = reason(question_for_retrieval, parsed.qa_type, None, hits)
        llm_generation = self.generator.status() if hasattr(self, "generator") else {"provider": "none", "enabled": False, "status": "disabled"}
        if choice_result is None and hits and not _has_terminal_operation(draft.operations):
            generator = getattr(self, "generator", None)
            if generator is not None and generator.enabled:
                report("generating", label="正在基于已检索证据生成回答")
                context_hits = _minimal_display_hits(hits, draft.operations, None)
                generated = generator.generate(question_for_retrieval, parsed, context_hits, draft.operations)
                llm_generation = {
                    **generator.status(),
                    "status": generated.status,
                    "context_evidence_ids": list(generated.context_evidence_ids),
                }
                generation_operation = {
                    "type": "llm_generation",
                    "provider": generator.config.provider,
                    "model": generator.config.model,
                    "status": generated.status,
                    "context_evidence_ids": list(generated.context_evidence_ids),
                }
                if generated.error:
                    generation_operation["error"] = generated.error
                if generated.answer:
                    candidate = AnswerDraft(
                        generated.answer,
                        split_grounded_claims(generated.answer),
                        [*draft.operations, generation_operation],
                    )
                    candidate_verification = verify_claims(candidate.answer, question, hits, candidate.claims)
                    if candidate_verification.passed:
                        generation_operation["status"] = "accepted"
                        draft = candidate
                        llm_generation["status"] = "accepted"
                    else:
                        generation_operation["status"] = "rejected_by_verification"
                        generation_operation["unsupported_claims"] = candidate_verification.unsupported_claims[:5]
                        draft.operations.append(generation_operation)
                else:
                    draft.operations.append(generation_operation)
        report("verifying", label="正在核验数值、实体、版本与证据支持")
        verification = verify_claims(draft.answer, question, hits, draft.claims)
        retry_record: dict[str, Any] | None = None
        if _should_retry_verification(verification, draft, choice_result, missing_year):
            retry_query = _verification_retry_query(parsed, question_for_retrieval)
            retry_hits = self.index.hybrid_search(retry_query, parsed.qa_type, max(retrieval_k, self.settings.top_k * 2), filters)
            hits = _merge_hits(hits, retry_hits)
            if choice_result is not None:
                # A selection question must retain option-wise retrieval after
                # a verification retry. Falling through to ``reason`` here
                # discarded the Choice Agent result and turned answerable
                # choices into generic clarification responses.
                choice_result = run_choice_agent(
                    self.index,
                    retry_query,
                    effective_choices,
                    parsed.qa_type,
                    filters,
                    max(self.settings.top_k * 2, 12),
                )
                hits = _merge_hits(hits, choice_result.all_hits)
            _enrich_hits(self.index, hits)
            retry_missing_year = _requested_year_missing(parsed.entities.get("years", []), hits)
            if retry_missing_year:
                draft = AnswerDraft(
                    f"知识库中没有找到与{retry_missing_year}年相匹配的监管或统计资料，无法可靠回答。请核对年份，或提供具体文件名称。",
                    [],
                    [{"type": "refusal", "source": None, "reason": f"知识库缺少{retry_missing_year}年证据"}],
                )
            else:
                draft = _choice_answer_draft(choice_result) if choice_result is not None else reason(question_for_retrieval, parsed.qa_type, None, hits)
            verification = verify_claims(draft.answer, question, hits, draft.claims)
            retry_record = {
                "type": "verification_retry",
                "query": retry_query,
                "reason": "初次答案存在未支持 Claim 或字段核验失败",
                "retrieved_evidence_ids": [hit.evidence_id for hit in retry_hits],
                "verification_passed": verification.passed,
            }
            draft.operations.append(retry_record)
        trust = trust_decision(hits, verification, parsed.qa_type, self.settings.min_trust, _draft_confidence(draft))
        refusal = next((operation for operation in draft.operations if operation.get("type") == "refusal"), None)
        clarification = next((operation for operation in draft.operations if operation.get("type") == "clarification"), None)
        human_in_loop = next((operation for operation in draft.operations if operation.get("type") == "human_in_loop"), None)
        if refusal is not None:
            trust["decision"] = "refuse"
            trust["score"] = min(trust["score"], 0.2)
            trust["components"]["evidence"] = 0.0
            trust.setdefault("reasons", []).append(str(refusal.get("reason") or "问题范围不明确，无法可靠回答"))
        elif clarification is not None:
            trust["decision"] = "clarify"
            if clarification.get("source"):
                trust.setdefault("reasons", []).append("已定位来源，但问题未指定具体指标")
            else:
                trust["score"] = min(trust["score"], 0.35)
                trust["components"]["evidence"] = 0.0
                trust.setdefault("reasons", []).append("问题需要补充信息，当前证据不能支持确定性结论")
        elif human_in_loop is not None:
            trust["decision"] = "clarify"
            trust["score"] = min(trust["score"], 0.35)
            trust["components"]["evidence"] = 0.0
            trust.setdefault("reasons", []).append("自动检索无法唯一确定选项，已转人工确认")
        display_hits = _minimal_display_hits(hits, draft.operations, clarification or refusal or human_in_loop)
        if trust["decision"] != "answer":
            if refusal is not None:
                pass
            elif trust["decision"] == "refuse":
                draft.answer = "证据不足或校验未通过，系统拒绝直接给出未经证实的结论。"
            elif human_in_loop is not None:
                pass
            elif not any(operation.get("type") == "clarification" for operation in draft.operations):
                draft.answer = "当前证据存在不确定性，请补充文件、版本、时间或业务场景后再查询。"
        trace_id = "trace_" + uuid.uuid4().hex[:16]
        latency = int((time.perf_counter() - started) * 1000)
        evidence = [hit.to_dict() for hit in display_hits]
        retrieval_routes = ["bm25", "metadata"]
        semantic = getattr(self, "semantic", None)
        if semantic is not None and semantic.enabled:
            retrieval_routes.append("bge_vector")
            retrieval_routes.append("bge_reranker")
        else:
            retrieval_routes.append("char_ngram_fallback")
        if parsed.requires_table:
            retrieval_routes.append("structured_table")
        if choice_result is not None:
            retrieval_routes.append("choice_agent")
        plan = {
            "original_query": parsed.original_query,
            "qa_type": parsed.qa_type,
            "entities": parsed.entities,
            "requires_table": parsed.requires_table,
            "requires_multi_hop": parsed.requires_multi_hop,
            "retrieval_routes": retrieval_routes,
            "model_status": self.index.model_status if hasattr(self.index, "model_status") else {"mode": "unknown"},
            "generation": llm_generation,
            "retrieved_evidence_ids": [hit.evidence_id for hit in hits],
            # A multi-quarter table lookup deliberately carries one exact cell
            # per quarter.  Do not truncate that evidence chain to three cells;
            # the same complete set is passed to the grounded LLM context.
            "minimal_evidence_ids": [hit.evidence_id for hit in display_hits],
            "operations": draft.operations,
            "scope": scope,
            "agent": identify_intent(question_for_retrieval, effective_choices, parsed.qa_type),
        }
        if choice_result is not None:
            plan["agent"] = choice_result.to_plan()
        minimal_ids = plan["minimal_evidence_ids"]
        _finalize_agent_workflow(agent_workflow, hits, minimal_ids, draft, verification, retry_record)
        plan["agent_workflow"] = agent_workflow
        response = QAResponse(draft.answer, parsed.qa_type, evidence, verification.to_dict(), trust, trace_id, latency, plan)
        self.store.save_qa(trace_id, question, parsed.qa_type, plan, [hit.evidence_id for hit in hits], draft.answer, trust["score"], verification.to_dict(), trust["decision"], latency)
        report("completed", label="回答与证据链已完成", latency_ms=latency)
        return response

    def _refuse_out_of_scope(
        self,
        original_question: str,
        question_for_retrieval: str,
        choices: list[str],
        parsed: Any,
        scope: dict[str, Any],
        started: float,
    ) -> QAResponse:
        """Return a clean refusal before unrelated retrieval can create a hit."""
        answer = "当前问题不在本服务的知识库范围内。请提交银行业监管制度、监管指标、统计报表或合规判断相关问题。"
        draft = AnswerDraft(
            answer,
            [],
            [{"type": "refusal", "source": None, "reason": scope.get("reason", "问题超出服务范围")}],
        )
        hits: list[Any] = []
        verification = verify_claims(draft.answer, original_question, hits, draft.claims)
        trust = trust_decision(hits, verification, parsed.qa_type, self.settings.min_trust, 0.0)
        trust["decision"] = "refuse"
        trust["score"] = 0.2
        trust["components"]["evidence"] = 0.0
        trust.setdefault("reasons", []).insert(0, str(scope.get("reason") or "问题超出服务范围"))
        trace_id = "trace_" + uuid.uuid4().hex[:16]
        latency = int((time.perf_counter() - started) * 1000)
        plan = {
            "original_query": parsed.original_query,
            "qa_type": parsed.qa_type,
            "entities": parsed.entities,
            "requires_table": False,
            "requires_multi_hop": False,
            "retrieval_routes": [],
            "model_status": self.index.model_status if hasattr(self.index, "model_status") else {"mode": "unknown"},
            "retrieved_evidence_ids": [],
            "minimal_evidence_ids": [],
            "operations": draft.operations,
            "scope": scope,
            "agent": identify_intent(question_for_retrieval, choices, parsed.qa_type),
        }
        workflow = build_agent_workflow(parsed, question_for_retrieval, choices)
        for task in workflow["tasks"]:
            task["status"] = "skipped" if task["id"] != "understand" else "completed"
        workflow["refusal"] = {"reason": scope.get("reason"), "stage": "scope_guard"}
        plan["agent_workflow"] = workflow
        response = QAResponse(answer, parsed.qa_type, [], verification.to_dict(), trust, trace_id, latency, plan)
        self.store.save_qa(trace_id, original_question, parsed.qa_type, plan, [], answer, trust["score"], verification.to_dict(), trust["decision"], latency)
        return response


def _choice_answer_draft(choice_result: Any) -> AnswerDraft:
    assessments = choice_result.assessments
    selected_index = choice_result.selected_index
    if selected_index is not None and selected_index < len(choice_result.choices):
        selected = assessments[selected_index]
        label = selected["label"]
        # The original option may strengthen a normative phrase relative to a
        # split clause (for example, an enumerated qualifying condition). The
        # answer therefore returns the verified option label and exposes the
        # exact option/evidence through the structured audit record instead of
        # restating a potentially stronger paraphrase as a generated claim.
        answer = f"选项 {label}：该选项已获得证据链支持。"
        return AnswerDraft(
            answer,
            [],
            [{
                "type": "choice_agent",
                "intent": "multiple_choice",
                "selected_option": label,
                "confidence": selected["score"],
                "option_assessments": assessments,
                "display_evidence_ids": selected.get("evidence_ids", []),
            }],
        )
    hitl = dict(choice_result.human_in_loop or {})
    all_evidence_ids = list(dict.fromkeys(
        evidence_id
        for item in assessments
        for evidence_id in item.get("evidence_ids", [])
    ))
    hitl["display_evidence_ids"] = all_evidence_ids
    answer = "当前没有足够的证据唯一确定正确选项，系统已进入人工确认环节。请核对右侧各选项证据后补充或确认答案。"
    return AnswerDraft(answer, [], [{"type": "human_in_loop", **hitl}])


def _draft_confidence(draft: Any) -> float:
    for operation in draft.operations:
        if isinstance(operation, dict) and "confidence" in operation:
            return float(operation["confidence"])
    return 0.0


def _has_terminal_operation(operations: list[dict[str, Any]]) -> bool:
    return any(operation.get("type") in {"refusal", "clarification", "human_in_loop"} for operation in operations)


def _requested_year_missing(years: list[str], hits: list[Any]) -> str | None:
    if not years:
        return None
    invalid_years = [year for year in years if not re.fullmatch(r"(?:19|20)\d{2}", year)]
    if invalid_years:
        return invalid_years[0]
    if not hits:
        return None
    evidence = " ".join(
        str(hit.item.get(key) or "")
        for hit in hits
        for key in ("content", "context", "source_title", "source_file_name", "source_local_path", "period")
    )
    missing = [year for year in years if year not in evidence]
    return missing[0] if missing else None


def _question_with_conversation_context(question: str, messages: list[dict[str, Any]] | None) -> str:
    """Add a bounded user-question-only hint for a conversational follow-up.

    Assistant answers are deliberately excluded: their prose is not source
    material.  The hint only helps recover omitted entities such as an
    indicator or table name; normal evidence retrieval still determines every
    answer.
    """
    prior_questions = [
        str(message.get("content") or "").strip()
        for message in (messages or [])
        if message.get("role") == "user" and str(message.get("content") or "").strip()
    ][-3:]
    if not prior_questions:
        return question
    history = "\n".join(f"历史问题：{item[:360]}" for item in prior_questions)
    return f"{history}\n当前问题：{question}"


def _retrieved_document_labels(hits: list[Any], limit: int = 6) -> list[str]:
    labels: list[str] = []
    for hit in hits:
        item = getattr(hit, "item", {})
        label = item.get("source_title") or item.get("source_file_name") or item.get("title") or item.get("file_name")
        if label:
            compact = _compact_source_title(label) or str(label)
            if compact not in labels:
                labels.append(compact[:100])
        if len(labels) >= limit:
            break
    return labels


def _compact_source_title(title: Any) -> str | None:
    if not title:
        return None
    value = str(title)
    left, separator, right = value.rpartition("_")
    if separator:
        right_base = re.sub(r"(?:\.p)?$", "", right)
        if left and right_base == left:
            return left
    return value


def _is_ratio_indicator(indicator: Any) -> bool:
    text = str(indicator or "")
    return any(term in text for term in ("率", "比例", "占比", "比率"))


def _enrich_hits(index: Any, hits: list[Any]) -> None:
    """Attach source metadata once, regardless of the retrieval attempt."""
    for hit in hits:
        source = getattr(index, "doc_by_id", {}).get(str(hit.item.get("doc_id")), {})
        hit.item.setdefault("source_file_name", source.get("file_name"))
        hit.item.setdefault("source_title", _compact_source_title(source.get("title")))
        hit.item.setdefault("source_local_path", source.get("local_path"))
        hit.item.setdefault("document_status", source.get("status"))
        if hit.kind == "table" and not hit.item.get("unit") and _is_ratio_indicator(hit.item.get("indicator")):
            hit.item["unit"] = "%"
            hit.item["unit_inferred"] = True


def _should_retry_verification(verification: Any, draft: AnswerDraft, choice_result: Any, missing_year: str | None) -> bool:
    if verification.passed or choice_result is not None or missing_year:
        return False
    terminal = {"refusal", "clarification", "human_in_loop"}
    return not any(operation.get("type") in terminal for operation in draft.operations)


def _verification_retry_query(parsed: Any, question: str) -> str:
    """A bounded rewrite for evidence repair; no hidden reasoning is exposed."""
    base = parsed.rewritten_queries[-1] if parsed.rewritten_queries else question
    anchors = [
        parsed.entities.get("indicator"),
        parsed.entities.get("table_name"),
        parsed.entities.get("period"),
        parsed.entities.get("article_no"),
    ]
    return " ".join(str(value) for value in [base, *anchors, "直接证据 条款 数值"] if value)


def _finalize_agent_workflow(
    workflow: dict[str, Any],
    hits: list[Any],
    minimal_evidence_ids: list[str],
    draft: AnswerDraft,
    verification: Any,
    retry_record: dict[str, Any] | None,
) -> None:
    for task in workflow.get("tasks", []):
        if task.get("status") == "planned":
            task["status"] = "completed"
    workflow["evidence_organization"] = {
        "strategy": "minimal_sufficient_evidence",
        "items": organize_evidence(hits, minimal_evidence_ids, draft.operations),
    }
    workflow["answer_generation"] = {
        "strategy": "llm_grounded_with_deterministic_fallback" if any(operation.get("type") == "llm_generation" and operation.get("status") == "accepted" for operation in draft.operations) else "grounded_deterministic_template",
        "claims": verification.claim_results,
        "llm": next((operation for operation in draft.operations if operation.get("type") == "llm_generation"), None),
    }
    workflow["assisted_verification"] = {
        "passed": verification.passed,
        "numeric_ok": verification.numeric_ok,
        "date_ok": verification.date_ok,
        "entity_ok": verification.entity_ok,
        "document_no_ok": verification.document_no_ok,
        "normative_strength_ok": verification.normative_strength_ok,
        "version_ok": verification.version_ok,
        "conflicts": verification.conflicts,
        "retry": retry_record,
    }


def _minimal_display_hits(hits: list[Any], operations: list[dict[str, Any]], control_operation: dict[str, Any] | None) -> list[Any]:
    """Return only the evidence needed to explain the displayed answer."""
    explicit_ids = []
    for operation in operations:
        explicit_ids.extend(operation.get("display_evidence_ids", []))
        if operation.get("type") == "cross_file_judgment":
            explicit_ids.extend(operation.get("evidence_ids", []))
    if explicit_ids:
        by_id = {hit.evidence_id: hit for hit in hits}
        return [by_id[evidence_id] for evidence_id in dict.fromkeys(explicit_ids) if evidence_id in by_id]
    if control_operation is not None and (control_operation.get("type") == "refusal" or not control_operation.get("source")):
        return []
    lookup = next((operation for operation in operations if operation.get("type") == "table_lookup"), None)
    if lookup and lookup.get("cell"):
        exact = [hit for hit in hits if hit.item.get("cell_address") == lookup["cell"]]
        if exact:
            return exact[:1]
    return hits[:4]


def _merge_hits(*groups: list[Any]) -> list[Any]:
    """Merge independently scoped retrieval routes without dropping a hop."""
    merged: dict[str, Any] = {}
    for group in groups:
        for hit in group:
            previous = merged.get(hit.evidence_id)
            if previous is None or hit.fused_score > previous.fused_score:
                merged[hit.evidence_id] = hit
    return sorted(merged.values(), key=lambda hit: hit.fused_score, reverse=True)
