from __future__ import annotations

import argparse
import py_compile
import re
import shutil
from pathlib import Path


TECHNICAL_REJECTION = "回答草稿包含无法由当前证据或计算结果核验的事实，已停止返回该草稿。请重试。"


def _backup(path: Path) -> Path:
    backup = path.with_suffix(path.suffix + ".before_agentic_full_fix.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def _insert_before(text: str, marker: str, payload: str, *, label: str) -> str:
    if marker not in text:
        raise RuntimeError(f"Cannot locate insertion point: {label}")
    return text.replace(marker, payload + marker, 1)


def patch_service(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []

    # 1) Multiple choice is an answer format, never a retrieval-route bypass.
    if "and not has_choices" in text:
        text = text.replace(
            '        has_choices = bool(valid_choices(choices or inline_choices))\n'
            '        if bool(getattr(self.settings, "agentic_planner_enabled", False)) and not has_choices:\n',
            '        effective_choices = valid_choices(choices or inline_choices)\n'
            '        if bool(getattr(self.settings, "agentic_planner_enabled", False)):\n',
            1,
        )
        old_call = (
            "                observer,\n"
            "                request_started=request_started,\n"
        )
        new_call = (
            "                observer,\n"
            "                choices=effective_choices,\n"
            "                request_started=request_started,\n"
        )
        if old_call in text:
            text = text.replace(old_call, new_call, 1)
        changes.append("选择题统一进入 Agentic Query Planner")

    agentic_start = text.find("    def _ask_agentic(")
    legacy_start = text.find("    def _ask_legacy(", agentic_start)
    if agentic_start < 0:
        raise RuntimeError("Cannot locate TrustRAGService._ask_agentic().")
    if legacy_start < 0:
        legacy_start = len(text)
    agentic = text[agentic_start:legacy_start]

    if "choices: list[str] | None = None" not in agentic:
        old_sig = (
            "        observer: Callable[[str, dict[str, Any]], None] | None,\n"
            "        *,\n"
            "        request_started: float | None = None,\n"
        )
        new_sig = (
            "        observer: Callable[[str, dict[str, Any]], None] | None,\n"
            "        *,\n"
            "        choices: list[str] | None = None,\n"
            "        request_started: float | None = None,\n"
        )
        if old_sig in agentic:
            agentic = agentic.replace(old_sig, new_sig, 1)
            changes.append("_ask_agentic 接收标准化候选选项")

    if "agentic_question = _agentic_question_with_choices(question, choices)" not in agentic:
        needle = '        executor = getattr(self, "agentic_executor", None)\n'
        if needle in agentic:
            agentic = agentic.replace(
                needle,
                "        agentic_question = _agentic_question_with_choices(question, choices)\n" + needle,
                1,
            )
            agentic = agentic.replace(
                "        state: AgentState = executor.run(question, conversation_context, report)\n",
                "        state: AgentState = executor.run(agentic_question, conversation_context, report)\n",
                1,
            )
            changes.append("API 独立 choices 作为 HumanMessage 上下文交给 Planner")

    # 2) Bridge Answer-Agent task refs -> exact evidence ids for Verification.
    if "_expanded_agentic_grounding_refs(" not in agentic:
        raw_grounding = (
            "            grounding_refs=(\n"
            "                state.answer_outcome.generated.output_refs_by_requirement\n"
            "                if state.answer_outcome else None\n"
            "            ),\n"
        )
        expanded_grounding = (
            "            grounding_refs=_expanded_agentic_grounding_refs(\n"
            "                state,\n"
            "                state.answer_outcome.generated.output_refs_by_requirement\n"
            "                if state.answer_outcome else None,\n"
            "            ),\n"
        )
        if raw_grounding in agentic:
            agentic = agentic.replace(raw_grounding, expanded_grounding, 1)
            changes.append("Verification 获得 RetrievalTask 对应的真实 evidence_id")

    raw_repair = "                    grounding_refs=repaired.generated.output_refs_by_requirement,\n"
    if raw_repair in agentic:
        agentic = agentic.replace(
            raw_repair,
            "                    grounding_refs=_expanded_agentic_grounding_refs("
            "state, repaired.generated.output_refs_by_requirement),\n",
            1,
        )

    if "agentic_question = _agentic_question_with_choices" in agentic:
        repair_block = (
            "            repaired = executor.answer_generator.generate(\n"
            "                question,\n"
        )
        if repair_block in agentic:
            agentic = agentic.replace(
                repair_block,
                "            repaired = executor.answer_generator.generate(\n"
                "                agentic_question,\n",
                1,
            )

    if TECHNICAL_REJECTION in agentic:
        agentic = agentic.replace(
            TECHNICAL_REJECTION,
            "当前证据仍不足以核验最终结论，无法可靠回答。",
        )
        changes.append("移除面向用户的内部“草稿核验失败”技术拒答")

    agentic = agentic.replace(
        '            "option_evaluation": False,\n',
        '            "option_evaluation": bool(choices),\n',
        1,
    )
    agentic = agentic.replace(
        '            "answer_format": "free_text",\n',
        '            "answer_format": "multiple_choice" if choices else "free_text",\n',
        1,
    )

    text = text[:agentic_start] + agentic + text[legacy_start:]

    if "def _agentic_question_with_choices(" not in text:
        helper = '''

def _agentic_question_with_choices(question: str, choices: list[str] | None) -> str:
    """Expose API-supplied choices to the planner without treating them as evidence."""
    options = valid_choices(choices or [])
    if not options:
        return question
    _, inline = extract_inline_choices(question)
    if valid_choices(inline):
        return question
    rendered = "\\n".join(
        f"{chr(ord('A') + index)}. {option}"
        for index, option in enumerate(options)
    )
    return (
        f"{question.rstrip()}\\n\\n"
        "候选选项（仅供最终判断，选项内容不是证据）：\\n"
        f"{rendered}"
    )

'''
        text = _insert_before(
            text,
            "\ndef _agentic_qa_type(plan: QueryPlan) -> str:\n",
            helper,
            label="_agentic_question_with_choices",
        )

    if "def _expanded_agentic_grounding_refs(" not in text:
        helper = '''

def _expanded_agentic_grounding_refs(
    state: AgentState,
    refs_by_requirement: dict[str, list[str]] | None,
) -> dict[str, list[str]] | None:
    """Expand Answer-Agent task/result refs to exact evidence provenance."""
    if not refs_by_requirement:
        return None
    expanded: dict[str, list[str]] = {}
    for requirement_id, refs in refs_by_requirement.items():
        values: list[str] = []
        for raw_ref in refs or []:
            ref = str(raw_ref)
            if ref and ref not in values:
                values.append(ref)

            retrieval = state.retrieval_results.get(ref)
            if retrieval is not None:
                for evidence_id in retrieval.evidence_ids:
                    evidence_id = str(evidence_id)
                    if evidence_id and evidence_id not in values:
                        values.append(evidence_id)

            calculation = state.calculation_results.get(ref)
            if calculation is not None:
                for evidence_id in getattr(calculation, "evidence_ids", []) or []:
                    evidence_id = str(evidence_id)
                    if evidence_id and evidence_id not in values:
                        values.append(evidence_id)

        if values:
            expanded[str(requirement_id)] = values
    return expanded or None

'''
        text = _insert_before(
            text,
            "\ndef _agentic_qa_type(plan: QueryPlan) -> str:\n",
            helper,
            label="_expanded_agentic_grounding_refs",
        )
        changes.append("新增 Answer Agent → Verification provenance bridge")

    return text, changes


def patch_verification(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []

    call_old = (
        "    verification.claim_results = _verify_claims_against_individual_evidence(\n"
        "        claims,\n"
        "        hits,\n"
        "        operations,\n"
        "        grounding_refs,\n"
        "    )\n"
    )
    call_new = (
        "    verification.claim_results = _verify_claims_against_individual_evidence(\n"
        "        claims,\n"
        "        hits,\n"
        "        operations,\n"
        "        grounding_refs,\n"
        "        question,\n"
        "    )\n"
    )
    if call_old in text:
        text = text.replace(call_old, call_new, 1)

    sig_old = (
        "def _verify_claims_against_individual_evidence(\n"
        "    claims: list[str],\n"
        "    hits: list[Hit],\n"
        "    operations: list[dict[str, Any]],\n"
        "    grounding_refs: dict[str, list[str]] | None = None,\n"
        ") -> list[dict[str, Any]]:\n"
    )
    sig_new = (
        "def _verify_claims_against_individual_evidence(\n"
        "    claims: list[str],\n"
        "    hits: list[Hit],\n"
        "    operations: list[dict[str, Any]],\n"
        "    grounding_refs: dict[str, list[str]] | None = None,\n"
        "    question: str | None = None,\n"
        ") -> list[dict[str, Any]]:\n"
    )
    if sig_old in text:
        text = text.replace(sig_old, sig_new, 1)

    if "declared_evidence_ids = {" not in text:
        needle = (
            "    declared_operations = [\n"
            "        (index, operation)\n"
            "        for index, operation in enumerate(operations)\n"
            "        if _calculation_id(operation, index) in declared_refs\n"
            "    ]\n"
        )
        addition = needle + (
            "    declared_evidence_ids = {\n"
            "        hit.evidence_id for hit in hits if hit.evidence_id in declared_refs\n"
            "    }\n"
            "    declared_evidence_blob = \"\\n\".join(\n"
            "        _evidence_for_hit(hit) for hit in hits if hit.evidence_id in declared_evidence_ids\n"
            "    )\n"
        )
        if needle not in text:
            raise RuntimeError("Cannot locate declared_operations in verification.py")
        text = text.replace(needle, addition, 1)

    if "_grounded_choice_conclusion(text, question, declared_evidence_blob)" not in text:
        needle = (
            "        if not evidence_ids and calculation_evidence_ids:\n"
            "            evidence_ids = calculation_evidence_ids\n"
        )
        addition = needle + '''        if not evidence_ids and not calculation_ids and declared_evidence_ids:
            # The Answer Agent explicitly bound this requirement to tool output.
            # Service expanded retrieval refs to exact evidence IDs.  Natural
            # paraphrases and final choice labels may use that provenance, while
            # numeric/date/entity/normative checks remain hard factual guards.
            if (
                _claim_supported(text, declared_evidence_blob)
                or _grounded_choice_conclusion(text, question, declared_evidence_blob)
            ):
                evidence_ids = sorted(declared_evidence_ids)
'''
        if needle not in text:
            raise RuntimeError("Cannot locate claim support insertion point.")
        text = text.replace(needle, addition, 1)
        changes.append("Claim 核验支持 Answer Agent 显式绑定的检索证据")

    if "def _grounded_choice_conclusion(" not in text:
        helper = r'''

def _grounded_choice_conclusion(
    claim: str,
    question: str | None,
    evidence: str,
) -> bool:
    """Validate '选择A/B/C/D' by checking the selected option proposition."""
    text = normalize_text(claim)
    prompt = normalize_text(question)
    evidence = normalize_text(evidence)
    if not text or not prompt or not evidence:
        return False

    options: dict[str, str] = {}
    pattern = re.compile(
        r"([A-H])\s*[\.．、:：]\s*(.*?)"
        r"(?=(?:[A-H]\s*[\.．、:：])|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(prompt):
        label = match.group(1).upper()
        value = normalize_text(match.group(2)).strip(" ，,。；;")
        if value:
            options[label] = value
    if not options:
        return False

    selected: str | None = None
    for candidate_pattern in (
        r"(?:正确答案|答案|选择|应选|故选|选项)\s*(?:是|为|：|:)?\s*([A-H])(?:项)?",
        r"\b([A-H])\s*项\s*(?:正确|符合|成立)",
    ):
        match = re.search(candidate_pattern, text, flags=re.IGNORECASE)
        if match:
            selected = match.group(1).upper()
            break
    if selected is None or selected not in options:
        return False

    # Critical safety rule: the user's option is a candidate, never evidence.
    # The proposition inside the selected option must itself match real evidence.
    if not _claim_supported(options[selected], evidence):
        return False

    remainder = re.sub(
        r"(?:明确)?(?:正确答案|答案|选择|应选|故选|选项)\s*(?:是|为|：|:)?\s*[A-H](?:项)?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    remainder = re.sub(
        r"\b[A-H]\s*项\s*(?:正确|符合|成立)",
        "",
        remainder,
        flags=re.IGNORECASE,
    )
    remainder = re.sub(
        r"^(?:因此|所以|据此|综上|结论(?:是|为|：|:)?)\s*",
        "",
        remainder,
    )
    remainder = remainder.strip(" ，,。；;：:（）()")
    return not remainder or _claim_supported(remainder, evidence)

'''
        text = _insert_before(
            text,
            "\ndef _claim_supported(claim: str, evidence: str) -> bool:\n",
            helper,
            label="_grounded_choice_conclusion",
        )
        changes.append("选择标签按‘所选选项事实→证据’核验，而非要求原文出现‘A项正确’")

    return text, changes


def patch_retrieval_tools(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []

    # Do NOT depend on the exact local spelling of:
    #     filters = _task_filters(task)
    # Earlier fixes may already have changed that line. Resolve/relax the
    # planner's source_hint at the RetrievalTask level immediately after task
    # normalization, before _task_query() and before any filter construction.
    if "_resolve_agentic_source_scope(task, self.index)" not in text:
        normalized_pattern = re.compile(
            r"(?m)^(?P<indent>[ \t]+)task\s*=\s*_normalized_task\(task\)\s*$"
        )
        match = normalized_pattern.search(text)
        if match:
            indent = match.group("indent")
            insertion = (
                match.group(0)
                + "\n"
                + indent
                + "task = _resolve_agentic_source_scope(task, self.index)"
            )
            text = text[:match.start()] + insertion + text[match.end():]
            changes.append("source_hint 在 RetrievalTask 层解析/放宽，不依赖具体 filters 写法")
        else:
            # Optional source-title repair must never block the core fix.
            changes.append("未找到 _normalized_task(task)，跳过可选 source_hint 修复")

    if "def _resolve_agentic_source_scope(" not in text:
        helper = r"""

def _resolve_agentic_source_scope(task: RetrievalTask, index: Any) -> RetrievalTask:
    # Resolve a planner source hint against REAL ingested document titles.
    source = getattr(task, "source_scope", None)
    requested = normalize_text(getattr(source, "document_title", None))
    if not requested:
        return task

    documents = list(getattr(index, "documents", []) or [])
    if not documents:
        documents = list(getattr(index, "doc_by_id", {}).values())
    if not documents:
        return task

    matched_titles: list[str] = []
    for document in documents:
        actual_blob = " ".join(
            str(document.get(key) or "")
            for key in ("title", "file_name", "local_path")
        )
        if not _source_hint_matches(requested, actual_blob):
            continue
        actual_title = normalize_text(document.get("title") or document.get("file_name"))
        if actual_title and actual_title not in matched_titles:
            matched_titles.append(actual_title)

    if len(matched_titles) == 1:
        new_source = source.model_copy(update={"document_title": matched_titles[0]})
        return task.model_copy(update={"source_scope": new_source})

    # No unique fuzzy resolution: drop only the hard title constraint, but keep
    # the planner hint in the semantic query so it still helps ranking.
    new_source = source.model_copy(update={"document_title": None})
    current_query = normalize_text(getattr(task, "query", ""))
    new_query = current_query
    if requested and requested not in current_query:
        new_query = f"{current_query} {requested}".strip()
    return task.model_copy(update={
        "source_scope": new_source,
        "query": new_query,
    })


def _source_hint_matches(expected: str, actual: str) -> bool:
    expected_text = normalize_text(expected)
    actual_text = normalize_text(actual)

    expected_years = set(re.findall(r"(?:19|20)\d{2}", expected_text))
    actual_years = set(re.findall(r"(?:19|20)\d{2}", actual_text))
    if expected_years and actual_years and not expected_years.intersection(actual_years):
        return False

    expected_key = _source_title_key(expected_text)
    actual_key = _source_title_key(actual_text)
    if not expected_key or not actual_key:
        return False
    if expected_key in actual_key or actual_key in expected_key:
        return True

    def bigrams(value: str) -> set[str]:
        return {value[index:index + 2] for index in range(max(0, len(value) - 1))}

    left, right = bigrams(expected_key), bigrams(actual_key)
    if not left or not right:
        return False
    overlap = len(left & right)
    coverage = overlap / max(1, len(left))
    reverse_coverage = overlap / max(1, len(right))
    return max(coverage, reverse_coverage) >= 0.78


def _source_title_key(value: Any) -> str:
    text = canonical_table_label(value)
    for token in (
        "excel", "word", "pdf", "文件", "工作表",
        "情况", "数据", "统计", "报表", "报告",
        "（", "）", "(", ")", "季度", "月度",
    ):
        text = text.replace(token, "")
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", text).lower()


"""
        marker = "\ndef _matches_semantic_constraints(hit: Hit, task: RetrievalTask, index: Any) -> bool:\n"
        if marker in text:
            text = text.replace(marker, helper + marker, 1)
        else:
            marker = "\ndef _institution_matches(requested: str, blob: str) -> bool:\n"
            if marker in text:
                text = text.replace(marker, helper + marker, 1)
            else:
                # Never abort the entire systemic fix for a local retriever refactor.
                changes.append("无法定位 source_hint helper 插入点，跳过可选 source_hint helper")

    return text, changes



def patch_answer_generator(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    marker = (
        "- 在 output_refs_by_requirement 中列出每项回答实际使用的 "
        "RetrievalTask/CalculationResult 引用；answered_requirement_ids 只能包含确实回答完成的要求。\n"
    )
    if "“选择A / A项正确”属于答案格式" not in text:
        addition = marker + (
            "- 如果用户给出 A/B/C/D 等候选项：先基于证据判断，再明确输出所选标签；"
            "选项文本只是候选而不是证据。不要为了‘解释充分’而逐项扩写未检索到的选项事实；"
            "只陈述 provided_evidence / calculation_results 能直接支持的理由。\n"
            "- “选择A / A项正确”属于答案格式与基于证据的结论表达；它本身不需要逐字出现在原文，"
            "但支撑该选择的事实必须由 output_refs_by_requirement 绑定到真实检索/计算结果。\n"
        )
        if marker not in text:
            raise RuntimeError("Cannot locate Answer Agent grounding rule.")
        text = text.replace(marker, addition, 1)
        changes.append("Answer Agent 明确区分‘候选选项’和‘来源证据’")

    return text, changes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Systemic Agentic/Verification/Retrieval fix for BankReg-TrustRAG."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()

    files = {
        "service": root / "bankreg_trustrag" / "service.py",
        "verification": root / "bankreg_trustrag" / "verification.py",
        "retrieval": root / "bankreg_trustrag" / "retrieval_tools.py",
        "answer": root / "bankreg_trustrag" / "answer_generator.py",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise SystemExit("Missing project files:\n" + "\n".join(missing))

    if args.check_only:
        texts = {key: path.read_text(encoding="utf-8") for key, path in files.items()}
        checks = {
            "choice_routes_through_agentic": "and not has_choices" not in texts["service"],
            "provenance_bridge": "def _expanded_agentic_grounding_refs(" in texts["service"],
            "choice_verification": "def _grounded_choice_conclusion(" in texts["verification"],
            "source_hint_resolution": ("def _resolve_agentic_source_scope(" in texts["retrieval"] or "def _validated_source_filters(" in texts["retrieval"]),
            "answer_choice_grounding": "“选择A / A项正确”属于答案格式" in texts["answer"],
            "no_internal_draft_rejection": TECHNICAL_REJECTION not in texts["service"],
        }
        for name, ok in checks.items():
            print(f"{name}: {'OK' if ok else 'MISSING'}")
        raise SystemExit(0 if all(checks.values()) else 2)

    original = {key: path.read_text(encoding="utf-8") for key, path in files.items()}
    patched: dict[str, str] = {}
    changes: list[str] = []

    patched["service"], c = patch_service(original["service"]); changes.extend(c)
    patched["verification"], c = patch_verification(original["verification"]); changes.extend(c)
    patched["retrieval"], c = patch_retrieval_tools(original["retrieval"]); changes.extend(c)
    patched["answer"], c = patch_answer_generator(original["answer"]); changes.extend(c)

    for path in files.values():
        _backup(path)
    for key, path in files.items():
        path.write_text(patched[key], encoding="utf-8")

    try:
        for path in files.values():
            py_compile.compile(str(path), doraise=True)
    except Exception:
        print("Syntax validation failed. Restoring backups.")
        for path in files.values():
            backup = path.with_suffix(path.suffix + ".before_agentic_full_fix.bak")
            if backup.exists():
                shutil.copy2(backup, path)
        raise

    print("Agentic full fix applied successfully.")
    for item in changes:
        print(" -", item)
    print(" - Python syntax validation: OK")
    print("Backups: *.before_agentic_full_fix.bak")
    print("No re-ingest / JSON / SQLite / BGE rebuild is required.")


if __name__ == "__main__":
    main()
