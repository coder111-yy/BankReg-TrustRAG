#!/usr/bin/env python3
"""Validate BankReg-TrustRAG evaluation JSONL files.

Usage:
    python .agents/skills/bankreg-rag-evaluation/scripts/validate_eval_jsonl.py path/to/eval.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REQUIRED = {
    "id",
    "question",
    "answer",
    "evidence",
    "source_title",
    "source_url",
    "local_path",
    "difficulty",
    "qa_type",
    "tags",
}

QA_TYPES = {
    "regulatory_fact",
    "clause_threshold",
    "business_process",
    "table_lookup",
    "cross_file_judgment",
}

DIFFICULTIES = {"easy", "medium", "hard"}
EXPECTED_BEHAVIORS = {"answer", "refuse", "clarify"}


def fail(line_no: int, message: str) -> str:
    return f"line {line_no}: {message}"


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python validate_eval_jsonl.py <eval.jsonl>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 2

    errors: list[str] = []
    ids: set[str] = set()
    count = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            count += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                errors.append(fail(line_no, f"invalid JSON: {e}"))
                continue

            missing = REQUIRED - set(row)
            if missing:
                errors.append(fail(line_no, f"missing fields: {sorted(missing)}"))

            row_id = row.get("id")
            if not isinstance(row_id, str) or not row_id.strip():
                errors.append(fail(line_no, "id must be a non-empty string"))
            elif row_id in ids:
                errors.append(fail(line_no, f"duplicate id: {row_id}"))
            else:
                ids.add(row_id)

            if row.get("qa_type") not in QA_TYPES:
                errors.append(fail(line_no, f"invalid qa_type: {row.get('qa_type')!r}"))

            if row.get("difficulty") not in DIFFICULTIES:
                errors.append(fail(line_no, f"invalid difficulty: {row.get('difficulty')!r}"))

            behavior = row.get("expected_behavior", "answer")
            if behavior not in EXPECTED_BEHAVIORS:
                errors.append(fail(line_no, f"invalid expected_behavior: {behavior!r}"))

            if not isinstance(row.get("evidence"), list):
                errors.append(fail(line_no, "evidence must be a list"))
            if not isinstance(row.get("tags"), list):
                errors.append(fail(line_no, "tags must be a list"))

            if behavior == "answer" and not str(row.get("answer", "")).strip():
                errors.append(fail(line_no, "answerable row must contain a non-empty answer"))

    if count == 0:
        errors.append("dataset contains no JSONL rows")

    if errors:
        print(f"FAILED: {len(errors)} validation error(s)")
        for e in errors:
            print(f"- {e}")
        return 1

    print(f"OK: {count} row(s), {len(ids)} unique id(s), schema valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
