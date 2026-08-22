"""Build the reproducible five-category BankReg evaluation set locally.

The organiser-provided workbook supplies 300 labelled answerable questions,
but its original labels cover only three categories and contain no deliberate
refusal cases. This script preserves every original row, maps two documented
source subsets to their more specific task routes, and appends deterministic
out-of-knowledge-base / insufficient-evidence cases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CLAUSE_THRESHOLD_IDS = {f"Q{number:03d}" for number in range(105, 194, 8)}
BUSINESS_PROCESS_IDS = {f"Q{number:03d}" for number in range(206, 297, 6)}

REFUSAL_CASES: tuple[dict[str, Any], ...] = (
    {"id": "OOD_001", "question": "今天天气怎么样？", "answer": None, "evidence": "", "source_title": "", "file_label": "", "difficulty": "easy", "qa_type": "regulatory_fact", "expected_behavior": "refuse", "tags": ["out_of_scope"]},
    {"id": "OOD_002", "question": "请查询209年商业银行不良贷款率。", "answer": None, "evidence": "", "source_title": "", "file_label": "", "difficulty": "easy", "qa_type": "table_lookup", "expected_behavior": "refuse", "tags": ["invalid_year"]},
    {"id": "OOD_003", "question": "请查询2028年商业银行主要监管指标情况表中的不良贷款率。", "answer": None, "evidence": "", "source_title": "", "file_label": "", "difficulty": "medium", "qa_type": "table_lookup", "expected_behavior": "refuse", "tags": ["missing_period"]},
    {"id": "OOD_004", "question": "请告诉我监管部门最新规定。", "answer": None, "evidence": "", "source_title": "", "file_label": "", "difficulty": "medium", "qa_type": "clause_threshold", "expected_behavior": "refuse", "tags": ["unbounded_current_version"]},
    {"id": "OOD_005", "question": "请查询2025年商业银行主要监管指标情况表中的经营数据。", "answer": None, "evidence": "", "source_title": "", "file_label": "", "difficulty": "easy", "qa_type": "table_lookup", "expected_behavior": "clarify", "tags": ["missing_indicator"]},
    {"id": "OOD_006", "question": "某银行明年是否一定不会发生风险？", "answer": None, "evidence": "", "source_title": "", "file_label": "", "difficulty": "medium", "qa_type": "business_process", "expected_behavior": "clarify", "tags": ["unsupported_prediction"]},
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_rows(base_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a stable full set without mutating the source evaluation rows."""
    rows: list[dict[str, Any]] = []
    for source in base_rows:
        row = dict(source)
        if row["id"] in CLAUSE_THRESHOLD_IDS:
            row["qa_type"] = "clause_threshold"
            row["tags"] = [*row.get("tags", []), "curated_clause_threshold"]
        elif row["id"] in BUSINESS_PROCESS_IDS:
            row["qa_type"] = "business_process"
            row["tags"] = [*row.get("tags", []), "curated_business_process"]
        rows.append(row)
    rows.extend(dict(row) for row in REFUSAL_CASES)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local five-category + refusal evaluation JSONL")
    parser.add_argument("base", type=Path, help="JSONL created from the organiser workbook")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = build_rows(read_jsonl(args.base))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
