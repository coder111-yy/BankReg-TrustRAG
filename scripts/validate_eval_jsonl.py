from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = {"id", "question", "answer", "evidence", "source_title", "file_label", "difficulty", "qa_type", "expected_behavior"}
QA_TYPES = {"regulatory_fact", "clause_threshold", "business_process", "table_lookup", "cross_file_judgment"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    seen: set[str] = set()
    count = 0
    with args.path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON: {exc}")
                continue
            count += 1
            missing = REQUIRED - set(row)
            if missing:
                errors.append(f"line {line_no}: missing {sorted(missing)}")
            if row.get("id") in seen:
                errors.append(f"line {line_no}: duplicate id {row.get('id')}")
            seen.add(row.get("id"))
            if row.get("qa_type") not in QA_TYPES:
                errors.append(f"line {line_no}: invalid qa_type {row.get('qa_type')}")
            if row.get("expected_behavior") not in {"answer", "refuse", "clarify"}:
                errors.append(f"line {line_no}: invalid expected_behavior")
    result = {"valid": not errors, "rows": count, "errors": errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not errors else 1)


if __name__ == "__main__":
    main()

