from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import openpyxl


TYPE_MAP = {
    "单事实检索": "regulatory_fact",
    "多事实检索": "cross_file_judgment",
    "表格取数": "table_lookup",
    "表格比较": "cross_file_judgment",
    "表格计算": "table_lookup",
}


def convert_row(record: dict[object, object]) -> dict[str, object]:
    """Convert one competition spreadsheet row to the local evaluation contract."""
    return {
        "id": record["id"], "question": record["question"],
        "choices": [record[f"option_{letter}"] for letter in "abcd"],
        "answer": record["answer"], "answer_text": record["answer_text"],
        "evidence": record["evidence"], "source_title": record["source_title"],
        "file_label": record["file_label"], "source_type": record["source_type"],
        "difficulty": record["difficulty"],
        "qa_type": TYPE_MAP.get(record["qa_type"], "regulatory_fact"),
        "expected_behavior": "answer",
        "tags": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("xlsx", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sheet = openpyxl.load_workbook(args.xlsx, read_only=True, data_only=True).active
    rows = iter(sheet.values)
    headers = list(next(rows))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = dict(zip(headers, row))
            item = convert_row(record)
            handle.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
