from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bankreg_trustrag.config import Settings
from bankreg_trustrag.reasoning import choose_option
from bankreg_trustrag.service import TrustRAGService


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare A/B/C/D retrieval paths on the same rows")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ablation_report.json"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    service = TrustRAGService(Settings.from_env(Path.cwd()))
    rows = list(read_jsonl(args.dataset))[: args.limit or None]
    variants = {"A_baseline": "dense", "B_hybrid": "hybrid", "C_structure": "structure", "D_trustrag": "trust"}
    report = {name: {"rows": 0, "correct": 0, "retrieval_hit": 0, "citation_hit": 0} for name in variants}
    for row in rows:
        parsed = service.index
        question = row["question"]
        qa_type = row.get("qa_type", "regulatory_fact")
        choices = [str(value) for value in row.get("choices", [])]
        text_hits = parsed.search_text(question, 8, {"file_name": [row.get("file_label")], "title": [row.get("source_title")]})
        table_hits = parsed.search_tables(question, 8, {"file_name": [row.get("file_label")], "title": [row.get("source_title")]})
        routes = {"dense": sorted(text_hits, key=lambda x: x.dense_score, reverse=True)[:8], "hybrid": text_hits, "structure": sorted(text_hits + table_hits, key=lambda x: x.fused_score, reverse=True)[:8], "trust": sorted(text_hits + table_hits, key=lambda x: x.fused_score, reverse=True)[:8]}
        for name, route in variants.items():
            hits = routes[route]
            predicted, _, _ = choose_option(question, choices, hits)
            report[name]["rows"] += 1
            report[name]["correct"] += int(predicted == row.get("answer"))
            expected_file = str(row.get("file_label") or "")
            report[name]["retrieval_hit"] += int(any(expected_file and expected_file in str(service.index.doc_by_id.get(str(h.item.get("doc_id")), {}).get("file_name") or h.item.get("file_name") or "") for h in hits))
            report[name]["citation_hit"] += int(bool(hits))
    for value in report.values():
        rows_count = max(value["rows"], 1)
        value["accuracy"] = value["correct"] / rows_count
        value["retrieval_recall_at_k"] = value["retrieval_hit"] / rows_count
        value["citation_hit_rate"] = value["citation_hit"] / rows_count
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
