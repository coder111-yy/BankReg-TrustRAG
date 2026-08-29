"""Run a controlled A/B/C/D retrieval and trust ablation.

The four variants share the same corpus, questions and answer scorer. Gold
source metadata is used only for measuring retrieval recall; it is never
passed as a filter to the system, avoiding source leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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


def _choices(row: dict[str, Any]) -> list[str]:
    return [str(value) for value in row.get("choices", [])]


def _source_matches(service: TrustRAGService, hits: list[Any], row: dict[str, Any]) -> bool:
    expected = str(row.get("file_label") or "")
    title = str(row.get("source_title") or "")
    for hit in hits:
        document = service.index.doc_by_id.get(str(hit.item.get("doc_id") or ""), {})
        file_name = str(document.get("file_name") or hit.item.get("file_name") or "")
        source_title = str(document.get("title") or hit.item.get("source_title") or "")
        if (expected and expected.lower() in file_name.lower()) or (title and title in source_title):
            return True
    return False


def _source_matches_from_dicts(items: list[dict[str, Any] | None], row: dict[str, Any]) -> bool:
    expected = str(row.get("file_label") or "").lower()
    title = str(row.get("source_title") or "")
    return any(
        item
        and ((expected and expected in str(item.get("source_file_name") or "").lower())
             or (title and title in str(item.get("source_title") or "")))
        for item in items
    )


def _direct_variant(
    service: TrustRAGService,
    question: str,
    qa_type: str,
    variant: str,
) -> list[Any]:
    """Retrieve one variant without executing generation or verification."""
    index = service.index
    if hasattr(index, "begin_query"):
        index.begin_query()
    if variant == "A_baseline_dense":
        # The baseline treats table records as flattened semantic documents.
        # This is intentionally weaker than the structured route.
        text_hits = index.search_text(question, 32, None, dense=True, lexical=False, metadata=False, fuse=False)
        table_hits = index.search_tables(question, 32, None, dense=True, lexical=False, metadata=False, fuse=False)
        return sorted(text_hits + table_hits, key=lambda hit: (hit.dense_score, hit.evidence_id), reverse=True)[:8]
    if variant == "B_hybrid":
        return index.hybrid_search(
            question, qa_type, 8, None, rerank=False, dense=True,
            lexical=True, metadata=False, structured=True,
        )
    if variant == "C_structure":
        return index.hybrid_search(
            question, qa_type, 8, None, rerank=True, dense=True,
            lexical=True, metadata=True, structured=True,
        )
    raise ValueError(f"unsupported direct variant: {variant}")


def _row_result(service: TrustRAGService, row: dict[str, Any], variant: str) -> dict[str, Any]:
    question = str(row["question"])
    qa_type = str(row.get("qa_type") or "regulatory_fact")
    choices = _choices(row)
    if variant == "D_trustrag":
        response = service.ask(question, choices=choices or None, qa_type=qa_type)
        evidence = [
            service.store.get_evidence(item["evidence_id"])
            for item in response.evidence
            if item.get("evidence_id")
        ]
        predicted = response.query_plan.get("agent", {}).get("selected_option") if choices else None
        return {
            "id": row.get("id"), "variant": variant,
            "predicted": predicted, "expected": row.get("answer"),
            "correct": bool(predicted and predicted == row.get("answer")),
            "retrieval_hit": bool(evidence),
            "source_hit": _source_matches_from_dicts(evidence, row),
            "citation_hit": bool(response.evidence),
            "verification_passed": bool(response.verification.get("passed")),
            "decision": response.trust.get("decision"),
            "answer": response.answer,
            "evidence_ids": [item.get("evidence_id") for item in response.evidence],
        }

    hits = _direct_variant(service, question, qa_type, variant)
    predicted, confidence, assessments = choose_option(question, choices, hits) if choices else (None, 0.0, [])
    return {
        "id": row.get("id"), "variant": variant,
        "predicted": predicted, "expected": row.get("answer"),
        "correct": bool(predicted and predicted == row.get("answer")),
        "retrieval_hit": bool(hits), "source_hit": _source_matches(service, hits, row),
        "citation_hit": bool(hits), "verification_passed": False,
        "decision": "not_run", "confidence": confidence,
        "assessments": assessments, "evidence_ids": [hit.evidence_id for hit in hits],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled A/B/C/D BankReg-TrustRAG ablation")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ablation_report.json"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    service = TrustRAGService(Settings.from_env(Path.cwd()))
    rows = list(read_jsonl(args.dataset))[: args.limit or None]
    variants = ("A_baseline_dense", "B_hybrid", "C_structure", "D_trustrag")
    results = [_row_result(service, row, variant) for row in rows for variant in variants]
    report: dict[str, Any] = {
        "protocol": {
            "same_corpus": True, "same_questions": True,
            "gold_source_filter_used": False,
            "A_baseline_dense": "dense only; no lexical, metadata, structured bonus or reranker",
            "B_hybrid": "BM25 + dense + unified RRF; no metadata, structured route or reranker",
            "C_structure": "BM25 + dense + metadata + structured table route + unified RRF + BGE-Reranker",
            "D_trustrag": "full service path including reasoning, verification, trust decision and refusal/clarification",
        },
        "rows": len(rows), "variants": {}, "results": results,
    }
    for variant in variants:
        subset = [item for item in results if item["variant"] == variant]
        count = max(len(subset), 1)
        report["variants"][variant] = {
            "rows": len(subset),
            "accuracy": sum(bool(item["correct"]) for item in subset) / count,
            "retrieval_recall_at_k": sum(bool(item["source_hit"]) for item in subset) / count,
            "citation_hit_rate": sum(bool(item["citation_hit"]) for item in subset) / count,
            "verification_pass_rate": sum(bool(item["verification_passed"]) for item in subset) / count,
            "answered": sum(item["decision"] == "answer" for item in subset),
            "clarified": sum(item["decision"] == "clarify" for item in subset),
            "refused": sum(item["decision"] == "refuse" for item in subset),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["variants"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
