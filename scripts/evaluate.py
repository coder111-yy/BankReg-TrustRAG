from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bankreg_trustrag.config import Settings
from bankreg_trustrag.service import TrustRAGService
from bankreg_trustrag.utils import normalize_text


TARGETS = {
    "regulatory_fact_accuracy": 0.85,
    "table_value_accuracy": 0.80,
    "citation_hit_rate": 0.90,
    "key_fact_error_rate_max": 0.05,
    "refusal_rate_unanswerable": 0.80,
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_option(answer: str) -> str | None:
    match = re.search(r"选项\s*([ABCD])", answer or "")
    return match.group(1) if match else None


def expected_cell(evidence: str) -> str | None:
    match = re.search(r"单元格\s*[:：]\s*([A-Z]+\d+)", str(evidence or ""), re.IGNORECASE)
    return match.group(1).upper() if match else None


def matches_expected_evidence(row: dict[str, Any], item: dict[str, Any]) -> bool:
    """Require a matching cell when ground truth specifies a cell."""
    cell = expected_cell(str(row.get("evidence") or ""))
    if cell:
        return cell == str(item.get("cell_address") or "").upper()
    expected_file = normalize_text(row.get("file_label") or "").lower()
    source_file = normalize_text(item.get("source_file_name") or item.get("file_name") or "").lower()
    if expected_file and (expected_file in source_file or source_file in expected_file):
        return True
    title = normalize_text(row.get("source_title") or "").lower()
    actual_title = normalize_text(item.get("source_title") or item.get("title") or "").lower()
    return bool(title and actual_title and (title in actual_title or actual_title in title))


def evidence_set_matches(row: dict[str, Any], evidence: list[dict[str, Any]]) -> bool:
    """Score multi-claim choices against the complete returned evidence set."""
    if any(matches_expected_evidence(row, item) for item in evidence):
        return True
    expected = normalize_text(row.get("evidence") or "")
    if not expected or not evidence:
        return False
    source = " ".join(
        normalize_text(" ".join(str(item.get(key) or "") for key in ("content", "context_window", "context")))
        for item in evidence
    )
    claims = [part.strip() for part in re.split(r"[；;]", expected) if len(part.strip()) >= 8]
    return bool(claims and all(part in source for part in claims))


def reciprocal_rank(row: dict[str, Any], evidence: list[dict[str, Any]]) -> float:
    for rank, item in enumerate(evidence, 1):
        if matches_expected_evidence(row, item):
            return 1.0 / rank
    return 0.0


def rate(items: list[dict[str, Any]], key: str) -> float:
    return sum(bool(item.get(key)) for item in items) / max(len(items), 1)


def percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return float(ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def category_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [item for item in items if item["expected_behavior"] == "answer"]
    unanswerable = [item for item in items if item["expected_behavior"] != "answer"]
    return {
        "rows": len(items), "answerable_rows": len(answerable), "unanswerable_rows": len(unanswerable),
        "answer_accuracy": rate(answerable, "answer_correct") if answerable else None,
        "retrieval_recall_at_k": rate(answerable, "retrieval_hit") if answerable else None,
        "mrr": round(statistics.mean([item["reciprocal_rank"] for item in answerable]), 6) if answerable else None,
        "citation_hit_rate": rate(answerable, "citation_hit") if answerable else None,
        "refusal_or_clarification_rate": rate(unanswerable, "refusal_correct") if unanswerable else None,
        "false_refusal_rate": sum(item["decision"] != "answer" for item in answerable) / len(answerable) if answerable else None,
    }


def failure_labels(row: dict[str, Any], response: Any, retrieval_hit: bool, correct: bool) -> list[str]:
    if correct:
        return []
    if row["expected_behavior"] != "answer":
        return ["should_have_refused"]
    if response.trust["decision"] != "answer":
        return ["incorrect_refusal"]
    if not retrieval_hit:
        return ["table_retrieval_failure" if row["qa_type"] == "table_lookup" else "lexical_retrieval_failure"]
    if response.verification.get("unsupported_claims") or response.verification.get("conflicts"):
        return ["verification_miss"]
    return ["generation_unsupported_claim"]


def acceptance_gates(by_type: dict[str, dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    answerable = [item for item in results if item["expected_behavior"] == "answer"]
    unanswerable = [item for item in results if item["expected_behavior"] != "answer"]
    values = {
        "regulatory_fact_accuracy": by_type.get("regulatory_fact", {}).get("answer_accuracy"),
        "table_value_accuracy": by_type.get("table_lookup", {}).get("answer_accuracy"),
        "citation_hit_rate": rate(answerable, "citation_hit") if answerable else None,
        "key_fact_error_rate_max": rate(answerable, "key_fact_error") if answerable else None,
        "refusal_rate_unanswerable": rate(unanswerable, "refusal_correct") if unanswerable else None,
    }
    return {
        name: {"target": target, "value": values[name], "passed": values[name] is not None and (values[name] <= target if name.endswith("_max") else values[name] >= target)}
        for name, target in TARGETS.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible BankReg evaluation")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation_results.jsonl"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", help="Comma-separated stable row IDs for a diagnostic subset")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="Continue from rows already written to --output")
    args = parser.parse_args()
    settings = Settings.from_env(Path.cwd())
    service = TrustRAGService(settings)
    rows = list(read_jsonl(args.dataset))
    if args.ids:
        selected_ids = {value.strip() for value in args.ids.split(",") if value.strip()}
        rows = [row for row in rows if str(row.get("id")) in selected_ids]
    if args.limit:
        rows = rows[:args.limit]
    results: list[dict[str, Any]] = list(read_jsonl(args.output)) if args.resume and args.output.exists() else []
    completed_ids = {str(result.get("id")) for result in results}
    pending_rows = [row for row in rows if str(row.get("id")) not in completed_ids]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(output_mode, encoding="utf-8") as handle:
      for row in pending_rows:
        started = time.perf_counter()
        choices = [str(value) for value in row.get("choices", []) if value is not None]
        filters = {
            "file_name": [row["file_label"]] if row.get("file_label") else [],
            "title": [row["source_title"]] if row.get("source_title") else [],
        }
        response = service.ask(row["question"], choices, row.get("qa_type"), filters)
        evidence = response.evidence
        retrieved = evidence_set_matches(row, evidence)
        expected_behavior = row.get("expected_behavior", "answer")
        predicted = extract_option(response.answer)
        correct = predicted == row.get("answer") if expected_behavior == "answer" else response.trust["decision"] == expected_behavior
        result = {
            "id": row["id"], "qa_type": row["qa_type"], "difficulty": row.get("difficulty"),
            "expected_behavior": expected_behavior, "expected_answer": row.get("answer"), "actual_answer": predicted,
            "actual_text": response.answer, "expected_evidence": [row.get("file_label"), row.get("evidence")],
            "retrieved_evidence": [item.get("evidence_id") for item in evidence],
            "cited_evidence": [item.get("evidence_id") for item in evidence], "retrieval_hit": retrieved,
            "citation_hit": retrieved, "reciprocal_rank": reciprocal_rank(row, evidence), "answer_correct": correct,
            "key_fact_error": bool(expected_behavior == "answer" and not correct),
            "refusal_correct": response.trust["decision"] == expected_behavior if expected_behavior != "answer" else None,
            "verification": response.verification, "decision": response.trust["decision"], "trust_score": response.trust["score"],
            "latency_ms": int((time.perf_counter() - started) * 1000), "trace_id": response.trace_id,
        }
        result["failure_labels"] = failure_labels(row, response, retrieved, correct)
        results.append(result)
        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        handle.flush()
        if args.progress_every and (len(results) % args.progress_every == 0 or len(results) == len(rows)):
            print(f"completed {len(results)}/{len(rows)}", file=sys.stderr, flush=True)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[result["qa_type"]].append(result)
    by_type = {name: category_metrics(items) for name, items in sorted(groups.items())}
    answerable = [item for item in results if item["expected_behavior"] == "answer"]
    report = {
        "run": {
            "started_at": datetime.now(timezone.utc).isoformat(), "dataset": str(args.dataset),
            "dataset_sha256": file_sha256(args.dataset),
            "corpus_manifest_sha256": file_sha256(settings.artifact_dir / "manifest.json") if (settings.artifact_dir / "manifest.json").exists() else None,
            "embedding_model": settings.bge_embedding_model, "reranker_model": settings.bge_reranker_model,
            "bge_mode": settings.bge_mode, "top_k": settings.top_k, "min_trust": settings.min_trust,
            "llm_provider": "none (deterministic local generation)",
        },
        "rows": len(results), "answer_accuracy": rate(answerable, "answer_correct") if answerable else None,
        "retrieval_recall_at_k": rate(answerable, "retrieval_hit") if answerable else None,
        "mrr": round(statistics.mean([item["reciprocal_rank"] for item in answerable]), 6) if answerable else None,
        "citation_hit_rate": rate(answerable, "citation_hit") if answerable else None,
        "key_fact_error_rate": rate(answerable, "key_fact_error") if answerable else None,
        "mean_latency_ms": round(statistics.mean([item["latency_ms"] for item in results]), 3) if results else 0,
        "p50_latency_ms": round(percentile([item["latency_ms"] for item in results], 0.5), 3),
        "p95_latency_ms": round(percentile([item["latency_ms"] for item in results], 0.95), 3),
        "by_qa_type": by_type, "failure_labels": dict(Counter(label for item in results for label in item["failure_labels"])),
        "acceptance_gates": acceptance_gates(by_type, results),
        "note": "Measured locally; an acceptance gate passes only with labelled rows and a measured threshold.",
    }
    report_stem = args.output.stem.replace("results", "report")
    args.output.with_name(f"{report_stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
