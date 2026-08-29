from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bankreg_trustrag.config import Settings
from bankreg_trustrag.query_plan import QueryPlan
from bankreg_trustrag.query_planner import QueryPlanner


QUESTION = (
    "2023年10月人身险和财产险公司的原保险保费收入加起来是多少，"
    "和全国总数差多少？"
)


def validate_cross_file_plan(plan: QueryPlan) -> list[str]:
    errors: list[str] = []
    tasks = {task.id: task for task in plan.retrieval_tasks}
    if set(tasks) != {"r1", "r2", "r3"}:
        errors.append(f"retrieval ids must be r1/r2/r3, got {sorted(tasks)}")
    task_text = {
        task_id: " ".join(filter(None, [
            task.query,
            task.expected_information,
            task.source_scope.document_title,
            task.semantic_constraints.institution,
            task.semantic_constraints.statistical_scope,
            task.semantic_constraints.row_label,
        ]))
        for task_id, task in tasks.items()
    }
    expected_markers = {"r1": "人身", "r2": "财产", "r3": "全国"}
    for task_id, marker in expected_markers.items():
        if task_id in task_text and marker not in task_text[task_id]:
            errors.append(f"{task_id} does not represent {marker}")

    operations = {operation.output_id: operation for operation in plan.operations}
    total = operations.get("calc1")
    difference = operations.get("calc2")
    if total is None or total.type != "sum" or total.input_refs() != ["r1", "r2"]:
        errors.append("calc1 must be sum(r1, r2)")
    if difference is None or difference.type != "subtract":
        errors.append("calc2 must be subtract(calc1, r3)")
    elif difference.input_refs() != ["calc1", "r3"] or difference.parameters.get("absolute") is not True:
        errors.append("calc2 must be abs(calc1 - r3)")
    return errors


def run_stability_check(runs: int = 5) -> dict[str, Any]:
    settings = Settings.from_env(Path.cwd())
    planner = QueryPlanner.from_settings(settings)
    attempts: list[dict[str, Any]] = []
    for index in range(1, runs + 1):
        outcome = planner.plan(QUESTION)
        errors = [] if outcome.status != "ok" else validate_cross_file_plan(outcome.plan)
        passed = outcome.status == "ok" and not errors
        attempts.append({
            "run": index,
            "passed": passed,
            "status": outcome.status,
            "request_attempts": outcome.attempts,
            "error": outcome.error,
            "validation_errors": errors,
            "retrieval_ids": [task.id for task in outcome.plan.retrieval_tasks],
            "operations": [
                {
                    "type": operation.type,
                    "output_id": operation.output_id,
                    "inputs": operation.input_refs(),
                    "absolute": operation.parameters.get("absolute"),
                }
                for operation in outcome.plan.operations
            ],
        })
    passed_count = sum(1 for item in attempts if item["passed"])
    success_rate = passed_count / max(runs, 1)
    return {
        "question": QUESTION,
        "planner_model": settings.planner_model or settings.llm_model,
        "runs": runs,
        "passed": passed_count,
        "success_rate": success_rate,
        "target": 0.95,
        "target_met": success_rate >= 0.95,
        "attempts": attempts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the compact query planner repeatedly.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("artifacts/planner_stability_report.json"))
    args = parser.parse_args()
    report = run_stability_check(max(1, args.runs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["target_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
