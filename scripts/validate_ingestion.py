from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate task-1 ingestion artifacts")
    parser.add_argument("artifact_dir", type=Path, nargs="?", default=Path("artifacts"))
    parser.add_argument("--strict", action="store_true", help="exit with code 1 when a core acceptance gate fails")
    args = parser.parse_args()

    report_path = args.artifact_dir / "ingestion_quality_report.json"
    if not report_path.exists():
        raise SystemExit(f"quality report not found: {report_path}. Re-run scripts/ingest_corpus.py first.")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    acceptance = report.get("acceptance", {})
    failed = [name for name, passed in acceptance.items() if not passed]

    print(json.dumps({
        "artifact_dir": str(args.artifact_dir.resolve()),
        "acceptance": acceptance,
        "failed_core_gates": failed,
        "document_metadata": report.get("documents", {}),
        "text_evidence": report.get("text_evidence", {}),
        "table_evidence": report.get("table_evidence", {}),
        "issues": report.get("issues", {}),
    }, ensure_ascii=False, indent=2))

    if args.strict and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
