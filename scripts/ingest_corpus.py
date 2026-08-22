from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bankreg_trustrag.config import Settings
from bankreg_trustrag.ingestion.manifest import build_manifest
from bankreg_trustrag.storage import Store


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse local BankReg corpus into JSONL and SQLite")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()
    settings = Settings.from_env(Path.cwd())
    data_dir = (args.data_dir or settings.data_dir).resolve()
    artifact_dir = (args.artifact_dir or settings.artifact_dir).resolve()
    summary = build_manifest(data_dir, artifact_dir)
    store = Store(settings.db_path if args.artifact_dir is None else artifact_dir / "bankreg.sqlite3")
    store.load_jsonl(artifact_dir)
    summary.update({"sqlite": str(store.path), "sqlite_documents": store.document_count(), "sqlite_text_evidence": store.text_count(), "sqlite_table_evidence": store.table_count()})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
