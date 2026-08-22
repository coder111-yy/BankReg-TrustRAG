"""Build explicit document relationships from existing local manifest artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bankreg_trustrag.ingestion.manifest import build_document_relations, write_jsonl
from bankreg_trustrag.storage import Store


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build safe local document relations without reparsing source files")
    parser.add_argument("artifact_dir", type=Path, nargs="?", default=Path("artifacts"))
    parser.add_argument("--db", type=Path)
    args = parser.parse_args()
    documents = read_jsonl(args.artifact_dir / "documents.jsonl")
    relations = build_document_relations(documents)
    write_jsonl(args.artifact_dir / "document_relations.jsonl", relations)
    db_path = args.db or args.artifact_dir / "bankreg.sqlite3"
    loaded = Store(db_path).load_relations(relations)
    print(json.dumps({"documents": len(documents), "relations": loaded, "output": str(args.artifact_dir / "document_relations.jsonl")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
