"""Run an in-process smoke test for the BankReg-TrustRAG HTTP service.

This avoids requiring a separately running uvicorn process while exercising the
same FastAPI application and routes exposed by server.py.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient

from bankreg_trustrag.api import create_app


def check(name: str, condition: bool, detail: str = "") -> dict[str, object]:
    return {"name": name, "passed": bool(condition), "detail": detail}


def main() -> int:
    started = time.perf_counter()
    client = TestClient(create_app())
    results: list[dict[str, object]] = []

    response = client.get("/")
    results.append(check("frontend_home", response.status_code == 200 and "BankReg-TrustRAG" in response.text, f"HTTP {response.status_code}"))

    response = client.get("/health")
    health = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    results.append(check("health", response.status_code == 200 and health.get("status") == "ok", json.dumps(health, ensure_ascii=False)))
    results.append(check("knowledge_base_loaded", health.get("documents", 0) > 0 and health.get("table_evidence", 0) > 0, json.dumps(health, ensure_ascii=False)))

    response = client.post(
        "/api/qa",
        json={
            "question": "请查询《2023年10月保险业经营情况表.xls》中的原保险保费收入",
        },
    )
    payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    schema_ok = response.status_code == 200 and all(key in payload for key in ["answer", "qa_type", "evidence", "verification", "trust", "trace_id"])
    trust_ok = isinstance(payload.get("trust"), dict) and isinstance(payload.get("trust", {}).get("score"), (int, float))
    results.append(check("qa_contract", schema_ok and trust_ok, f"HTTP {response.status_code}; keys={sorted(payload) if isinstance(payload, dict) else []}"))
    results.append(check("qa_has_traceable_evidence", isinstance(payload.get("evidence"), list), f"evidence_count={len(payload.get('evidence', [])) if isinstance(payload.get('evidence'), list) else 0}"))

    response = client.post("/api/qa", json={})
    invalid_payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    results.append(check("qa_validation", response.status_code == 422 and "detail" in invalid_payload, f"HTTP {response.status_code}"))

    evidence_id = None
    if isinstance(payload.get("evidence"), list) and payload["evidence"]:
        evidence_id = payload["evidence"][0].get("evidence_id")
    if evidence_id:
        response = client.get(f"/api/evidence/{evidence_id}")
        results.append(check("evidence_detail", response.status_code == 200 and response.json().get("evidence_id") == evidence_id, f"HTTP {response.status_code}"))
    else:
        results.append(check("evidence_detail", False, "QA response returned no evidence_id"))

    response = client.get("/api/history?limit=3")
    history = response.json() if response.headers.get("content-type", "").startswith("application/json") else None
    results.append(check("history", response.status_code == 200 and isinstance(history, list), f"HTTP {response.status_code}"))

    failed = [result for result in results if not result["passed"]]
    report = {"passed": not failed, "checks": results, "elapsed_ms": round((time.perf_counter() - started) * 1000, 2)}
    artifact = Path("artifacts")
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "service_smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
