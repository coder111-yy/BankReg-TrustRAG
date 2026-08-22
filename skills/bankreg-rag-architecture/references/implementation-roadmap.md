# Recommended Implementation Roadmap

## Phase 0 — Repo baseline

- Inspect existing files, dependencies, DB, and API conventions.
- Create a minimal smoke test.
- Add `.env.example`; never commit real keys.

## Phase 1 — Ingestion foundation

- Manifest ingestion and SHA-256 dedup/version metadata.
- Word/PDF hierarchical parser.
- Excel structured parser with merged headers, units, periods, and cell addresses.
- Persist parsed results and tests.

## Phase 2 — Retrieval baseline

- BM25 retriever.
- Dense retriever.
- Metadata filters.
- Table retriever.
- Common `Evidence` model.

## Phase 3 — Fusion and routing

- Five-type query router.
- RRF fusion.
- Reranker.
- Minimal sufficient evidence selection.

## Phase 4 — Reasoning paths

- Clause reasoner.
- Table tool reasoner with deterministic calculations.
- Cross-file multi-hop reasoner.

## Phase 5 — Trust layer

- Grounded answer prompt.
- Claim splitter.
- Numeric/date/entity/document-number/normative-strength/citation/version checks.
- Retry, clarify, refuse state transitions.

## Phase 6 — Evaluation and UI

- Evaluation harness using the companion `bankreg-rag-evaluation` skill.
- Evidence drawer/source viewer.
- Trust/verification display.
- History/audit screen.

## Phase 7 — Packaging

- Reproducible setup instructions.
- Docker only after local tests are stable.
- Offline/local data path for contest data.
- Demo dataset must be synthetic/public or otherwise permitted.
