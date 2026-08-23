from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    artifact_dir: Path
    db_path: Path
    top_k: int = 8
    min_trust: float = 0.58
    bge_mode: str = "auto"
    bge_embedding_model: str = "BAAI/bge-small-zh-v1.5"
    bge_reranker_model: str = "BAAI/bge-reranker-base"
    bge_cache_dir: Path | None = None
    bge_vector_dir: Path | None = None
    bge_device: str | None = None
    bge_batch_size: int = 32
    bge_max_length: int = 512
    bge_local_files_only: bool = True
    bge_rerank_top_k: int = 32
    llm_provider: str = "none"
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_timeout_seconds: float = 45.0
    llm_max_tokens: int = 800
    llm_temperature: float = 0.0
    llm_max_context_chars: int = 12000

    @classmethod
    def from_env(cls, root: Path | None = None) -> "Settings":
        root = (root or Path.cwd()).resolve()
        try:
            from dotenv import load_dotenv

            load_dotenv(root / ".env", override=False)
        except ImportError:
            pass
        data_value = os.getenv("BANKREG_DATA_DIR")
        if data_value:
            data_dir = Path(data_value)
            if not data_dir.is_absolute():
                data_dir = root / data_dir
        else:
            candidates = sorted(root.glob("03-*"))
            data_dir = candidates[0] if candidates else root / "data"
        artifact_value = os.getenv("BANKREG_ARTIFACT_DIR", "artifacts")
        artifact_dir = Path(artifact_value)
        if not artifact_dir.is_absolute():
            artifact_dir = root / artifact_dir
        db_value = os.getenv("BANKREG_DB")
        db_path = Path(db_value) if db_value else artifact_dir / "bankreg.sqlite3"
        if not db_path.is_absolute():
            db_path = root / db_path
        cache_value = os.getenv("BANKREG_BGE_CACHE_DIR")
        cache_dir = Path(cache_value).expanduser() if cache_value else None
        if cache_dir and not cache_dir.is_absolute():
            cache_dir = root / cache_dir
        vector_value = os.getenv("BANKREG_BGE_VECTOR_DIR")
        vector_dir = Path(vector_value).expanduser() if vector_value else artifact_dir / "bge_vectors"
        if not vector_dir.is_absolute():
            vector_dir = root / vector_dir
        return cls(
            data_dir=data_dir.resolve(),
            artifact_dir=artifact_dir.resolve(),
            db_path=db_path.resolve(),
            top_k=int(os.getenv("BANKREG_TOP_K", "8")),
            min_trust=float(os.getenv("BANKREG_MIN_TRUST", "0.58")),
            bge_mode=os.getenv("BANKREG_BGE_MODE", "auto").strip().lower(),
            bge_embedding_model=os.getenv("BANKREG_BGE_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            bge_reranker_model=os.getenv("BANKREG_BGE_RERANKER_MODEL", "BAAI/bge-reranker-base"),
            bge_cache_dir=cache_dir.resolve() if cache_dir else None,
            bge_vector_dir=vector_dir.resolve(),
            bge_device=os.getenv("BANKREG_BGE_DEVICE") or None,
            bge_batch_size=int(os.getenv("BANKREG_BGE_BATCH_SIZE", "32")),
            bge_max_length=int(os.getenv("BANKREG_BGE_MAX_LENGTH", "512")),
            bge_local_files_only=os.getenv("BANKREG_BGE_LOCAL_FILES_ONLY", "1").lower() in {"1", "true", "yes"},
            bge_rerank_top_k=int(os.getenv("BANKREG_BGE_RERANK_TOP_K", "32")),
            llm_provider=os.getenv("BANKREG_LLM_PROVIDER", "none").strip().lower(),
            llm_model=os.getenv("BANKREG_LLM_MODEL") or None,
            llm_base_url=os.getenv("BANKREG_LLM_BASE_URL") or None,
            # Keep the generic name as the canonical setting, while accepting
            # DeepSeek's familiar name so an existing .env does not silently
            # omit the Authorization header.
            llm_api_key=os.getenv("BANKREG_LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or None,
            llm_timeout_seconds=float(os.getenv("BANKREG_LLM_TIMEOUT_SECONDS", os.getenv("BANKREG_LLM_TIMEOUT", "45"))),
            llm_max_tokens=int(os.getenv("BANKREG_LLM_MAX_TOKENS", "800")),
            llm_temperature=float(os.getenv("BANKREG_LLM_TEMPERATURE", "0")),
            llm_max_context_chars=int(os.getenv("BANKREG_LLM_MAX_CONTEXT_CHARS", os.getenv("BANKREG_CONTEXT_MAX_CHARS", "12000"))),
        )
