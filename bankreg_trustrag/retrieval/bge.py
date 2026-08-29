"""Local BGE embedding and reranking components.

The module deliberately imports sentence-transformers lazily.  This keeps the
deterministic test path usable when the optional ML runtime is not installed,
while the production path uses the configured BGE models locally.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class BGEConfig:
    mode: str = "auto"  # auto, required, disabled
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    cache_dir: Path | None = None
    vector_dir: Path | None = None
    device: str | None = None
    batch_size: int = 32
    max_length: int = 512
    local_files_only: bool = False
    rerank_top_k: int = 32


class BGEUnavailable(RuntimeError):
    """Raised when BGE is required but its runtime/model is unavailable."""


class BGEPipeline:
    """Lazy local BGE-M3 + BGE-Reranker pipeline."""

    def __init__(self, config: BGEConfig):
        self.config = config
        self._embedder: Any | None = None
        self._reranker: Any | None = None
        self._error: str | None = None

    @property
    def enabled(self) -> bool:
        return self.config.mode != "disabled"

    @property
    def embedding_loaded(self) -> bool:
        """Whether the embedding model has actually loaded successfully."""
        return self._embedder is not None

    @property
    def reranker_loaded(self) -> bool:
        """Whether the cross-encoder has actually loaded successfully."""
        return self._reranker is not None

    @property
    def status(self) -> dict[str, Any]:
        return {
            "mode": self.config.mode,
            "embedding_model": self.config.embedding_model,
            "reranker_model": self.config.reranker_model,
            "embedding_loaded": self.embedding_loaded,
            "reranker_loaded": self.reranker_loaded,
            "embedding_available": self.embedding_loaded,
            "reranker_available": self.reranker_loaded,
            "reranker_score_calibration": "sigmoid(logit) for binary relevance",
            "error": self._error,
        }

    def _fail(self, message: str) -> None:
        self._error = message
        if self.config.mode == "required":
            raise BGEUnavailable(message)

    def _model_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"local_files_only": self.config.local_files_only}
        if self.config.cache_dir:
            kwargs["cache_folder"] = str(self.config.cache_dir)
        if self.config.device:
            kwargs["device"] = self.config.device
        return kwargs

    def _model_source(self, model_name: str) -> str | None:
        """Resolve a cached HF repo to a snapshot when offline mode is used."""
        candidate = Path(model_name).expanduser()
        if candidate.exists():
            return str(candidate)
        if self.config.local_files_only and self.config.cache_dir and "/" in model_name:
            repo_dir = self.config.cache_dir / f"models--{model_name.replace('/', '--')}"
            snapshots = sorted((repo_dir / "snapshots").glob("*"))
            if snapshots:
                snapshot = snapshots[-1]
                weights = list(snapshot.rglob("*.safetensors")) + list(snapshot.rglob("pytorch_model*.bin"))
                if weights:
                    return str(snapshot)
                return None
            return None
        return model_name

    def _load_embedder(self) -> Any | None:
        if self._embedder is not None:
            return self._embedder
        if self._error and self.config.mode != "required":
            return None
        if not self.enabled:
            return None
        try:
            from sentence_transformers import SentenceTransformer

            kwargs = self._model_kwargs()
            # Newer sentence-transformers forwards local_files_only through
            # model_kwargs; older versions accept it directly. Try the modern
            # form first, then the compatible fallback.
            source = self._model_source(self.config.embedding_model)
            if source is None:
                self._fail(
                    f"BGE embedding model is not cached locally: {self.config.embedding_model}. "
                    "Download it first or set BANKREG_BGE_LOCAL_FILES_ONLY=0."
                )
                return None
            try:
                self._embedder = SentenceTransformer(
                    source,
                    model_kwargs={"local_files_only": self.config.local_files_only},
                    cache_folder=kwargs.get("cache_folder"),
                    device=kwargs.get("device"),
                )
            except TypeError:
                self._embedder = SentenceTransformer(
                    source,
                    cache_folder=kwargs.get("cache_folder"),
                    device=kwargs.get("device"),
                )
            self._embedder.max_seq_length = self.config.max_length
            return self._embedder
        except Exception as exc:  # model/runtime errors must be explicit
            self._fail(
                f"BGE embedding unavailable ({self.config.embedding_model}): "
                f"{type(exc).__name__}: {exc}. Install sentence-transformers/torch "
                "and download the model locally, or set BANKREG_BGE_MODE=disabled."
            )
            return None

    def _load_reranker(self) -> Any | None:
        if self._reranker is not None:
            return self._reranker
        if self._error and self.config.mode != "required":
            return None
        if not self.enabled:
            return None
        try:
            from sentence_transformers import CrossEncoder

            kwargs: dict[str, Any] = {"max_length": self.config.max_length}
            if self.config.device:
                kwargs["device"] = self.config.device
            # CrossEncoder versions differ in how model_kwargs are accepted.
            source = self._model_source(self.config.reranker_model)
            if source is None:
                self._fail(
                    f"BGE reranker model is not cached locally: {self.config.reranker_model}. "
                    "Download it first or set BANKREG_BGE_LOCAL_FILES_ONLY=0."
                )
                return None
            try:
                self._reranker = CrossEncoder(
                    source,
                    model_kwargs={"local_files_only": self.config.local_files_only},
                    **kwargs,
                )
            except TypeError:
                self._reranker = CrossEncoder(source, **kwargs)
            return self._reranker
        except Exception as exc:
            self._fail(
                f"BGE reranker unavailable ({self.config.reranker_model}): "
                f"{type(exc).__name__}: {exc}. Install sentence-transformers/torch "
                "and download the model locally, or set BANKREG_BGE_MODE=disabled."
            )
            return None

    def encode(self, texts: Sequence[str]) -> Any | None:
        model = self._load_embedder()
        if model is None or not texts:
            return None
        return model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def similarity(self, query: str, texts: Sequence[str]) -> list[float] | None:
        vectors = self.encode([query, *texts])
        if vectors is None:
            return None
        query_vector = vectors[0]
        return [float(vector @ query_vector) for vector in vectors[1:]]

    def rerank(self, query: str, texts: Sequence[str]) -> list[float] | None:
        model = self._load_reranker()
        if model is None or not texts:
            return None
        pairs = [(query, text) for text in texts]
        # BGE cross-encoders are binary relevance models.  Ask the installed
        # sentence-transformers version to apply sigmoid explicitly instead of
        # guessing from the numeric range of the output: a logit can itself be
        # between 0 and 1, so range-based detection is not reliable.
        used_explicit_activation = True
        try:
            import torch

            raw = model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                activation_fn=torch.sigmoid,
            )
        except TypeError:
            # Compatibility with older CrossEncoder implementations that do
            # not expose activation_fn. Their output is handled defensively
            # below, but this path is explicitly a compatibility fallback.
            used_explicit_activation = False
            raw = model.predict(
                pairs,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
            )
        result: list[float] = []
        for value in raw:
            # Some versions return one-element arrays/tensors for a
            # single-label model. Convert those to a scalar consistently.
            if hasattr(value, "item"):
                value = value.item()
            elif isinstance(value, (list, tuple)):
                value = value[0] if value else 0.0
            score = float(value)
            # Current sentence-transformers accepts activation_fn and has
            # already applied sigmoid. Older versions return the raw binary
            # logit by default; calibrate that path with the same sigmoid.
            if not used_explicit_activation:
                score = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, score))))
            result.append(max(0.0, min(1.0, score)))
        return result


class PersistentVectorIndex:
    """A persisted cosine-similarity index backed by normalized NumPy vectors.

    FAISS is not mandatory for correctness here: normalized matrix dot products
    are exact cosine search and keep the project portable. The manifest records
    the model and corpus fingerprint so stale vectors are never silently reused.
    """

    def __init__(self, pipeline: BGEPipeline, directory: Path, name: str):
        self.pipeline = pipeline
        self.directory = directory
        self.name = name
        self.vectors: Any | None = None
        self.item_ids: list[str] = []
        self._id_to_index: dict[str, int] = {}
        self.manifest: dict[str, Any] = {}

    @property
    def available(self) -> bool:
        return self.vectors is not None and bool(self.item_ids)

    def _paths(self) -> tuple[Path, Path]:
        return self.directory / f"{self.name}.npy", self.directory / f"{self.name}.json"

    @staticmethod
    def _fingerprint(items: Sequence[dict[str, Any]], text_fn: Callable[[dict[str, Any]], str]) -> str:
        digest = hashlib.sha256()
        for item in items:
            digest.update(str(item.get("evidence_id", "")).encode("utf-8"))
            digest.update(b"\0")
            digest.update(text_fn(item).encode("utf-8", "ignore"))
            digest.update(b"\n")
        return digest.hexdigest()

    def load(self, items: Sequence[dict[str, Any]], text_fn: Callable[[dict[str, Any]], str]) -> bool:
        vector_path, manifest_path = self._paths()
        if not vector_path.exists() or not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = self._fingerprint(items, text_fn)
            if manifest.get("corpus_fingerprint") != expected or manifest.get("model") != self.pipeline.config.embedding_model:
                return False
            import numpy as np

            vectors = np.load(vector_path, mmap_mode="r")
            item_ids = list(manifest.get("item_ids", []))
            if len(item_ids) != len(vectors):
                return False
            self.vectors = vectors
            self.item_ids = item_ids
            self._id_to_index = {value: index for index, value in enumerate(item_ids)}
            self.manifest = manifest
            return True
        except Exception:
            return False

    def build_or_load(self, items: Sequence[dict[str, Any]], text_fn: Callable[[dict[str, Any]], str]) -> bool:
        if not self.pipeline.enabled or not items:
            return False
        if self.load(items, text_fn):
            return True
        vectors = self.pipeline.encode([text_fn(item) for item in items])
        if vectors is None:
            return False
        self.directory.mkdir(parents=True, exist_ok=True)
        vector_path, manifest_path = self._paths()
        import numpy as np

        np.save(vector_path, np.asarray(vectors, dtype="float32"))
        self.item_ids = [str(item.get("evidence_id")) for item in items]
        self._id_to_index = {value: index for index, value in enumerate(self.item_ids)}
        self.vectors = np.load(vector_path, mmap_mode="r")
        self.manifest = {
            "model": self.pipeline.config.embedding_model,
            "dimension": int(self.vectors.shape[1]),
            "count": len(self.item_ids),
            "item_ids": self.item_ids,
            "corpus_fingerprint": self._fingerprint(items, text_fn),
        }
        manifest_path.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return True

    def search(self, query: str, top_k: int, allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        if not self.available:
            return []
        vector = self.pipeline.encode([query])
        if vector is None:
            return []
        scores = self.vectors @ vector[0]
        candidates = [
            (index, float(score))
            for index, score in enumerate(scores)
            if allowed_ids is None or self.item_ids[index] in allowed_ids
        ]
        candidates.sort(key=lambda item: item[1], reverse=True)
        return [(self.item_ids[index], score) for index, score in candidates[:top_k]]
