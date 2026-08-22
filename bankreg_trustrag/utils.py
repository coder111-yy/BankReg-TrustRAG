from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: object) -> str:
    raw = "|".join(str(p) for p in parts)
    return f"{prefix}:{hashlib.sha1(raw.encode('utf-8', 'ignore')).hexdigest()[:16]}"


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    text = normalize_text(text).lower()
    # Preserve Chinese characters as single terms and group latin/numeric terms.
    return re.findall(r"[\u4e00-\u9fff]|[a-z]+|\d+(?:\.\d+)?%?", text)


def char_ngrams(text: str, n: int = 2) -> set[str]:
    clean = re.sub(r"\s+", "", normalize_text(text).lower())
    if len(clean) <= n:
        return {clean} if clean else set()
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def normalized_number(value: object) -> float | None:
    if value is None:
        return None
    text = normalize_text(value).replace(",", "")
    percent = "%" in text or "％" in text
    text = re.sub(r"[^0-9.\-]", "", text)
    if not text or text in {".", "-", "-."}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number / 100 if percent else number


def numbers_in(text: str) -> list[float]:
    values: list[float] = []
    for match in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?\s*%?", normalize_text(text)):
        value = normalized_number(match)
        if value is not None:
            values.append(value)
    return values


def compact_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def first_nonempty(values: Iterable[object]) -> str | None:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return None

