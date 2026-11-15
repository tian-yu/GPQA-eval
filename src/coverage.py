"""4-gram lazy greedy coverage selector (EESFT, Dr.SCI arXiv:2602.08321).

Selects the k examples that maximize unique 4-gram coverage normalized by
sqrt(word_count), maximizing scientific concept diversity per token without
biasing toward overly long responses.
"""

from __future__ import annotations

import heapq
import math
import re

import numpy as np

WORD_RE = re.compile(r"\w+|[^\w\s]")
TABLE_BITS = 28


def row_text(row: dict) -> str:
    """Extract plain text from messages or chosen/prompt fields for n-gram hashing."""
    msgs = row.get("messages") or row.get("chosen") or row.get("prompt")
    if isinstance(msgs, list):
        return "\n".join(str(m.get("content", "")) for m in msgs if isinstance(m, dict))
    return str(row.get("question", "")) + "\n" + str(row.get("answer", ""))


def ngram_hashes(text: str, n: int = 4, bits: int = TABLE_BITS) -> tuple[np.ndarray, int]:
    """Return unique hashed n-grams of lower-cased tokens and the token count."""
    words = WORD_RE.findall(text.lower())
    if len(words) < n:
        return np.zeros(0, dtype=np.int64), len(words)
    ids = np.array([hash(w) & 0xFFFFFFFF for w in words], dtype=np.uint64)
    h = np.zeros(len(words) - n + 1, dtype=np.uint64)
    for k in range(n):
        h = h * np.uint64(1_000_003) + ids[k : len(words) - n + 1 + k]
    h ^= h >> np.uint64(29)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(32)
    mask = np.uint64((1 << bits) - 1)
    return np.unique((h & mask).astype(np.int64)), len(words)


def select_indices(texts: list[str], k: int, norm: str = "sqrt", n: int = 4, bits: int = TABLE_BITS) -> list[int]:
    """Lazily select k indices maximizing normalized new n-gram coverage."""
    if k >= len(texts):
        return list(range(len(texts)))
    grams, scale = [], []
    for t in texts:
        g, nw = ngram_hashes(t, n, bits)
        grams.append(g)
        if norm == "none":
            scale.append(1.0)
        elif norm == "sqrt":
            scale.append(math.sqrt(max(nw, 1)))
        else:
            scale.append(float(max(nw, 1)))

    covered = np.zeros(1 << bits, dtype=bool)
    heap = [(-len(g) / s, i) for i, (g, s) in enumerate(zip(grams, scale))]
    heapq.heapify(heap)
    chosen: list[int] = []
    while heap and len(chosen) < k:
        _, i = heapq.heappop(heap)
        gain = int(np.count_nonzero(~covered[grams[i]])) / scale[i]
        if heap and gain < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-gain, i))
            continue
        chosen.append(i)
        covered[grams[i]] = True
    return chosen


def select_by_part(rows: list[dict], targets: dict[str, int], norm: str = "sqrt") -> tuple[list[dict], dict[str, dict]]:
    """Select targets[part] rows per part using greedy 4-gram coverage."""
    by_part: dict[str, list[dict]] = {}
    for r in rows:
        by_part.setdefault(str(r.get("part", "default")), []).append(r)

    selected: list[dict] = []
    stats: dict[str, dict] = {}
    for part, target in targets.items():
        pool = by_part.get(part, [])
        idx = select_indices([row_text(r) for r in pool], target, norm=norm)
        picked = [pool[i] for i in idx]
        selected.extend(picked)
        stats[part] = {
            "pool": len(pool),
            "target": target,
            "selected": len(picked),
            "tokens": sum(int(r.get("n_tokens", 0)) for r in picked),
        }
    return selected, stats
