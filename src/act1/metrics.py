"""Metrics: QA-F1 (SQuAD-style) + paired bootstrap 95% CIs.
PRD §5a: 'Metric: QA-F1, paired, bootstrap 95% CI on each pairwise delta vs bear.'
"""
from __future__ import annotations
import re
import string
from collections import Counter
from typing import Iterable
import numpy as np


def _normalize(s: str) -> list[str]:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return [w for w in s.split() if w.strip()]


def qa_f1(pred: str, gold: str) -> float:
    """Token-overlap F1 between pred and gold (SQuAD-style)."""
    p, g = _normalize(pred), _normalize(gold)
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    prec = n_common / len(p)
    rec = n_common / len(g)
    return 2 * prec * rec / (prec + rec)


def exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def bootstrap_ci(deltas: Iterable[float], n_iters: int = 1000, seed: int = 42) -> tuple[float, float]:
    """Paired bootstrap 95% CI on a list of per-instance deltas (method A - method B).
    Returns (lower, upper) bounds on the mean delta.
    """
    arr = np.asarray(list(deltas), dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_iters):
        idx = rng.integers(0, len(arr), len(arr))
        means.append(arr[idx].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)
