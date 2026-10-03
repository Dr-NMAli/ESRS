"""Scoring rules and the three offline baselines."""
from __future__ import annotations
import numpy as np

from .config import K, TAU


def _unit_rows(E):
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


def score_matrix(P, E, eco, alpha):
    """Score(u,i;α) = α·cos(P_u, E_i) + (1-α)·Eco(i)."""
    cos = P @ _unit_rows(E).T                       # (U, I)
    return alpha * cos + (1.0 - alpha) * eco[None, :]


# ---------------------------------------------------------------------------
# Baseline runners — every baseline is driven by the shared (pos, neg)
# candidate sets so comparisons are apples-to-apples.
# ---------------------------------------------------------------------------
def run_std_rs(P, E, eco, pos_map, neg_map, K=K, tau=TAU):
    """Content-based relevance-only ranking (α = 1)."""
    cos = P @ _unit_rows(E).T

    def score(u, cand):
        return cos[u, cand]

    from .metrics import evaluate
    return evaluate(score, pos_map, neg_map, eco, K=K, tau=tau)


def run_scalarisation(P, E, eco, pos_map, neg_map, alpha, K=K, tau=TAU):
    """Linear scalarisation with trade-off α."""
    S = score_matrix(P, E, eco, alpha)

    def score(u, cand):
        return S[u, cand]

    from .metrics import evaluate
    return evaluate(score, pos_map, neg_map, eco, K=K, tau=tau)


def run_postfilter(P, E, eco, pos_map, neg_map, swap=5, K=K, tau=TAU):
    """Std RS ranking, then swap the top-`swap` slots for the highest-eco
    items present in the candidate pool."""
    cos = P @ _unit_rows(E).T
    green = (eco > tau)

    P_list, R_list, N_list, E_list, G_list = [], [], [], [], []
    for u, pos in pos_map.items():
        cand = np.concatenate([pos, neg_map[u]])
        s = cos[u, cand]
        top = cand[np.argsort(-s)[:K]].copy()

        # highest-eco items in the candidate pool
        top_set = set(top.tolist())
        pool = sorted((i for i in cand if i not in top_set),
                      key=lambda i: -eco[i])
        repl = pool[:swap]
        top[:len(repl)] = repl

        is_pos = np.isin(top, pos)
        hits = int(is_pos.sum())
        P_list.append(hits / K)
        R_list.append(hits / len(pos))
        dcg = float((is_pos / np.log2(np.arange(2, K + 2))).sum())
        ideal_n = min(K, len(pos))
        idcg = float((1.0 / np.log2(np.arange(2, ideal_n + 2))).sum())
        N_list.append(dcg / idcg if idcg > 0 else 0.0)
        E_list.append(float(eco[top].mean()))
        G_list.append(float(green[top].mean()))

    return {
        "P":      float(np.mean(P_list)),
        "R":      float(np.mean(R_list)),
        "NDCG":   float(np.mean(N_list)),
        "AvgEco": float(np.mean(E_list)),
        "GIR":    float(np.mean(G_list)),
    }