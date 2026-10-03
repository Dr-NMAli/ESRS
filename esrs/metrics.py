"""Offline ranking metrics and the α-sweep helper."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import (
    K, TAU, SEED, N_CANDIDATES, N_POS_PER_USER,
)


# ---------------------------------------------------------------------------
# Candidate-set construction (shared across all systems)
# ---------------------------------------------------------------------------
def sample_candidates(by_user, n_items, n_pos=N_POS_PER_USER,
                      n_cand=N_CANDIDATES, seed=SEED):
    rng = np.random.default_rng(seed)
    pos_map, neg_map = {}, {}
    all_items = np.arange(n_items)
    for u, hist in by_user.items():
        if len(hist) < n_pos + 1:
            continue
        hist_arr = np.asarray(hist)
        pos = rng.choice(hist_arr, size=n_pos, replace=False)
        neg_pool = np.setdiff1d(all_items, hist_arr, assume_unique=False)
        n_neg = min(n_cand - n_pos, len(neg_pool))
        negs = rng.choice(neg_pool, size=n_neg, replace=False)
        pos_map[u] = pos
        neg_map[u] = negs
    return pos_map, neg_map


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------
def evaluate(score_fn, pos_map, neg_map, eco, K=K, tau=TAU):
    """score_fn(u, candidate_indices) -> array of scores.

    Metrics: Precision@K, Recall@K, NDCG@K, mean eco of the top-K list,
             and Green Item Ratio.
    """
    green = (eco > tau)
    Ps, Rs, Ns, Es, Gs = [], [], [], [], []
    for u, pos in pos_map.items():
        cand = np.concatenate([pos, neg_map[u]])
        s = score_fn(u, cand)
        order = np.argsort(-s, kind="stable")[:K]
        top = cand[order]

        is_pos = np.isin(top, pos)
        hits = int(is_pos.sum())
        Ps.append(hits / K)
        Rs.append(hits / len(pos))

        # binary-relevance NDCG@K
        disc = 1.0 / np.log2(np.arange(2, K + 2))
        dcg = float((is_pos * disc).sum())
        n_ideal = min(K, int(is_pos.sum()) if hits > 0 else 0)
        # proper ideal: the best possible ordering is min(K, |pos|) positives on top
        n_ideal = min(K, len(pos))
        idcg = float((1.0 / np.log2(np.arange(2, n_ideal + 2))).sum())
        Ns.append(dcg / idcg if idcg > 0 else 0.0)

        Es.append(float(eco[top].mean()))
        Gs.append(float(green[top].mean()))

    return {
        "P":      float(np.mean(Ps)),
        "R":      float(np.mean(Rs)),
        "NDCG":   float(np.mean(Ns)),
        "AvgEco": float(np.mean(Es)),
        "GIR":    float(np.mean(Gs)),
    }


# ---------------------------------------------------------------------------
# α-sweep on the validation split
# ---------------------------------------------------------------------------
def alpha_sweep(P, E, eco, pos_map, neg_map,
                alphas=np.arange(0.0, 1.01, 0.1)):
    from .ranking import run_scalarisation
    rows = []
    for a in alphas:
        m = run_scalarisation(P, E, eco, pos_map, neg_map, alpha=float(a))
        rows.append((float(a), m["P"], m["NDCG"], m["AvgEco"], m["GIR"]))
        print(f"[α={a:.1f}]  P@10={m['P']:.3f}  NDCG@10={m['NDCG']:.3f}  "
              f"Eco={m['AvgEco']:.3f}  GIR={m['GIR']:.3f}")
    return pd.DataFrame(rows, columns=["alpha", "P@10", "NDCG@10", "AvgEco", "GIR"])