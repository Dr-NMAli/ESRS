"""End-to-end offline experiment orchestration."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import (
    SEED, ALPHA, K, TAU, D_EMB, N_USERS,
)
from .data import load_catalog
from .interactions import simulate_interactions, sustainable_popularity
from .encoder import build_item_features, train_encoder
from .profiles import build_user_profiles
from .ranking import run_std_rs, run_scalarisation, run_postfilter
from .metrics import sample_candidates, alpha_sweep


def _seed_everything(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed)
    try:
        import torch; torch.manual_seed(seed)
    except ImportError:
        pass


def _dataset_stats(items):
    df = pd.DataFrame([{"cat": it["cat"],
                        "eco": it["eco"],
                        "green": it["eco"] > TAU} for it in items])
    tbl = (df.groupby("cat")
             .agg(items=("eco", "size"),
                  avg_eco=("eco", "mean"),
                  gir=("green", "mean"))
             .round(3))
    print("\n[dataset statistics]")
    print(tbl)
    print(f"overall avg_eco={df['eco'].mean():.3f}  "
          f"overall GIR={df['green'].mean():.3f}")
    return tbl


def main(use_amazon: bool = False):
    _seed_everything()

    # -------- 1. catalog --------
    items = load_catalog(use_amazon=use_amazon)
    _dataset_stats(items)

    # -------- 2. interactions --------
    interactions, user_lens = simulate_interactions(items, N_USERS, SEED)
    print(f"[data] interactions = {len(interactions)}  "
          f"mean/user = {len(interactions) / N_USERS:.1f}")
    items = sustainable_popularity(items, interactions)

    # -------- 3. encoder --------
    X = build_item_features(items)
    print(f"[features] X.shape = {X.shape}")
    E = train_encoder(X, d=D_EMB)

    # -------- 4. candidate sets (shared across all systems) --------
    # Build a temporary "full history" index for candidate sampling
    from collections import defaultdict
    full_hist = defaultdict(list)
    for u, i in interactions:
        full_hist[u].append(i)
    pos_map, neg_map = sample_candidates(full_hist, len(items))
    print(f"[eval] users evaluated: {len(pos_map)}  "
          f"|pos|={len(next(iter(pos_map.values())))}  "
          f"|neg|={len(next(iter(neg_map.values())))}")

    # -------- 5. profiles EXCLUDING the held-out positives --------
    exclude = {u: set(pos.tolist()) for u, pos in pos_map.items()}
    P, by_user = build_user_profiles(interactions, E, N_USERS, exclude=exclude)

    # -------- 6. evaluate the systems --------
    eco = np.asarray([it["eco"] for it in items], dtype=np.float32)

    results = {
        "Std RS":       run_std_rs(P, E, eco, pos_map, neg_map),
        "PostFilter":   run_postfilter(P, E, eco, pos_map, neg_map),
        f"ESRS (α={ALPHA})": run_scalarisation(P, E, eco, pos_map, neg_map, ALPHA),
    }

    print("\n=== Systems (test split) ===")
    header = f"{'System':<16}{'P@10':>8}{'R@10':>8}{'NDCG@10':>10}{'AvgEco':>9}{'GIR':>7}"
    print(header); print("-" * len(header))
    for k, v in results.items():
        print(f"{k:<16}{v['P']:>8.3f}{v['R']:>8.3f}{v['NDCG']:>10.3f}"
              f"{v['AvgEco']:>9.3f}{v['GIR']:>7.3f}")

    # -------- 7. α-sweep --------
    print("\n=== α-sensitivity (Sec. 5.2) ===")
    sw = alpha_sweep(P, E, eco, pos_map, neg_map)
    print("\n", sw.round(3).to_string(index=False))

    # -------- 8. compare with the man --------
    print("\n=== Comparison against the paper (Table 7) ===")
    paper = {
        "Std RS":         (0.72, 0.65, 0.74, 0.48, 0.21),
        f"ESRS (α={ALPHA})": (0.69, 0.62, 0.71, 0.73, 0.68),
    }
    for name, ref in paper.items():
        got = results[name]
        row = (got["P"], got["R"], got["NDCG"], got["AvgEco"], got["GIR"])
        delta = [abs(g - w) for g, w in zip(row, ref)]
        print(f"{name:<16} got={np.round(row, 3)}  paper={ref}  |Δ|={np.round(delta, 3)}")

    return results, sw