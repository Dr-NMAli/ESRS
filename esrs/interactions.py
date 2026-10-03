"""User–item interaction simulation + sustainable popularity."""
from __future__ import annotations
from collections import defaultdict
import numpy as np

from .config import (
    N_USERS, ZIPF_S, N_INTER_MIN, N_INTER_MAX, N_INTER_MEAN, TAU,
)


def _sample_histogram(n_users, min_len, max_len, mean_len, rng):
    """Gamma-shaped histogram clipped to [min_len, max_len], tuned to hit the
    paper's mean of ~43.7 interactions per user."""
    shape = 2.0
    scale = max(1.0, (mean_len - min_len) / shape)
    raw = rng.gamma(shape, scale, size=n_users) + min_len
    return np.clip(raw.astype(int), min_len, max_len)


def simulate_interactions(items, n_users=N_USERS, seed=42):
    """Zipf-1.2 category preference per user, distinct items, histogram
    of history lengths."""
    rng = np.random.default_rng(seed)

    by_cat = defaultdict(list)
    for idx, it in enumerate(items):
        by_cat[it["cat"]].append(idx)

    # Zipf weights over items within each category
    cat_weights = {}
    for cat, idxs in by_cat.items():
        w = 1.0 / np.power(np.arange(1, len(idxs) + 1), ZIPF_S)
        cat_weights[cat] = w / w.sum()

    user_lens = _sample_histogram(n_users, N_INTER_MIN, N_INTER_MAX,
                                  N_INTER_MEAN, rng)
    cat_names = list(by_cat.keys())
    cat_prior = np.array([0.50, 0.30, 0.20])       # Electronics-heavy is plausible

    interactions = []
    for u in range(n_users):
        n = int(user_lens[u])
        # pick a category mix for this user
        mix = rng.dirichlet(cat_prior * 8)          # concentrated-ish
        for _ in range(n):
            cat = rng.choice(cat_names, p=mix)
            idxs = by_cat[cat]
            w = cat_weights[cat]
            it_i = int(rng.choice(idxs, p=w))
            interactions.append((u, it_i))
    return interactions, user_lens


def sustainable_popularity(items, interactions, tau=TAU):
    """pop_s(i) = interactions from sustainability-oriented users, normalized
    within category."""
    by_user = defaultdict(list)
    for u, i in interactions:
        by_user[u].append(i)

    green_users = {
        u for u, its in by_user.items()
        if np.mean([items[i]["eco"] > tau for i in its]) >= 0.30
    }
    print(f"[pop] sustainability-oriented users: {len(green_users)}/{len(by_user)}")

    counts = defaultdict(int)
    for u, i in interactions:
        if u in green_users:
            counts[i] += 1

    by_cat = defaultdict(list)
    for idx, it in enumerate(items):
        by_cat[it["cat"]].append(idx)
    for cat, idxs in by_cat.items():
        m = max((counts[i] for i in idxs), default=1) or 1
        for i in idxs:
            items[i]["pop_s"] = counts[i] / m
    return items