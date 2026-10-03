"""Average-pooled user preference vectors."""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def build_user_profiles(interactions, embeddings, n_users, exclude=None):
    """Return (P_norm, by_user).

    exclude: dict {user_id: set(item_idx)} of items to omit from the profile
             (used to keep held-out test positives out of the training signal).
    """
    exclude = exclude or {}
    by_user = defaultdict(list)
    for u, i in interactions:
        if i in exclude.get(u, ()):
            continue
        by_user[u].append(i)

    P = np.zeros((n_users, embeddings.shape[1]), dtype=np.float32)
    for u, its in by_user.items():
        if its:
            P[u] = embeddings[its].mean(axis=0)

    norms = np.linalg.norm(P, axis=1, keepdims=True) + 1e-9
    return P / norms, by_user