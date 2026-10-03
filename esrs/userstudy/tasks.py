"""The three shopping tasks and the per-condition product slices."""
from __future__ import annotations
import random
import numpy as np
from collections import defaultdict

from ..config import K, TAU, ALPHA
from ..ranking import score_matrix


# Three tasks, mapped to real categories of the synthetic/Amazon catalog.
TASKS = [
    {"id": "coffee",    "label": "Find a coffee maker",         "cat": "Home"},
    {"id": "footwear",  "label": "Buy athletic footwear",       "cat": "Clothing"},
    {"id": "desklamp",  "label": "Select a desk lamp",          "cat": "Electronics"},
]

CONDITIONS = ("A", "B", "C")
CONDITION_LABEL = {
    "A": "Std RS (relevance only, no explanations)",
    "B": "Sust RS (sustainability-aware ranking, no explanations)",
    "C": "ESRS (sustainability-aware ranking + explanations)",
}


def _user_profile_from_catalog(items, embeddings, seed=12345):
    """A single deterministic 'average shopper' profile, so that every
    participant in a given condition sees the same 10 items per task
    (the paper randomises grid *position*, not the item set)."""
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(items), size=min(40, len(items)), replace=False)
    P = embeddings[idxs].mean(axis=0)
    return P / (np.linalg.norm(P) + 1e-9)


def build_task_slices(items, embeddings, alpha_esrs=ALPHA, K=K):
    """Return {task_id: {condition: [item dicts sorted by rank]}}.

    Ranking rule per condition:
        A: pure cosine relevance  (α = 1)
        B: scalarisation α = 0.6, no explanations
        C: same ranking as B, plus explanations
    """
    eco = np.asarray([it["eco"] for it in items], dtype=np.float32)
    P = _user_profile_from_catalog(items, embeddings)

    S_A = score_matrix(P[None, :], embeddings, eco, alpha=1.0)[0]
    S_BC = score_matrix(P[None, :], embeddings, eco, alpha=alpha_esrs)[0]

    cat_median_cf = {}
    by_cat = defaultdict(list)
    for it in items:
        by_cat[it["cat"]].append(it["raw_cf"])
    for c, v in by_cat.items():
        cat_median_cf[c] = float(np.median(v))

    from ..explanations import assign_explanation_type, render_explanation

    slices = {}
    for task in TASKS:
        cat = task["cat"]
        cat_idx = np.array([i for i, it in enumerate(items) if it["cat"] == cat])

        # Top-K under each ranking rule, restricted to the task's category
        top_A = cat_idx[np.argsort(-S_A[cat_idx])[:K]]
        top_BC = cat_idx[np.argsort(-S_BC[cat_idx])[:K]]

        def _pack(top_idx, with_expl):
            out = []
            for i in top_idx:
                it = dict(items[int(i)])
                if with_expl:
                    etype = assign_explanation_type(it)
                    it["explanation_type"] = etype
                    it["explanation"] = render_explanation(it, etype, cat_median_cf)
                else:
                    it["explanation_type"] = None
                    it["explanation"] = None
                out.append(it)
            return out

        slices[task["id"]] = {
            "A": _pack(top_A, with_expl=False),
            "B": _pack(top_BC, with_expl=False),
            "C": _pack(top_BC, with_expl=True),
        }
    return slices


# ---------------------------------------------------------------------------
# Rendering: deterministic per (participant, task) grid order
# ---------------------------------------------------------------------------
def grid_order(participant_id, task_id, n_items, seed=0):
    """Deterministic shuffle so we can reproduce what each participant saw."""
    rng = random.Random(f"{seed}|{participant_id}|{task_id}")
    idx = list(range(n_items))
    rng.shuffle(idx)
    return idx


def product_card_html(item, position, condition):
    """Minimal, self-contained HTML for a single card (no external assets)."""
    title = _esc(item.get("title", "Product"))
    price = item.get("price", 0.0)
    if not price or np.isnan(price):
        price = 9.99
    price_str = f"${price:,.2f}"
    expl = item.get("explanation")
    expl_html = (f'<p class="expl"><strong>Why this?</strong> {_esc(expl)}</p>'
                 if expl else "")
    thumb = _svg_placeholder(item["cat"])
    return f"""
    <div class="card" data-item-id="{item['item_id']}"
         data-green="{int(item['eco'] > TAU)}"
         data-position="{position}">
      <div class="thumb">{thumb}</div>
      <h4>{title}</h4>
      <div class="price">{price_str}</div>
      {expl_html}
      <button type="button" class="click-btn"
              onclick="recordClick(this, '{item['item_id']}')">Select</button>
    </div>
    """


def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _svg_placeholder(cat):
    color = {"Electronics": "#4a90d9", "Home": "#e8a24a",
             "Clothing": "#a863b4"}.get(cat, "#888")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 60 60" '
            f'width="60" height="60"><rect width="60" height="60" '
            f'fill="{color}" rx="6"/></svg>')