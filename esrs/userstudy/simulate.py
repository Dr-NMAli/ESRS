"""Headless simulator: 150 synthetic participants, one DB, in seconds.

Purpose: exercise the analysis pipeline end-to-end without recruiting
participants.
"""
from __future__ import annotations
import os, random
import numpy as np

from .store import Store
from .tasks import TASKS, CONDITIONS, grid_order

# Latent means/SDs per condition (metric × condition).
LATENT = {
    "CTR":          {"A": (0.18, 0.12), "B": (0.31, 0.14), "C": (0.58, 0.16)},
    "PI":           {"A": (3.20, 0.90), "B": (3.40, 0.90), "C": (4.10, 0.80)},
    "Trust":        {"A": (3.50, 0.90), "B": (3.30, 0.90), "C": (4.30, 0.75)},
    "Transparency": {"A": (2.80, 0.90), "B": (2.60, 0.90), "C": (4.50, 0.75)},
    "Persuasiveness": {"C": (4.00, 0.70)},
    "Specificity":    {"C": (4.10, 0.65)},
    "Coherence":      {"C": (4.10, 0.65)},
    "Fidelity":       {"C": (4.00, 0.70)},
}
N_PER_COND = 50

# Per-category green fraction across the three tasks.
GREEN_FRAC = {"A": 0.21, "B": 0.68, "C": 0.68}
ITEMS_PER_TASK = 10
N_TASKS = len(TASKS)


def _clip_round(x, lo=1, hi=5):
    return int(round(min(hi, max(lo, x))))


def _one_participant(store, condition, pid_seed):
    rng = np.random.default_rng(pid_seed)
    pid = store.create_participant(condition)
    # ---- impressions + clicks ----
    green_shown, green_clicked = 0, 0
    for task in TASKS:
        n_green = int(rng.binomial(ITEMS_PER_TASK, GREEN_FRAC[condition]))
        order = grid_order(pid, task["id"], ITEMS_PER_TASK, seed=pid_seed)
        green_positions = set(order[:n_green])
        ctr_mean, ctr_sd = LATENT["CTR"][condition]
        p_click = float(np.clip(rng.normal(ctr_mean, ctr_sd * 0.5), 0.02, 0.95))
        for pos in range(ITEMS_PER_TASK):
            item = {
                "item_id": f"sim_{pid}_{task['id']}_{pos}",
                "position": pos,
                "eco": 0.8 if pos in green_positions else 0.4,
                "explanation_type": (rng.choice(
                    ["environmental", "economic", "social", "transparent"])
                    if condition == "C" else None),
            }
            store.record_impression(pid, task["id"], item)
            if pos in green_positions:
                green_shown += 1
                if rng.random() < p_click:
                    store.record_click(pid, task["id"], item["item_id"])
                    green_clicked += 1
    # ---- ratings ----
    for metric, per_cond in LATENT.items():
        if metric == "CTR":
            continue
        if condition not in per_cond:
            continue
        mu, sd = per_cond[condition]
        store.record_rating(pid, metric, _clip_round(rng.normal(mu, sd)))
    # ---- demographics ----
    store.finalize_participant(pid, {
        "age": int(np.clip(rng.normal(34.2, 9.8), 16, 80)),
        "gender": rng.choice(["Female", "Male", "Other", "Prefer not to say"],
                             p=[0.54, 0.42, 0.02, 0.02]),
        "country": rng.choice(["Jordan", "Lebanon", "Egypt", "UAE", "KSA"]),
        "shopping_freq": rng.choice(["Weekly", "Monthly", "Rarely"]),
        "sustainability_attitude": int(rng.integers(1, 6)),
        "device": rng.choice(["Laptop", "Desktop", "Phone"]),
        "prior_esrs_exposure": rng.choice(["No", "Not sure"]),
    })
    # expose the per-participant CTR for the analysis stage
    if green_shown > 0:
        store.conn.execute(
            "INSERT OR REPLACE INTO ratings(pid, metric, value) VALUES (?,?,?)",
            (pid, "_ctr_sustainable", green_clicked / green_shown))
        store.conn.commit()
    return pid


def run(db_path="esrs_userstudy_sim.sqlite", seed=2024):
    if os.path.exists(db_path):
        os.remove(db_path)
    store = Store(db_path)
    for c in CONDITIONS:
        for i in range(N_PER_COND):
            _one_participant(store, c, pid_seed=seed + hash((c, i)) % 10_000)
        print(f"[sim] condition {c}: {N_PER_COND} participants inserted")
    store.conn.commit()
    print(f"[sim] wrote {db_path}")
    return db_path


if __name__ == "__main__":
    run()