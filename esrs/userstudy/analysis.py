"""Statistical analysis.

Reads a Store and reports:
    * per-condition means for CTR, PI, Trust, Transparency
    * one-way ANOVA across conditions (per metric)
    * pairwise two-sample t-tests with Bonferroni correction (α=.0167)
    * Cohen's d for each pairwise comparison

If ``statsmodels`` is installed, Tukey HSD is also reported.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

CONDITIONS = ("A", "B", "C")
METRICS = ("CTR", "Purchase Intention", "Trust", "Perceived Transparency")


def _per_participant(parts):
    """Assemble a wide DataFrame: one row per participant, one column per metric."""
    p = parts["participants"][["pid", "condition"]].copy()
    r = parts["ratings"]

    # CTR is stored as a rating with metric name "_ctr_sustainable"
    ctr = (r[r["metric"] == "_ctr_sustainable"][["pid", "value"]]
           .rename(columns={"value": "CTR"}))
    likert = r[r["metric"] != "_ctr_sustainable"].pivot(
        index="pid", columns="metric", values="value")

    wide = p.merge(ctr, on="pid", how="left").merge(likert, on="pid", how="left")
    rename = {"PI": "Purchase Intention",
              "Trust": "Trust",
              "Transparency": "Perceived Transparency"}
    wide = wide.rename(columns=rename)
    return wide


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2))
    if sp == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / sp)


def analyse(store):
    parts = store.as_dataframes()
    wide = _per_participant(parts)

    print(f"[analysis] participants: {len(wide)}")
    print(wide["condition"].value_counts().to_string())

    # -------------------------------------------
    print("\n=== Table 10 — means by condition ===")
    agg = wide.groupby("condition")[list(METRICS)].mean().round(3)
    print(agg.to_string())

    # ---------- one-way ANOVA ----------
    print("\n=== One-way ANOVA (across A, B, C) ===")
    for m in METRICS:
        groups = [wide.loc[wide["condition"] == c, m].dropna().values
                  for c in CONDITIONS]
        if any(len(g) < 2 for g in groups):
            continue
        F, p = stats.f_oneway(*groups)
        print(f"{m:<24} F={F:6.2f}  p={p:.4f}")

    # ---------- Table 11 — pairwise t-tests, Bonferroni ----------
    print("\n=== Table 11 — pairwise comparisons (Bonferroni α=.05/3≈.0167) ===")
    pairs = [("B", "A"), ("C", "B"), ("C", "A")]
    for m in METRICS:
        print(f"\n{m}")
        for x, y in pairs:
            gx = wide.loc[wide["condition"] == x, m].dropna().values
            gy = wide.loc[wide["condition"] == y, m].dropna().values
            if len(gx) < 2 or len(gy) < 2:
                continue
            t, p = stats.ttest_ind(gx, gy, equal_var=False)
            d = cohens_d(gx, gy)
            sig = "***" if p < .001 else ("**" if p < .01 else
                                          ("*" if p < .0167 else "n.s."))
            print(f"  {x} vs {y}:  Δmean={gx.mean() - gy.mean():+.3f}  "
                  f"t={t:+.2f}  p={p:.4f}  d={d:+.2f}  [{sig}]")

    # ---------- Tukey HSD (optional) ----------
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        print("\n=== Tukey HSD (reported by paper as the post-hoc test) ===")
        for m in METRICS:
            df = wide[["condition", m]].dropna()
            res = pairwise_tukeyhsd(df[m].values, df["condition"].values,
                                    alpha=0.05)
            print(f"\n{m}\n{res.summary()}")
    except ImportError:
        print("\n[info] statsmodels not installed; skipping Tukey HSD "
              "(pip install statsmodels)")

    # ---------- Explanation-type breakdown (condition C) ----------
    c_only = parts["participants"].query("condition == 'C'")["pid"].tolist()
    imp = parts["impressions"][parts["impressions"]["pid"].isin(c_only)]
    clk = parts["clicks"]
    if not imp.empty and "explanation_type" in imp.columns:
        joined = imp.merge(clk[["pid", "task_id", "item_id"]],
                           on=["pid", "task_id", "item_id"], how="left",
                           indicator=True)
        joined["clicked"] = (joined["_merge"] == "both").astype(int)
        print("\n=== Table 12 — CTR by explanation type (condition C) ===")
        rows = []
        for et, sub in joined.dropna(subset=["explanation_type"]).groupby(
                "explanation_type"):
            if len(sub):
                rows.append((et, len(sub), sub["clicked"].mean()))
        for et, n, ctr in sorted(rows, key=lambda r: r[2]):
            print(f"  {et:<14} n={n:>5}  CTR={ctr:.3f}")
    return wide


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(db_path="esrs_userstudy.sqlite"):
    from .store import Store
    store = Store(db_path)
    return analyse(store)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="esrs_userstudy.sqlite")
    args = ap.parse_args()
    main(args.db)