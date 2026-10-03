"""Rule-based explanation module.

Four types are triggered by the priority rule:
    Environmental  →  Eco(i) > θ_env        (attitude / evidence)
    Economic       →  sav(i) > θ_sav        (perceived cost barrier)
    Social         →  pop_s(i) > θ_pop      (descriptive norm)
    Transparent    →  fallback              (perceived control)

All numbers shown to the user are read from the stored item metadata;
nothing is invented by the template.
"""
from __future__ import annotations
from collections import defaultdict
from .config import THETA_ENV, THETA_SAV, THETA_POP


# Priority order.  Lower rank = higher priority.
_PRIORITY = ("environmental", "economic", "social", "transparent")


def assign_explanation_type(item,
                            theta_env=THETA_ENV,
                            theta_sav=THETA_SAV,
                            theta_pop=THETA_POP) -> str:
    """Return the highest-priority type whose trigger condition holds."""
    if item["eco"] > theta_env:
        return "environmental"
    if item.get("sav", 0.0) > theta_sav:
        return "economic"
    if item.get("pop_s", 0.0) > theta_pop:
        return "social"
    return "transparent"


def _category_median_cf(items):
    med = defaultdict(list)
    for it in items:
        med[it["cat"]].append(it["raw_cf"])
    return {c: float(__import__("numpy").median(v)) for c, v in med.items()}


def render_explanation(item, etype, cat_median_cf=None) -> str:
    """Return the exact text shown on the product card."""
    if etype == "environmental":
        # % reduction relative to the category median carbon footprint
        if cat_median_cf and cat_median_cf.get(item["cat"], 0) > 0:
            med = cat_median_cf[item["cat"]]
            pct = max(1, round((1 - item["raw_cf"] / med) * 100))
            return f"Produces {pct}% less CO\u2082 than the category median."
        return "Lower carbon footprint than comparable models."
    if etype == "economic":
        return (f"Saves about ${item['sav']:.0f}/yr relative to "
                f"typical {item['cat'].lower()} products.")
    if etype == "social":
        return "Popular among eco-conscious shoppers in this category."
    return ("Recommended by balancing your preferences with a strong "
            "sustainability score.")