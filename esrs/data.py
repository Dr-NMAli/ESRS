"""Catalog construction: Amazon Reviews 2023 + synthetic fallback."""
from __future__ import annotations
import gzip, json, urllib.request
import numpy as np
from collections import defaultdict

from .config import (
    SEED, CAT_SIZES, ECO_WEIGHTS, TAU,
)

# ---------------------------------------------------------------------------
# 1. Amazon Reviews 2023 (McAuley Lab) 
# ---------------------------------------------------------------------------
# Official landing page:  https://amazon-reviews-2023.github.io/
# Mirror (HuggingFace):   https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
# Raw file pattern:       raw/meta_categories/meta_<Category>.jsonl
HF_BASE = ("https://huggingface.co/datasets/McAuley-Lab/"
           "Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_{}.jsonl")
CATEGORY_FILE = {
    "Electronics": "Electronics",
    "Home":        "Home_and_Kitchen",
    "Clothing":    "Clothing_Shoes_and_Jewelry",
}
# Only stream this many lines per category; files are multi-GB.
MAX_LINES = {"Electronics": 30_000, "Home": 25_000, "Clothing": 20_000}


def _stream_jsonl(url: str, max_lines: int):
    """Stream JSON objects from a (optionally gzip'd) text file."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            first = resp.read(2)
            resp = _PrefixedStream(first, resp)
            stream = gzip.GzipFile(fileobj=resp) if first[:2] == b"\x1f\x8b" else resp
            for i, raw in enumerate(stream):
                if i >= max_lines:
                    break
                try:
                    line = raw.decode("utf-8").strip() if isinstance(raw, bytes) else raw
                    if line:
                        yield json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
    except Exception as e:
        print(f"[data] stream failed ({url}): {e}")


class _PrefixedStream:
    """Re-attach the 2 bytes we peeked to check gzip magic."""
    def __init__(self, prefix, stream):
        self._prefix = prefix
        self._stream = stream
        self._consumed = False

    def read(self, *a, **k):
        if not self._consumed:
            self._consumed = True
            head = self._prefix
            rest = self._stream.read(*a, **k)
            return head + rest
        return self._stream.read(*a, **k)


def load_amazon_metadata(max_lines: dict | None = None):
    """Return {category: [item dicts]} or None if the mirror is unreachable."""
    max_lines = max_lines or MAX_LINES
    out = {}
    for cat, fname in CATEGORY_FILE.items():
        url = HF_BASE.format(fname)
        print(f"[data] streaming {cat} metadata ...")
        items = []
        for rec in _stream_jsonl(url, max_lines[cat]):
            title = str(rec.get("title") or "").strip()
            if not title:
                continue
            desc = rec.get("description") or []
            if isinstance(desc, list):
                desc = " ".join(map(str, desc))
            price = rec.get("price")
            try:
                price = float(str(price).replace("$", "").replace(",", "")) if price else np.nan
            except ValueError:
                price = np.nan
            items.append({"cat": cat, "title": title,
                          "description": str(desc), "price": price})
            if len(items) >= CAT_SIZES[cat]:
                break
        if len(items) < CAT_SIZES[cat]:
            print(f"[data] {cat}: only {len(items)} usable rows — falling back to synthetic for this category")
            return None
        out[cat] = items
    return out


# ---------------------------------------------------------------------------
# 2. Synthetic fallback — exactly mirrors Sec. 4.1.1
# ---------------------------------------------------------------------------
_TOKENS = ("eco green smart ultra pro max rechargeable sustainable organic "
           "durable efficient compact energy saving recycled carbon neutral "
           "premium wireless portable ergonomic lightweight").split()


def _fake_title(cat):
    return f"{cat} {np.random.choice(_TOKENS)} {np.random.choice(_TOKENS)}"


def _fake_desc():
    return " ".join(np.random.choice(_TOKENS, size=25))


def build_synthetic_catalog():
    items = []
    for cat, n in CAT_SIZES.items():
        for _ in range(n):
            items.append({
                "cat": cat,
                "title": _fake_title(cat),
                "description": _fake_desc(),
                "price": float(np.round(np.random.lognormal(3, 0.6), 2)),
            })
    return items


def _fill_missing(items):
    for it in items:
        if not it.get("title"):
            it["title"] = _fake_title(it.get("cat", "Item"))
        if not it.get("description"):
            it["description"] = _fake_desc()
        if it.get("price") is None or np.isnan(it.get("price", np.nan)):
            it["price"] = float(np.round(np.random.lognormal(3, 0.6), 2))
    return items


# ---------------------------------------------------------------------------
# 3. Sustainability attributes (Sec. 3.3)
# ---------------------------------------------------------------------------
_CF_DIST = {"Electronics": (8.0, 1.2), "Home": (5.0, 1.0), "Clothing": (3.0, 0.8)}
_RM_A    = {"Electronics": 2.0, "Home": 2.5, "Clothing": 3.0}
_RM_B    = {"Electronics": 5.0, "Home": 4.0, "Clothing": 3.5}
_EE_P    = {"Electronics": [.05,.15,.35,.30,.15],
            "Home":        [.10,.25,.35,.20,.10],
            "Clothing":    [.30,.35,.20,.10,.05]}
_RUN     = {"Electronics": (45, 15), "Home": (30, 10), "Clothing": (12, 5)}


def generate_sustainability(items):
    for it in items:
        c = it["cat"]
        mu, sig = _CF_DIST[c]
        it["raw_cf"] = float(np.random.lognormal(np.log(mu), sig))
        it["raw_rm"] = float(np.random.beta(_RM_A[c], _RM_B[c]) * 100)
        it["raw_ee"] = int(np.random.choice(5, p=_EE_P[c]) + 1)
    return items


def _minmax_within(vals, groups):
    out = np.zeros_like(vals, dtype=float)
    groups = np.asarray(groups)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        v = vals[idx]
        lo, hi = v.min(), v.max()
        out[idx] = 0.0 if hi == lo else (v - lo) / (hi - lo)
    return out


def compute_eco_score(items):
    cats = [it["cat"] for it in items]
    cf = np.array([it["raw_cf"] for it in items])
    rm = np.array([it["raw_rm"] for it in items])
    ee = np.array([it["raw_ee"] for it in items], dtype=float)
    cf_n = 1.0 - _minmax_within(cf, cats)              # lower footprint = greener
    rm_n = _minmax_within(rm, cats)
    ee_n = _minmax_within(ee, cats)
    eco = ECO_WEIGHTS[0]*cf_n + ECO_WEIGHTS[1]*rm_n + ECO_WEIGHTS[2]*ee_n
    for i, it in enumerate(items):
        it["eco"] = float(eco[i])
    return items


def generate_savings(items):
    by_cat = defaultdict(list)
    for it in items:
        mu, s = _RUN[it["cat"]]
        it["run_cost"] = max(0.0, float(np.random.normal(mu, s)))
        by_cat[it["cat"]].append(it["run_cost"])
    med = {c: float(np.median(v)) for c, v in by_cat.items()}
    for it in items:
        it["sav"] = max(0.0, med[it["cat"]] - it["run_cost"])
    return items


# ---------------------------------------------------------------------------
# 4. Public entry point
# ---------------------------------------------------------------------------
def load_catalog(use_amazon: bool = False):
    items = None
    if use_amazon:
        items = load_amazon_metadata()
        if items is not None:
            flat = []
            for cat, lst in items.items():
                for r in lst:
                    r["cat"] = cat
                    flat.append(r)
            items = _fill_missing(flat)
            print(f"[data] using REAL Amazon metadata ({len(items)} items)")
    if items is None:
        items = build_synthetic_catalog()
        print(f"[data] using SYNTHETIC catalog ({len(items)} items)")

    items = generate_sustainability(items)
    items = compute_eco_score(items)
    items = generate_savings(items)
    return items