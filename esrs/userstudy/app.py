"""Flask web app for the participant study.

Run:
    python -m esrs.userstudy.app
Then open http://127.0.0.1:5000/ .
"""
from __future__ import annotations
import os, random, json, atexit
from flask import (Flask, request, session, redirect, url_for,
                   render_template_string, jsonify)

from .store import Store
from .tasks import (TASKS, CONDITIONS, CONDITION_LABEL, grid_order,
                    product_card_html)
from .questionnaire import likert_items_for_condition, DEMOGRAPHICS
from ..config import TAU

DB_PATH = os.environ.get("ESRS_DB", "esrs_userstudy.sqlite")
STORE = Store(DB_PATH)
SLICES = None      # populated by load_slices()
N_PER_COND = 50    # target sample size per condition (N=150)

app = Flask(__name__)
app.secret_key = os.environ.get("ESRS_SECRET", "esrs-dev-secret")


# ---------------------------------------------------------------------------
# One-time slice build (re-uses the cached synthetic catalog + encoder)
# ---------------------------------------------------------------------------
def load_slices():
    global SLICES
    if SLICES is not None:
        return SLICES
    from ..data import load_catalog
    from ..encoder import build_item_features, train_encoder
    from ..interactions import simulate_interactions
    from collections import defaultdict
    import numpy as np

    print("[app] building catalog and embeddings ...")
    items = load_catalog(use_amazon=False)
    for i, it in enumerate(items):
        it["item_id"] = f"i{i:05d}"
    interactions, _ = simulate_interactions(items, n_users=200)
    X = build_item_features(items)
    E = train_encoder(X, verbose=False)

    # minimal pop_s (the full pipeline already assigns it in the offline run)
    from ..interactions import sustainable_popularity
    items = sustainable_popularity(items, interactions)
    SLICES = _build_slices(items, E)
    return SLICES


def _build_slices(items, embeddings):
    from .tasks import build_task_slices
    return build_task_slices(items, embeddings)


# ---------------------------------------------------------------------------
# Templates (inline, no external assets)
# ---------------------------------------------------------------------------
BASE_CSS = """
body{font-family:system-ui,Segoe UI,Roboto,sans-serif;margin:0;background:#f7f8fa;color:#222}
header{background:#1f2a44;color:#fff;padding:14px 24px}
main{max-width:1100px;margin:24px auto;padding:0 16px}
.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:14px}
.card{background:#fff;border:1px solid #e5e6ea;border-radius:10px;padding:12px;
      display:flex;flex-direction:column;gap:6px}
.card h4{margin:0;font-size:14px;line-height:1.25}
.price{color:#0d5bd7;font-weight:600}
.expl{font-size:12px;color:#4a5568;background:#f0f7ff;border-left:3px solid #4a90d9;
      padding:6px 8px;border-radius:4px;margin:6px 0 0}
.thumb{display:flex;justify-content:center}
.click-btn{margin-top:auto;padding:8px;border:0;border-radius:6px;
           background:#1f2a44;color:#fff;cursor:pointer}
.click-btn.clicked{background:#2e9c52}
.nav{margin-top:22px;display:flex;justify-content:space-between;align-items:center}
.btn{background:#1f2a44;color:#fff;border:0;border-radius:6px;
     padding:10px 18px;cursor:pointer;text-decoration:none}
.q{background:#fff;border:1px solid #e5e6ea;border-radius:10px;
   padding:14px 18px;margin-bottom:12px}
.q label{margin-right:16px}
.small{color:#666;font-size:13px}
"""

WELCOME = """
<!doctype html><html><head><title>ESRS Study</title><style>{{css}}</style></head>
<body><header><h1>ESRS User Study</h1></header><main>
<h2>Welcome</h2>
<p>This is a short research study about product recommendations. It takes
about 8–10 minutes and is completely anonymous.</p>
<p>You will complete <b>three shopping tasks</b> using a simulated
e-commerce interface, then a short questionnaire.</p>
<p class="small">Participation is voluntary and unpaid. You may stop at any
time. No personal identifiers are stored.</p>
<form method="post" action="{{url}}">
  <button class="btn" type="submit">I understand and consent — begin</button>
</form>
</main></body></html>
"""

TASK_PAGE = """
<!doctype html><html><head><title>ESRS · Task {{idx}}/3</title>
<style>{{css}}</style>
<script>
function recordClick(btn, itemId){
  fetch('{{click_url}}', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({item_id:itemId})});
  btn.classList.add('clicked'); btn.textContent='Selected';
}
</script></head>
<body><header><h1>{{label}}</h1></header><main>
<p class="small">Click the product you would choose. When you're done,
press <b>Continue</b>.</p>
<div class="grid">{{cards|safe}}</div>
<div class="nav">
  <span class="small">Task {{idx}} of 3</span>
  <a class="btn" href="{{next_url}}">Continue</a>
</div>
</main></body></html>
"""

QUESTIONNAIRE = """
<!doctype html><html><head><title>ESRS · Questionnaire</title>
<style>{{css}}</style></head>
<body><header><h1>Post-study questionnaire</h1></header><main>
<form method="post" action="{{post_url}}">
<h3>Please rate the following statements (1 = strongly disagree,
5 = strongly agree).</h3>
{% for key, metric, text in likert %}
  <div class="q">
    <div>{{text}}</div>
    {% for v in [1,2,3,4,5] %}
      <label><input type="radio" name="{{key}}" value="{{v}}"
        {% if v==3 %}checked{% endif %} required> {{v}}</label>
    {% endfor %}
  </div>
{% endfor %}

<h3>Demographics</h3>
{% for key, kind, text, choices in demo %}
  <div class="q">
    <div>{{text}}</div>
    {% if kind == 'integer' %}
      <input type="number" name="{{key}}" min="16" max="99" value="30" required>
    {% elif kind == 'text' %}
      <input type="text" name="{{key}}" value="" maxlength="60">
    {% elif kind == 'likert' %}
      {% for v in [1,2,3,4,5] %}
        <label><input type="radio" name="{{key}}" value="{{v}}"
          {% if v==3 %}checked{% endif %}> {{v}}</label>
      {% endfor %}
    {% elif kind == 'choice' %}
      <select name="{{key}}" required>
        {% for c in choices %}<option>{{c}}</option>{% endfor %}
      </select>
    {% endif %}
  </div>
{% endfor %}

<div class="nav"><button class="btn" type="submit">Submit</button></div>
</form>
</main></body></html>
"""

THANKS = """
<!doctype html><html><head><title>ESRS · Thanks</title>
<style>{{css}}</style></head>
<body><header><h1>Thank you</h1></header><main>
<p>Your responses have been recorded anonymously.</p>
<p class="small">You may now close this window.</p>
</main></body></html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
def _next_condition():
    """Assign the condition with the smallest current count (balanced)."""
    parts = STORE.as_dataframes()["participants"]
    if parts.empty:
        return random.choice(CONDITIONS)
    counts = parts["condition"].value_counts().to_dict()
    for c in CONDITIONS:
        counts.setdefault(c, 0)
    # simplest balanced rule: pick the least-populated condition
    least = min(CONDITIONS, key=lambda c: counts[c])
    return least


@app.route("/")
def index():
    return render_template_string(WELCOME, css=BASE_CSS,
                                  url=url_for("begin"))


@app.route("/begin", methods=["POST"])
def begin():
    condition = _next_condition()
    pid = STORE.create_participant(condition)
    session["pid"] = pid
    session["condition"] = condition
    session["task_order"] = random.sample([t["id"] for t in TASKS], k=3)
    session["task_idx"] = 0
    return redirect(url_for("task", idx=0))


def _task_context(idx):
    pid = session.get("pid")
    condition = session.get("condition")
    task_order = session.get("task_order") or []
    tid = task_order[idx]
    task = next(t for t in TASKS if t["id"] == tid)
    sl = load_slices()[tid][condition]

    # random grid order, deterministic per (pid, tid)
    order = grid_order(pid, tid, len(sl))
    cards = []
    for pos, j in enumerate(order):
        it = dict(sl[j])
        it["position"] = pos
        it["item_id"] = it.get("item_id", f"{tid}_{j}")
        STORE.record_impression(pid, tid, it)
        cards.append(product_card_html(it, pos, condition))
    return task, tid, cards


@app.route("/task/<int:idx>")
def task(idx):
    if "pid" not in session:
        return redirect(url_for("index"))
    task_obj, tid, cards = _task_context(idx)
    nxt = url_for("task", idx=idx + 1) if idx < 2 else url_for("questionnaire")
    return render_template_string(
        TASK_PAGE, css=BASE_CSS,
        idx=idx + 1, label=task_obj["label"],
        cards="".join(cards),
        click_url=url_for("click", tid=tid),
        next_url=nxt)


@app.route("/click/<tid>", methods=["POST"])
def click(tid):
    pid = session.get("pid")
    if not pid:
        return jsonify(ok=False), 401
    item_id = (request.get_json(force=True) or {}).get("item_id")
    if item_id:
        STORE.record_click(pid, tid, item_id)
    return jsonify(ok=True)


@app.route("/questionnaire", methods=["GET", "POST"])
def questionnaire():
    pid = session.get("pid")
    if not pid:
        return redirect(url_for("index"))
    if request.method == "POST":
        form = request.form
        cond = session.get("condition")
        for key, metric, _ in likert_items_for_condition(cond):
            val = form.get(key)
            if val is not None:
                STORE.record_rating(pid, metric, int(val))
        demographics = {k: (form.get(k) or "").strip()
                        for k, *_ in DEMOGRAPHICS}
        STORE.finalize_participant(pid, demographics)
        return redirect(url_for("thanks"))

    cond = session["condition"]
    likert = likert_items_for_condition(cond)
    demo = [(k, kind, text, choices) if kind == "choice" else (k, kind, text, [])
            for (k, kind, text, *rest) in DEMOGRAPHICS
            for choices in [rest[0] if rest else []]]
    return render_template_string(
        QUESTIONNAIRE, css=BASE_CSS, likert=likert, demo=demo,
        post_url=url_for("questionnaire"))


@app.route("/thanks")
def thanks():
    return render_template_string(THANKS, css=BASE_CSS)


# ---------------------------------------------------------------------------
# Dev convenience: reset & shutdown
# ---------------------------------------------------------------------------
@app.route("/_admin/reset")
def admin_reset():
    STORE.conn.executescript(
        "DELETE FROM participants; DELETE FROM impressions; "
        "DELETE FROM clicks; DELETE FROM ratings;")
    STORE.conn.commit()
    return "reset ok"


def _shutdown():
    try: STORE.conn.close()
    except Exception: pass
atexit.register(_shutdown)


if __name__ == "__main__":
    load_slices()
    app.run(host="127.0.0.1", port=5000, debug=False)