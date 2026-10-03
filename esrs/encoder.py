"""Item encoder g: shallow autoencoder."""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from .config import (
    SEED, D_EMB, TF_DIM, AE_HIDDEN, AE_LR, AE_BATCH, AE_EPOCHS,
    AE_PATIENCE, AE_VAL_SPLIT, CAT_SIZES,
)
from .data import _minmax_within        # reuse the exact category-wise normaliser


# ---------------------------------------------------------------------------
# Feature matrix X (135-dim)
# ---------------------------------------------------------------------------
def build_item_features(items, tf_dim=TF_DIM):
    cats = list(CAT_SIZES.keys())
    cat_of = [it["cat"] for it in items]

    # category-normalized price
    prices = np.array([it["price"] for it in items], dtype=float)
    pmin, pmax = prices.min(), prices.max()
    price_n = (prices - pmin) / (pmax - pmin + 1e-9)

    # category-normalized sustainability
    cf = np.array([it["raw_cf"] for it in items])
    rm = np.array([it["raw_rm"] for it in items])
    ee = np.array([it["raw_ee"] for it in items], dtype=float)
    cf_n = 1.0 - _minmax_within(cf, cat_of)
    rm_n = _minmax_within(rm, cat_of)
    ee_n = _minmax_within(ee, cat_of)

    # term-frequency block
    vocab = {}
    for it in items:
        for w in (it["title"] + " " + it["description"]).lower().split():
            vocab.setdefault(w, len(vocab))

    N = len(items)
    in_dim = 3 + 1 + 3 + tf_dim
    X = np.zeros((N, in_dim), dtype=np.float32)
    for i, it in enumerate(items):
        X[i, cats.index(it["cat"])] = 1.0
        X[i, 3] = price_n[i]
        X[i, 4], X[i, 5], X[i, 6] = cf_n[i], rm_n[i], ee_n[i]
        for w in (it["title"] + " " + it["description"]).lower().split():
            X[i, 7 + (vocab[w] % tf_dim)] += 1.0

    # unit-normalise the TF block
    tf = X[:, 7:]
    norms = np.linalg.norm(tf, axis=1, keepdims=True) + 1e-9
    X[:, 7:] = tf / norms
    return X


# ---------------------------------------------------------------------------
# Autoencoder 135 → 256 → 128 → 256 → 135
# ---------------------------------------------------------------------------
class AutoEncoder(nn.Module):
    def __init__(self, in_dim, hidden=AE_HIDDEN, code=D_EMB):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, code),                    # linear code layer
        )
        self.dec = nn.Sequential(
            nn.Linear(code, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim), nn.Sigmoid(),    # symmetric decoder
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


def train_encoder(X, d=D_EMB, device=None, verbose=True):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)

    Xtr, Xva = train_test_split(X, test_size=AE_VAL_SPLIT, random_state=SEED)
    tr = DataLoader(TensorDataset(torch.from_numpy(Xtr)),
                    batch_size=AE_BATCH, shuffle=True)
    va = torch.from_numpy(Xva).to(device)

    model = AutoEncoder(X.shape[1], AE_HIDDEN, d).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=AE_LR)
    loss_fn = nn.MSELoss()

    best_loss = math.inf
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad = 0
    for ep in range(AE_EPOCHS):
        model.train()
        for (xb,) in tr:
            xb = xb.to(device)
            opt.zero_grad()
            xr, _ = model(xb)
            loss = loss_fn(xr, xb)
            loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            xr, _ = model(va)
            vloss = loss_fn(xr, va).item()
        if vloss < best_loss - 1e-5:
            best_loss, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= AE_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, Z = model(torch.from_numpy(X).to(device))
    if verbose:
        print(f"[encoder] val MSE = {best_loss:.5f}  (stopped at epoch {ep+1})")
    return Z.cpu().numpy()