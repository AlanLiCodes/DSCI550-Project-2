"""Per-cell CNN: estimate the added-vessel energy fraction of every (minute, band) cell.

Each training mixture comes with its exact per-cell fraction added / mixture
(``make_mixtures(..., return_cells=True)``), which is much richer supervision than
the single window share. The network predicts a fraction f_ij in [0, 1] for every
cell; the window share is the mixture-energy-weighted mean of f over observed
cells, so the model is trained directly on the leaderboard quantity plus a
per-cell term.

Inputs are level-robust channels computed per window (missing cells -> 0 plus a
mask channel), so absolute loudness is only one of five inputs.

Requires PyTorch (CPU is enough: ~50k parameters, 15 x 25 inputs).
"""

from __future__ import annotations

import numpy as np

from . import config


def window_channels(X: np.ndarray) -> np.ndarray:
    """(n, 15, 25) dB with NaN -> (n, 5, 15, 25) float32 network input.

    0: level relative to the window median          (x - med) / 10
    1: temporal deviation per band                   (x - band median over time) / 10
    2: spectral shape per minute                     (x - minute median over bands) / 10
    3: observed mask
    4: absolute level                                (x - 80) / 20
    """
    X = np.asarray(X, dtype=np.float32)
    obs = np.isfinite(X)
    with np.errstate(invalid="ignore"), _quiet():
        med = np.nanmedian(X.reshape(len(X), -1), axis=1)[:, None, None]
        band_med = np.nanmedian(X, axis=1, keepdims=True)
        min_med = np.nanmedian(X, axis=2, keepdims=True)
    ch = np.stack(
        [(X - med) / 10.0, (X - band_med) / 10.0, (X - min_med) / 10.0, obs.astype(np.float32), (X - 80.0) / 20.0],
        axis=1,
    )
    ch[~np.repeat(obs[:, None], 5, axis=1)] = 0.0
    ch[:, 3] = obs
    return np.nan_to_num(ch, nan=0.0).astype(np.float32)


def energy_weights(X: np.ndarray) -> np.ndarray:
    """Per-cell mixture energy normalised within each window (0 where unobserved)."""
    X = np.asarray(X, dtype=np.float64)
    with _quiet():
        mx = np.nanmax(X.reshape(len(X), -1), axis=1)[:, None, None]
    w = np.power(10.0, (X - mx) / 10.0)
    w[~np.isfinite(w)] = 0.0
    s = w.sum(axis=(1, 2), keepdims=True)
    return (w / np.where(s > 0, s, 1.0)).astype(np.float32)


class _quiet:
    def __enter__(self):
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


def _build_net(width: int = 32):
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, c, dil):
            super().__init__()
            self.conv1 = nn.Conv2d(c, c, 3, padding=dil, dilation=dil)
            self.conv2 = nn.Conv2d(c, c, 3, padding=1)
            self.norm = nn.GroupNorm(4, c)
            self.act = nn.GELU()

        def forward(self, x):
            return x + self.conv2(self.act(self.norm(self.conv1(x))))

    class CellNet(nn.Module):
        def __init__(self, c=width):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(5, c, 3, padding=1), nn.GELU())
            self.b1, self.b2, self.b3 = Block(c, 1), Block(c, 2), Block(c, 4)
            self.ctx = nn.Sequential(nn.Linear(c, c), nn.GELU(), nn.Linear(c, 2 * c))
            self.b4 = Block(c, 1)
            self.head = nn.Conv2d(c, 1, 1)

        def forward(self, x):
            mask = x[:, 3:4]
            h = self.b3(self.b2(self.b1(self.stem(x))))
            pooled = (h * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1.0)
            scale, shift = self.ctx(pooled).chunk(2, dim=1)  # window-level context (FiLM)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
            return torch.sigmoid(self.head(self.b4(h)))[:, 0]

    return CellNet()


class CellCNN:
    """sklearn-style wrapper: fit(X, y, cells) / predict(X)."""

    def __init__(self, epochs: int = 25, batch_size: int = 256, lr: float = 2e-3, weight_decay: float = 1e-4,
                 cell_weight: float = 1.0, width: int = 32, time_flip: bool = True, seed: int = config.SEED,
                 threads: int | None = None, verbose: bool = True):
        self.epochs, self.batch_size, self.lr, self.weight_decay = epochs, batch_size, lr, weight_decay
        self.cell_weight, self.width, self.time_flip, self.seed = cell_weight, width, time_flip, seed
        self.threads, self.verbose = threads, verbose
        self.history_: list = []

    def _tensors(self, X):
        import torch

        return torch.from_numpy(window_channels(X)), torch.from_numpy(energy_weights(X))

    def fit(self, X, y, cells, X_val=None, y_val=None):
        import torch

        torch.manual_seed(self.seed)
        if self.threads:
            torch.set_num_threads(self.threads)
        xin, w = self._tensors(X)
        tgt = torch.from_numpy(np.nan_to_num(np.asarray(cells, dtype=np.float32), nan=0.0))
        yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
        self.net_ = _build_net(self.width)
        opt = torch.optim.AdamW(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        steps = self.epochs * int(np.ceil(len(xin) / self.batch_size))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps, pct_start=0.15)
        g = torch.Generator().manual_seed(self.seed)
        best = (np.inf, None)
        for ep in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(len(xin), generator=g)
            tot = 0.0
            for i in range(0, len(xin), self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb, wb, tb, yb = xin[idx], w[idx], tgt[idx], yt[idx]
                if self.time_flip:
                    flip = torch.rand(len(idx), generator=g) < 0.5
                    xb, wb, tb = xb.clone(), wb.clone(), tb.clone()
                    xb[flip] = xb[flip].flip(2)
                    wb[flip] = wb[flip].flip(1)
                    tb[flip] = tb[flip].flip(1)
                f = self.net_(xb)
                share = (f * wb).sum((1, 2))
                loss = ((share - yb) ** 2).mean() + self.cell_weight * ((f - tb) ** 2 * wb).sum((1, 2)).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += float(loss.detach()) * len(idx)
            rec = {"epoch": ep + 1, "train_loss": tot / len(xin)}
            if X_val is not None:
                p = self.predict(X_val)
                rec["val_rmse"] = float(np.sqrt(np.mean((p - np.asarray(y_val)) ** 2)))
                if rec["val_rmse"] < best[0]:
                    best = (rec["val_rmse"], {k: v.detach().clone() for k, v in self.net_.state_dict().items()})
            self.history_.append(rec)
            if self.verbose:
                print("  cnn " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
        if best[1] is not None:
            self.net_.load_state_dict(best[1])
        return self

    def predict(self, X, return_cells: bool = False, batch: int = 2048):
        import torch

        self.net_.eval()
        xin, w = self._tensors(X)
        shares, cells = [], []
        with torch.no_grad():
            for i in range(0, len(xin), batch):
                f = self.net_(xin[i : i + batch])
                shares.append((f * w[i : i + batch]).sum((1, 2)).numpy())
                if return_cells:
                    cells.append(f.numpy())
        share = np.clip(np.concatenate(shares), 0.0, 1.0)
        return (share, np.concatenate(cells)) if return_cells else share
