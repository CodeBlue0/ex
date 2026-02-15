"""
End-to-end accuracy optimizer (FT-Transformer)
- Target: y_uw
- Data: /workspace/data/uw_train_preprocessed.csv
- Objective: maximize validation accuracy
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EPS = 1e-15


def balanced_log_loss(y_true: np.ndarray, p1: np.ndarray) -> float:
    p1 = np.clip(p1.astype(float), EPS, 1.0 - EPS)
    p0 = 1.0 - p1
    y_true = y_true.astype(int)
    n0 = max(np.sum(y_true == 0), 1)
    n1 = max(np.sum(y_true == 1), 1)
    loss0 = -np.sum(np.log(p0[y_true == 0])) / n0
    loss1 = -np.sum(np.log(p1[y_true == 1])) / n1
    return 0.5 * (loss0 + loss1)


def binary_ce(y_true: np.ndarray, p1: np.ndarray) -> float:
    p1 = np.clip(p1.astype(float), EPS, 1.0 - EPS)
    y = y_true.astype(float)
    return float(-np.mean(y * np.log(p1) + (1.0 - y) * np.log(1.0 - p1)))


def best_threshold_for_acc(y_true: np.ndarray, p1: np.ndarray, n_steps: int = 1001) -> tuple[float, float]:
    ts = np.linspace(0.0, 1.0, n_steps)
    best_t, best_acc = 0.5, -1.0
    for t in ts:
        acc = accuracy_score(y_true, (p1 >= t).astype(int))
        if acc > best_acc:
            best_acc = float(acc)
            best_t = float(t)
    return best_t, best_acc


def encode_categoricals(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = train_df.copy()
    va = valid_df.copy()
    cat_cols = tr.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for c in cat_cols:
        tr_c = tr[c].astype("string").fillna("__NA__")
        va_c = va[c].astype("string").fillna("__NA__")
        uniques = pd.Index(tr_c.unique())
        mapping = {k: i for i, k in enumerate(uniques)}
        tr[c] = tr_c.map(mapping).astype(np.int32)
        va[c] = va_c.map(mapping).fillna(-1).astype(np.int32)
    return tr, va


def rebalance_to_5_5(y: np.ndarray, seed: int) -> np.ndarray:
    cls, cnt = np.unique(y, return_counts=True)
    if len(cls) != 2:
        return np.arange(len(y))

    major = int(cls[np.argmax(cnt)])
    minor = int(cls[np.argmin(cnt)])
    n_minor = int(cnt[np.argmin(cnt)])

    rng = np.random.default_rng(seed)
    major_idx = np.where(y == major)[0]
    minor_idx = np.where(y == minor)[0]

    major_keep = rng.choice(major_idx, size=n_minor, replace=False)
    keep = np.concatenate([major_keep, minor_idx])
    rng.shuffle(keep)
    return keep


class FTTransformer(nn.Module):
    def __init__(self, n_features: int, d_token: int, n_heads: int, n_layers: int, ff_mult: int, dropout: float):
        super().__init__()
        self.feature_weight = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.feature_bias = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_token,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x_tok = torch.cat([cls, tokens], dim=1)
        x_enc = self.encoder(x_tok)
        cls_out = self.norm(x_enc[:, 0, :])
        return self.head(cls_out).squeeze(1)


@torch.no_grad()
def predict_proba(model: nn.Module, x: np.ndarray, device: torch.device, infer_batch_size: int) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(x), infer_batch_size):
        xb = torch.tensor(x[s : s + infer_batch_size], dtype=torch.float32, device=device)
        pb = torch.sigmoid(model(xb)).detach().cpu().numpy()
        out.append(pb)
    p1 = np.concatenate(out, axis=0)
    return np.clip(p1, EPS, 1.0 - EPS)


def train_one_trial(
    cfg: dict,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    device: torch.device,
    infer_batch_size: int,
    seed: int,
    class_weights: tuple[float, float] | None,
) -> dict:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = FTTransformer(
        n_features=x_tr.shape[1],
        d_token=cfg["d_token"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        ff_mult=cfg["ff_mult"],
        dropout=cfg["dropout"],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(reduction="none")

    xt = torch.tensor(x_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    n = len(xt)
    total_batches = (n + cfg["batch_size"] - 1) // cfg["batch_size"]

    best = {
        "acc": -1.0,
        "thr": 0.5,
        "state": copy.deepcopy(model.state_dict()),
        "epoch": 0,
        "ce": None,
        "auc": None,
        "bll": None,
    }

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(n)
        losses = []
        for bi, s in enumerate(range(0, n, cfg["batch_size"]), start=1):
            idx = perm[s : s + cfg["batch_size"]]
            xb = xt[idx].to(device)
            yb = yt[idx].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            losses_raw = crit(logits, yb)
            if class_weights is None:
                loss = losses_raw.mean()
            else:
                w0, w1 = class_weights
                sample_w = torch.where(yb > 0.5, torch.full_like(yb, w1), torch.full_like(yb, w0))
                loss = (losses_raw * sample_w).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            if bi % 20 == 0 or bi == total_batches:
                print(f"[trial {cfg['name']}] epoch {epoch}/{cfg['epochs']} batch {bi}/{total_batches} loss={losses[-1]:.6f}")

        p_va = predict_proba(model, x_va, device, infer_batch_size)
        thr, acc = best_threshold_for_acc(y_va, p_va)
        ce = binary_ce(y_va, p_va)
        auc = roc_auc_score(y_va, p_va)
        bll = balanced_log_loss(y_va, p_va)
        print(
            f"[trial {cfg['name']}] epoch {epoch}/{cfg['epochs']} "
            f"train_loss={np.mean(losses):.6f} val_ce={ce:.6f} val_acc={acc:.6f} thr={thr:.3f}"
        )

        if acc > best["acc"]:
            best.update(
                {
                    "acc": float(acc),
                    "thr": float(thr),
                    "state": copy.deepcopy(model.state_dict()),
                    "epoch": int(epoch),
                    "ce": float(ce),
                    "auc": float(auc),
                    "bll": float(bll),
                }
            )

    model.load_state_dict(best["state"])
    p_va = predict_proba(model, x_va, device, infer_batch_size)
    acc_05 = float(accuracy_score(y_va, (p_va >= 0.5).astype(int)))

    return {
        "name": cfg["name"],
        "cfg": cfg,
        "best_acc": best["acc"],
        "best_thr": best["thr"],
        "best_epoch": best["epoch"],
        "acc_05": acc_05,
        "val_auc": best["auc"],
        "val_bll": best["bll"],
        "val_ce": best["ce"],
    }


def main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    df = pd.read_csv(args.data_path)
    if args.target_col not in df.columns:
        raise ValueError(f"target column '{args.target_col}' not found")

    if args.ins_gbn_only_one:
        if "INS_GBN_CD" not in df.columns:
            raise ValueError("INS_GBN_CD column not found")
        df = df[df["INS_GBN_CD"] == 1].reset_index(drop=True)

    y = df[args.target_col].astype(int).values
    X = df.drop(columns=[args.target_col]).copy()

    tr_idx, va_idx = train_test_split(
        np.arange(len(df)),
        test_size=args.valid_size,
        random_state=args.seed,
        stratify=y,
    )

    X_tr = X.iloc[tr_idx].reset_index(drop=True)
    X_va = X.iloc[va_idx].reset_index(drop=True)
    y_tr = y[tr_idx]
    y_va = y[va_idx]

    if args.balance_mode == "downsample_5_5":
        keep_idx_tr = rebalance_to_5_5(y_tr, seed=args.seed)
        X_tr = X_tr.iloc[keep_idx_tr].reset_index(drop=True)
        y_tr = y_tr[keep_idx_tr]

    X_tr, X_va = encode_categoricals(X_tr, X_va)

    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_tr)
    X_va = imp.transform(X_va)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)

    device = torch.device("cuda" if (not args.cpu_only and torch.cuda.is_available()) else "cpu")

    print(f"device={device.type}")
    print(f"n_samples={len(df)}, n_features={X_tr.shape[1]}, target={args.target_col}")
    uniq, cnt = np.unique(y_tr, return_counts=True)
    print(f"train_class_counts={dict(zip(uniq.tolist(), cnt.tolist()))}")
    uniq_va, cnt_va = np.unique(y_va, return_counts=True)
    print(f"val_class_counts={dict(zip(uniq_va.tolist(), cnt_va.tolist()))}")
    print(f"balance_mode={args.balance_mode}")

    class_weights = None
    if args.balance_mode == "full_weighted":
        n0 = max(int((y_tr == 0).sum()), 1)
        n1 = max(int((y_tr == 1).sum()), 1)
        n = n0 + n1
        w0 = n / (2.0 * n0)
        w1 = n / (2.0 * n1)
        class_weights = (float(w0), float(w1))
        print(f"class_weights={{0:{w0:.6f}, 1:{w1:.6f}}}")

    grid = [
        {
            "name": "A",
            "epochs": args.epochs,
            "lr": 9e-4,
            "batch_size": args.batch_size,
            "d_token": 128,
            "n_heads": 8,
            "n_layers": 4,
            "ff_mult": 4,
            "dropout": 0.10,
        },
        {
            "name": "B",
            "epochs": args.epochs,
            "lr": 7e-4,
            "batch_size": args.batch_size,
            "d_token": 160,
            "n_heads": 8,
            "n_layers": 6,
            "ff_mult": 4,
            "dropout": 0.12,
        },
        {
            "name": "C",
            "epochs": args.epochs,
            "lr": 1.0e-3,
            "batch_size": args.batch_size,
            "d_token": 192,
            "n_heads": 12,
            "n_layers": 6,
            "ff_mult": 4,
            "dropout": 0.10,
        },
        {
            "name": "D",
            "epochs": args.epochs,
            "lr": 6e-4,
            "batch_size": args.batch_size,
            "d_token": 256,
            "n_heads": 8,
            "n_layers": 8,
            "ff_mult": 4,
            "dropout": 0.15,
        },
    ]

    if args.max_trials is not None:
        grid = grid[: max(1, args.max_trials)]

    best = None
    for cfg in grid:
        result = train_one_trial(
            cfg,
            X_tr,
            y_tr,
            X_va,
            y_va,
            device,
            args.infer_batch_size,
            args.seed,
            class_weights,
        )
        print(
            f"[trial {result['name']} done] acc@0.5={result['acc_05']:.10f} "
            f"best_acc={result['best_acc']:.10f} best_thr={result['best_thr']:.4f} "
            f"best_epoch={result['best_epoch']} auc={result['val_auc']:.10f}"
        )
        if best is None or result["best_acc"] > best["best_acc"]:
            best = result

    majority_acc = float(max((y_va == 0).mean(), (y_va == 1).mean()))
    print("[best]")
    print(f"best_trial={best['name']}")
    print(f"best_config={best['cfg']}")
    print(f"val_acc@0.5={best['acc_05']:.10f}")
    print(f"val_acc_best_thr={best['best_acc']:.10f}")
    print(f"best_threshold={best['best_thr']:.6f}")
    print(f"val_auc={best['val_auc']:.10f}")
    print(f"validation_balanced_log_loss={best['val_bll']:.10f}")
    print(f"val_ce={best['val_ce']:.10f}")
    print(f"majority_val_acc={majority_acc:.10f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/workspace/data/uw_train_preprocessed.csv")
    parser.add_argument("--target-col", type=str, default="y_uw")
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--ins-gbn-only-one", action="store_true")
    parser.add_argument("--balance-mode", type=str, default="full_weighted", choices=["full_weighted", "downsample_5_5"])

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--infer-batch-size", type=int, default=256)
    parser.add_argument("--max-trials", type=int, default=1)

    args = parser.parse_args()
    main(args)
