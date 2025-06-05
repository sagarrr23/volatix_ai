# ai/tft_trainer.py — v3.0 (2025-06-05)
"""
Volatix-AI TFT trainer (resume-safe, NaN-guarded, weak-label optional)

Key points
──────────
• WeightedRandomSampler + FocalLoss for class imbalance
• Warm-up + cosine LR, AMP, grad-clip, pin-memory loaders
• NaN-/Inf-aware checkpoint resume
• Optional collapse of Weak-Sell/Weak-Buy ⇒ Strong-Sell/Strong-Buy
• Checkpoint filenames auto-suffix by label-mode so progress is never lost
"""

from __future__ import annotations
import argparse, math, os, random, sys
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import (DataLoader, TensorDataset, WeightedRandomSampler,
                              random_split)
from torch.utils.tensorboard import SummaryWriter

# ── local imports ───────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
from data import prepare_tft_data                       # type: ignore
from ai.tft_model import TemporalFusionTransformer      # type: ignore

# ── run-time switches ───────────────────────────────────────────────────
COLLAPSE_WEAK   = True       # True → map 1→0 (Weak-Sell) and 3→4 (Weak-Buy)
GAMMA_FOCAL     = 1.5        # raise to 2.0 if majority over-predicted
ACC_WEIGHT_CONF = 0.40       # after 10 epochs you may bump to 0.40/0.20
ACC_WEIGHT_REW  = 0.20

# ── constants ───────────────────────────────────────────────────────────
MODEL_BASENAME  = "tft_brain_v2"
CKPT_BASENAME   = "tft_checkpoint"
SUFFIX          = "_3cls" if COLLAPSE_WEAK else ""
MODEL_PATH      = f"C:/NEWW/volatix_ai/models/{MODEL_BASENAME}{SUFFIX}.pt"
CHECKPOINT_PATH = f"C:/NEWW/volatix_ai/models/{CKPT_BASENAME}{SUFFIX}.pt"
DATA_PATH       = "historical_data/processed_tft_data.npz"
LOG_DIR         = f"runs/tft_{datetime.now():%Y%m%d_%H%M%S}{SUFFIX}"

BATCH_SIZE   = 128
EPOCHS       = 50
PATIENCE     = 8
INIT_LR      = 3e-4
WARMUP_EP    = 5
CLIP_NORM    = 0.5
LABEL_SMOOTH = 0.05
USE_MULTI_LOSS = True

SEED     = 42
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if COLLAPSE_WEAK:
    LABELS       = [0, 2, 4]
    LABEL_NAMES  = ["Sell", "Wait", "Buy"]
else:
    LABELS       = [0, 1, 2, 3, 4]
    LABEL_NAMES  = ["Strong Sell", "Weak Sell", "Wait", "Weak Buy", "Strong Buy"]

# ── helpers ─────────────────────────────────────────────────────────────
def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sanitise(t: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp(torch.nan_to_num(t, nan=0.0, posinf=hi, neginf=lo), lo, hi)

class FocalLoss(nn.Module):
    """Focal CE with class-alpha & label smoothing."""
    def __init__(self, alpha: torch.Tensor, gamma: float, smooth: float = 0.0) -> None:
        super().__init__()
        self.alpha, self.gamma, self.smooth = alpha, gamma, smooth
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.smooth:
            n = logits.size(1)
            soft = torch.full_like(logits, self.smooth / n)
            soft.scatter_(1, target.unsqueeze(1), 1 - self.smooth)
            ce = -(soft * torch.log_softmax(logits, dim=1)).sum(1)
        else:
            ce = self.ce(logits, target)
        pt = torch.exp(-ce)
        return (self.alpha[target] * (1 - pt).pow(self.gamma) * ce).mean()

def has_bad_weights(model: nn.Module) -> bool:
    return any(torch.isnan(p).any() or torch.isinf(p).any() for p in model.parameters())

# ── data loader ─────────────────────────────────────────────────────────
def load_or_prepare():
    if not os.path.exists(DATA_PATH):
        print("⚙️  Processed data missing — running data prep…")
        prepare_tft_data.main()

    data = np.load(DATA_PATH, allow_pickle=False)
    X      = sanitise(torch.tensor(data["X"], dtype=torch.float32), -10, 10)
    y_dir  = torch.tensor(data["y_direction"], dtype=torch.long)
    y_conf = sanitise(torch.tensor(data["y_confidence"], dtype=torch.float32).unsqueeze(1), 0, 1)
    y_rew  = sanitise(torch.tensor(data["y_reward"], dtype=torch.float32).unsqueeze(1), -3, 3)

    if COLLAPSE_WEAK:
        y_dir = y_dir.clone()
        y_dir[y_dir == 1] = 0
        y_dir[y_dir == 3] = 4

    print("📊 Label distribution:", dict(Counter(y_dir.numpy())))

    ds  = TensorDataset(X, y_dir, y_conf, y_rew)
    n   = len(ds)
    tr, va = int(.7*n), int(.2*n)
    te     = n - tr - va
    return random_split(ds, [tr, va, te], generator=torch.Generator().manual_seed(SEED))

# ── training routine ────────────────────────────────────────────────────
def train(resume: bool) -> None:
    seed_everything()
    tr_ds, va_ds, te_ds = load_or_prepare()

    y_tr    = torch.tensor([y for _, y, *_ in tr_ds])
    cls_cnt = torch.bincount(y_tr, minlength=len(LABELS)).float()
    cls_w   = cls_cnt.sum() / (cls_cnt + 1e-6)
    sampler = WeightedRandomSampler(cls_w[y_tr].cpu(), len(y_tr), replacement=True)
    cls_w   = cls_w.to(DEVICE)

    dl_tr = DataLoader(tr_ds, BATCH_SIZE, sampler=sampler, drop_last=True, pin_memory=True)
    dl_va = DataLoader(va_ds, BATCH_SIZE, pin_memory=True)
    dl_te = DataLoader(te_ds, BATCH_SIZE, pin_memory=True)

    seq_len = dl_tr.dataset[0][0].shape[0]
    in_dim  = dl_tr.dataset[0][0].shape[1]
    model   = TemporalFusionTransformer(in_dim, 256, seq_len, 4, 4, .3).to(DEVICE)

    focal = FocalLoss(cls_w, GAMMA_FOCAL, LABEL_SMOOTH)
    mse   = nn.MSELoss()
    opt   = torch.optim.AdamW(model.parameters(), lr=INIT_LR, weight_decay=1e-4)

    # warm-up + cosine LR
    def lr_lambda(ep): return (ep+1)/WARMUP_EP if ep < WARMUP_EP \
        else .5*(1+math.cos(math.pi*(ep-WARMUP_EP)/(EPOCHS-WARMUP_EP)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    writer = SummaryWriter(LOG_DIR)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_f1, no_imp, start_ep = 0.0, 0, 1

    # ── resume ────────────────────────────────────────────────────────
    if resume and os.path.exists(CHECKPOINT_PATH):
        ck = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ck["model_state"])
        if has_bad_weights(model):
            print("⚠️  Checkpoint has NaNs – starting fresh.")
            model.apply(lambda m: getattr(m, "reset_parameters", lambda: None)())
        else:
            opt.load_state_dict(ck["optimizer_state"])
            if st := ck.get("scheduler_state"):
                try: sched.load_state_dict(st)
                except (KeyError, ValueError): pass
            if torch.cuda.is_available() and "scaler_state" in ck:
                scaler.load_state_dict(ck["scaler_state"])
            start_ep, best_f1, no_imp = ck["epoch"]+1, ck["best_f1"], ck["epochs_no_improve"]
            print(f"⏯️  Resuming from epoch {start_ep} | best F1={best_f1:.2%}")

    # ── epoch loop ───────────────────────────────────────────────────
    for ep in range(start_ep, EPOCHS+1):
        model.train(); ep_loss = 0.0
        for xb, yb_dir, yb_conf, yb_rew in dl_tr:
            xb, yb_dir = xb.to(DEVICE), yb_dir.to(DEVICE)
            yb_conf, yb_rew = yb_conf.to(DEVICE), yb_rew.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                o_dir, o_conf, o_rew = model(xb)
                loss = focal(o_dir, yb_dir)
                if USE_MULTI_LOSS:
                    loss += ACC_WEIGHT_CONF * mse(o_conf, yb_conf)
                    loss += ACC_WEIGHT_REW  * mse(o_rew,  yb_rew)
            if torch.isnan(loss): continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(opt); scaler.update()
            ep_loss += loss.item()

        sched.step()
        writer.add_scalar("Loss/Train", ep_loss, ep)
        writer.add_scalar("LR", sched.get_last_lr()[0], ep)

        # ── validation ─
        model.eval(); preds, labs = [], []
        with torch.no_grad():
            for xb, yb_dir, *_ in dl_va:
                preds.extend(model(xb.to(DEVICE))[0].argmax(1).cpu().numpy())
                labs.extend(yb_dir.numpy())
        rep = classification_report(labs, preds, labels=LABELS,
                                    target_names=LABEL_NAMES,
                                    output_dict=True, zero_division=0)
        acc, f1 = rep["accuracy"], rep["macro avg"]["f1-score"]
        writer.add_scalar("Accuracy/Val", acc, ep)
        writer.add_scalar("F1/Val",      f1,  ep)

        print(f"📈 Ep{ep:02d} | loss={ep_loss:.2f} | acc={acc:.2%} | F1={f1:.2%} "
              f"| dist={dict(Counter(preds))}")

        # ── early stop ─
        if f1 > best_f1:
            best_f1, no_imp = f1, 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            no_imp += 1
            if no_imp >= PATIENCE: break

        torch.save({"epoch": ep, "model_state": model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "scheduler_state": sched.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "best_f1": best_f1, "epochs_no_improve": no_imp},
                   CHECKPOINT_PATH)

    # ── test set ─
    print("\n🧪 Testing best model …")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval(); preds, labs = [], []
    with torch.no_grad():
        for xb, yb_dir, *_ in dl_te:
            preds.extend(model(xb.to(DEVICE))[0].argmax(1).cpu().numpy())
            labs.extend(yb_dir.numpy())
    print(classification_report(labs, preds, labels=LABELS,
                                target_names=LABEL_NAMES, zero_division=0))
    writer.close()

# ── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train / resume Volatix-AI TFT")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from the matching checkpoint")
    train(resume=ap.parse_args().resume)
