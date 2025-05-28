# ai/tft_trainer.py
import os, sys, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data import prepare_tft_data

MODEL_PATH   = "models/tft_brain.pt"
BATCH_SIZE   = 128
EPOCHS       = 40
LR           = 1e-3
DIR_W, CONF_W, REW_W = 1.0, 0.4, 0.1          # loss weights
CLASS_W      = torch.tensor([2.0, 1.0, 2.0])  # Sell, Wait, Buy

# ── Model ────────────────────────────────────────────────────────────
class TFTBrain(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm  = nn.LSTM(input_dim, 128,
                             num_layers=2,
                             dropout=0.2,
                             batch_first=True)
        self.drop  = nn.Dropout(0.3)
        self.dir   = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
        self.conf  = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.rew   = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = self.drop(h[-1])
        return self.dir(h), self.conf(h), self.rew(h)

# ── Data loader ──────────────────────────────────────────────────────
def load_or_prepare():
    req = ["X.npy","y_direction.npy","y_confidence.npy","y_reward.npy"]
    if not all(os.path.exists(f"historical_data/{f}") for f in req):
        print("⚙️  Running data preparation…")
        prepare_tft_data.main()

    X      = np.load("historical_data/X.npy")
    y_dir  = np.load("historical_data/y_direction.npy")
    y_conf = np.load("historical_data/y_confidence.npy")
    y_rew  = np.load("historical_data/y_reward.npy")

    # convert to tensors
    X      = torch.tensor(X,      dtype=torch.float32)
    y_dir  = torch.tensor(y_dir,  dtype=torch.long)
    y_conf = torch.tensor(y_conf, dtype=torch.float32).unsqueeze(1)
    y_rew  = torch.tensor(y_rew,  dtype=torch.float32).unsqueeze(1)

    ds     = TensorDataset(X, y_dir, y_conf, y_rew)
    tr_sz  = int(len(ds)*0.8)
    return DataLoader(ds, BATCH_SIZE, shuffle=True, drop_last=True), \
           DataLoader(torch.utils.data.Subset(ds, range(tr_sz, len(ds))), BATCH_SIZE)

# ── Training loop ────────────────────────────────────────────────────
def train():
    train_dl, val_dl = load_or_prepare()
    model     = TFTBrain(input_dim=train_dl.dataset.tensors[0].shape[2])
    opt       = torch.optim.AdamW(model.parameters(), lr=LR)
    sch       = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.7)
    ce_loss   = nn.CrossEntropyLoss(weight=CLASS_W)
    mse       = nn.MSELoss()

    for epoch in range(1, EPOCHS+1):
        model.train(); tot = 0
        for xb, yb_dir, yb_conf, yb_rew in train_dl:
            opt.zero_grad()
            o_dir, o_conf, o_rew = model(xb)
            loss = ( DIR_W  * ce_loss(o_dir, yb_dir)
                   + CONF_W * mse(o_conf, yb_conf)
                   + REW_W  * mse(o_rew,  yb_rew) )
            loss.backward(); opt.step(); tot += loss.item()
        sch.step()

        # ─ validation
        model.eval(); correct = n = 0
        with torch.no_grad():
            for xb, yb_dir, *_ in val_dl:
                pred = model(xb)[0].argmax(1)
                correct += (pred == yb_dir).sum().item()
                n += len(yb_dir)
        acc = correct / n if n else 0
        print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={tot:.2f} | val_acc={acc:.2%}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print("💾 Model saved →", MODEL_PATH)

if __name__ == "__main__":
    train()
