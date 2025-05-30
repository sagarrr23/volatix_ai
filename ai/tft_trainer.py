import os, sys, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data import prepare_tft_data
from ai.tft_model import TemporalFusionTransformer

# === CONFIGURATION ===
MODEL_PATH = "models/tft_brain.pt"
BATCH_SIZE = 128
EPOCHS = 40
LR = 1e-3
DIR_W, CONF_W, REW_W = 1.0, 0.4, 0.1
CLASS_W = torch.tensor([2.0, 1.0, 2.0])  # Sell, Wait, Buy
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === PREPARE DATA ===
def load_or_prepare():
    req = ["X.npy", "y_direction.npy", "y_confidence.npy", "y_reward.npy"]
    if not all(os.path.exists(f"historical_data/{f}") for f in req):
        print("⚙️  Running data preparation…")
        prepare_tft_data.main()

    # Load and convert to torch tensors
    X = torch.tensor(np.load("historical_data/X.npy"), dtype=torch.float32)
    y_dir = torch.tensor(np.load("historical_data/y_direction.npy"), dtype=torch.long)
    y_conf = torch.tensor(np.load("historical_data/y_confidence.npy"), dtype=torch.float32).unsqueeze(1)
    y_rew = torch.tensor(np.load("historical_data/y_reward.npy"), dtype=torch.float32).unsqueeze(1)

    # Wrap into dataset
    dataset = TensorDataset(X, y_dir, y_conf, y_rew)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    return DataLoader(train_ds, BATCH_SIZE, shuffle=True, drop_last=True), \
           DataLoader(val_ds, BATCH_SIZE)

# === TRAINING FUNCTION ===
def train():
    print("🚀 Starting TFT Brain Training...")
    train_dl, val_dl = load_or_prepare()

    # Infer shape
    sample_input = train_dl.dataset[0][0]
    seq_len = sample_input.shape[0]
    input_dim = sample_input.shape[1]

    # Init model
    model = TemporalFusionTransformer(
        input_dim=input_dim,
        hidden_size=256,
        seq_len=seq_len,
        heads=4,
        num_layers=4,
        dropout=0.3
    ).to(DEVICE)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

    # Loss functions
    ce_loss = nn.CrossEntropyLoss(weight=CLASS_W.to(DEVICE))
    mse = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    print(f"📊 Model input shape: {seq_len} x {input_dim} | Device: {DEVICE}")

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for xb, yb_dir, yb_conf, yb_rew in train_dl:
            xb, yb_dir = xb.to(DEVICE), yb_dir.to(DEVICE)
            yb_conf, yb_rew = yb_conf.to(DEVICE), yb_rew.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                out_dir, out_conf, out_rew = model(xb)
                loss = (DIR_W * ce_loss(out_dir, yb_dir) +
                        CONF_W * mse(out_conf, yb_conf) +
                        REW_W * mse(out_rew, yb_rew))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        optimizer.step()
        scheduler.step()

        # === VALIDATION ===
        model.eval(); correct = total = 0
        with torch.no_grad():
            for xb, yb_dir, *_ in val_dl:
                xb, yb_dir = xb.to(DEVICE), yb_dir.to(DEVICE)
                with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    preds = model(xb)[0].argmax(dim=1)
                correct += (preds == yb_dir).sum().item()
                total += len(yb_dir)

        val_acc = correct / total if total else 0.0
        print(f"📈 Epoch {epoch:02d}/{EPOCHS} | Loss: {total_loss:.4f} | Val Acc: {val_acc:.2%}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Model saved (Best so far 🔥): {MODEL_PATH}")

    print(f"🏁 Training complete | Best Val Acc: {best_val_acc:.2%} | Model saved to: {MODEL_PATH}")

# === ENTRY POINT ===
if __name__ == "__main__":
    train()
