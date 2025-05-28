# ai/test_tft_performance.py
import os, sys, numpy as np, torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels

# ── import the identical model class used in training ───────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ai.tft_trainer import TFTBrain, MODEL_PATH   # 💡 class + path

# ── load sequences & labels ─────────────────────────────────────────
X      = np.load("historical_data/X.npy")
y_dir  = np.load("historical_data/y_direction.npy")
y_conf = np.load("historical_data/y_confidence.npy")
y_rwd  = np.load("historical_data/y_reward.npy")

X_t = torch.tensor(X, dtype=torch.float32)

# ── build model & load weights ──────────────────────────────────────
model = TFTBrain(input_dim=X.shape[2])
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ── inference ───────────────────────────────────────────────────────
with torch.no_grad():
    out_dir, out_conf, out_rwd = model(X_t)

pred_labels = out_dir.argmax(1).numpy()
conf_pred   = out_conf.numpy().flatten()
rwd_pred    = out_rwd.numpy().flatten()

# ── metrics ─────────────────────────────────────────────────────────
acc = accuracy_score(y_dir, pred_labels)
print(f"\n✅ Overall Accuracy: {acc*100:.2f}%")

labels = unique_labels(y_dir, pred_labels)           # present classes only
name_map = {0: "Sell", 1: "Wait", 2: "Buy"}
names = [name_map[l] for l in labels]

print("\n🧾 Classification Report:")
print(classification_report(
    y_dir, pred_labels,
    labels=labels,
    target_names=names,
    zero_division=0
))

print("📊 Confusion Matrix (rows=true, cols=pred):")
print(confusion_matrix(y_dir, pred_labels, labels=labels))

print(f"\n🔐 Mean Confidence  | true: {y_conf.mean():.4f} | pred: {conf_pred.mean():.4f}")

rwd_corr = np.corrcoef(y_rwd, rwd_pred)[0,1]
print(f"💰 Reward correlation: {rwd_corr:.4f}")

# ── sample rows ─────────────────────────────────────────────────────
print("\n📌 Sample predictions:")
for i in range(min(5, len(X))):
    print(f"{i+1}. true={y_dir[i]}  pred={pred_labels[i]}  "
          f"conf={conf_pred[i]:.3f}  reward={rwd_pred[i]:.4f}")
