# ai/tft_predictor.py

import torch
import torch.nn as nn
import numpy as np

MODEL_PATH = "models/tft_brain.pt"
SEQ_LEN = 20
FEATURE_DIM = 32  # update if different

# Must match the architecture used in tft_trainer.py
class TFTBrain(nn.Module):
    def __init__(self, input_size):
        super(TFTBrain, self).__init__()
        self.lstm = nn.LSTM(input_size, 64, batch_first=True)
        self.dropout = nn.Dropout(0.3)

        self.head_direction = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        self.head_confidence = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.head_reward = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        hn = self.dropout(hn[-1])
        return self.head_direction(hn), self.head_confidence(hn), self.head_reward(hn)

# Load model once and reuse
_model = None

def load_model():
    global _model
    _model = TFTBrain(input_size=FEATURE_DIM)
    _model.load_state_dict(torch.load(MODEL_PATH))
    _model.eval()
    print("✅ Model loaded: tft_brain.pt")

def predict_next_move(sequence: np.ndarray):
    """
    sequence: numpy array of shape (20, 32)
    returns: dict {direction, confidence, reward}
    """
    if _model is None:
        load_model()

    if sequence.shape != (SEQ_LEN, FEATURE_DIM):
        raise ValueError(f"Invalid input shape {sequence.shape}. Expected (20, {FEATURE_DIM})")

    input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # shape: (1, 20, 32)

    with torch.no_grad():
        out_dir, out_conf, out_reward = _model(input_tensor)

    dir_idx = torch.argmax(out_dir, dim=1).item()
    dir_map = {0: "Sell", 1: "Wait", 2: "Buy"}  # Adjust if label encoding was different

    return {
        "direction": dir_map.get(dir_idx, "Unknown"),
        "confidence": round(out_conf.item(), 4),
        "reward": round(out_reward.item(), 4)
    }

# 🔁 Test use
if __name__ == "__main__":
    dummy = np.random.randn(20, 32)
    result = predict_next_move(dummy)
    print("🧠 Prediction:", result)
