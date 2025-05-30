import torch
import numpy as np
import os

from ai.tft_model import TemporalFusionTransformer  # Correct brain
MODEL_PATH = "models/tft_brain.pt"
SEQ_LEN    = 20  # Same as used in training
INPUT_DIM  = 60  # You must update if your feature count changes

_model = None

def load_model():
    global _model
    if _model is None:
        _model = TemporalFusionTransformer(
            input_dim=INPUT_DIM,
            hidden_size=256,
            seq_len=SEQ_LEN,
            heads=4,
            num_layers=4,
            dropout=0.3
        )
        _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        _model.eval()
        print("✅ Transformer model loaded.")


def predict_next_move(sequence: np.ndarray):
    """
    sequence: numpy array of shape (SEQ_LEN, INPUT_DIM)
    returns: dict with direction, confidence, reward
    """
    if _model is None:
        load_model()

    if sequence.shape != (SEQ_LEN, INPUT_DIM):
        raise ValueError(f"Expected input shape {(SEQ_LEN, INPUT_DIM)}, got {sequence.shape}")

    input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, input_dim)
    with torch.no_grad():
        out_dir, out_conf, out_reward = _model(input_tensor)

    dir_idx = torch.argmax(out_dir, dim=1).item()
    dir_map = {0: "Sell", 1: "Wait", 2: "Buy"}

    return {
        "direction": dir_map.get(dir_idx, "Unknown"),
        "confidence": round(out_conf.item(), 4),
        "reward": round(out_reward.item(), 4)
    }

# 🔁 Test
if __name__ == "__main__":
    dummy = np.random.randn(SEQ_LEN, INPUT_DIM)
    result = predict_next_move(dummy)
    print("🧠 Prediction:", result)
