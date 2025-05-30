import sys
import os
import numpy as np
import torch
import torch.nn as nn

# === Fix Python Path ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# === Import Model ===
from ai.tft_model import TemporalFusionTransformer

# === CONFIGURATION ===
MODEL_PATH = os.path.join(ROOT_DIR, "models", "tft_brain.pt")
SEQ_LEN    = 20   # Must match training
INPUT_DIM  = 60   # Must match feature count

_model = None  # Global cached model

def load_model():
    """
    Load the trained Temporal Fusion Transformer model.
    """
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
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
        _model.eval()
        print("✅ Transformer model loaded successfully.")

def predict_next_move(sequence: np.ndarray):
    """
    Predict the next market move from a given input sequence.
    
    Args:
        sequence (np.ndarray): Shape (SEQ_LEN, INPUT_DIM)
    
    Returns:
        dict: {
            'direction': 'Buy' | 'Sell' | 'Wait',
            'confidence': float,
            'reward': float
        }
    """
    if _model is None:
        load_model()

    if sequence.shape != (SEQ_LEN, INPUT_DIM):
        raise ValueError(f"❌ Invalid input shape: Expected {(SEQ_LEN, INPUT_DIM)}, got {sequence.shape}")

    input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # Shape: (1, SEQ_LEN, INPUT_DIM)
    
    with torch.no_grad():
        dir_logits, confidence, reward = _model(input_tensor)

    direction_idx = torch.argmax(dir_logits, dim=1).item()
    dir_map = {0: "Sell", 1: "Wait", 2: "Buy"}

    return {
        "direction": dir_map.get(direction_idx, "Unknown"),
        "confidence": round(confidence.item(), 4),
        "reward": round(reward.item(), 4)
    }

# === Standalone Test ===
if __name__ == "__main__":
    print("🧪 Running test prediction on dummy input...")
    dummy_input = np.random.randn(SEQ_LEN, INPUT_DIM)
    result = predict_next_move(dummy_input)
    print("🧠 Prediction Result:", result)
