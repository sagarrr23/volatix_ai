import os
import sys
import torch
import numpy as np
import torch.serialization

# === Root Path Fix ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

# === Imports ===
from ai.tft_trainer import AdvancedTFT
from core.config_loader import BotConfig

# === Constants from Config ===
SEQ_LEN = BotConfig.SEQ_LEN
FEATURE_DIM = BotConfig.TFT_FEATURE_DIM
MODEL_PATH = os.path.join(ROOT_DIR, BotConfig.MODEL_PATH)
FUTURE_STEP_DEFAULT = 1

# === Global Cache ===
_model = None
_checkpoint = None

def load_model():
    """
    Loads the trained TFT model from a full checkpoint with PyTorch 2.6+ safe unpickling.
    """
    global _model, _checkpoint

    if _model is None:
        _model = AdvancedTFT(input_size=FEATURE_DIM)

        # ✅ PyTorch 2.6 fix: allow safe globals for NumPy scalar
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])

        # ✅ Load full checkpoint
        _checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)
        if 'model_state_dict' not in _checkpoint:
            raise RuntimeError("❌ Checkpoint does not contain 'model_state_dict'. Verify training code.")

        _model.load_state_dict(_checkpoint['model_state_dict'])
        _model.eval()

        print("✅ TFT model loaded from checkpoint.")
        print(f"📈 Accuracy: {_checkpoint.get('accuracy', 'N/A')}")
        print(f"📅 Epoch: {_checkpoint.get('epoch', 'N/A')}")

def predict_next(sequence: np.ndarray, future_step: int = FUTURE_STEP_DEFAULT) -> dict:
    """
    Predict market direction, confidence, and reward using the trained model.

    Args:
        sequence (np.ndarray): Input sequence of shape (SEQ_LEN, FEATURE_DIM)
        future_step (int): Horizon to predict [1, 3, 6]

    Returns:
        dict: { direction: str, confidence: float, reward: float }
    """
    if _model is None:
        load_model()

    if sequence.shape != (SEQ_LEN, FEATURE_DIM):
        raise ValueError(f"❌ Invalid input shape: Expected {(SEQ_LEN, FEATURE_DIM)}, got {sequence.shape}")

    x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # Shape: (1, SEQ_LEN, FEATURE_DIM)

    with torch.no_grad():
        output = _model(x)
        step_out = output.get(future_step)
        if step_out is None:
            raise ValueError(f"❌ No output for step {future_step}. Valid: {list(output.keys())}")

    direction_idx = torch.argmax(step_out["direction"], dim=1).item()

    return {
        "direction": {0: "Sell", 1: "Wait", 2: "Buy"}.get(direction_idx, "Unknown"),
        "confidence": round(float(step_out["confidence"].item()), 4),
        "reward": round(float(step_out["reward"].item()), 4)
    }

# === CLI Test Mode ===
if __name__ == "__main__":
    print("🔍 Testing TFT Predictor...")
    dummy_input = np.random.randn(SEQ_LEN, FEATURE_DIM)
    try:
        result = predict_next(dummy_input, future_step=1)
        print("🧠 Prediction:", result)
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
    print("✅ Test completed.")
