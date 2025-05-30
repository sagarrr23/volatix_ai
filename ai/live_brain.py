# ai/live_brain.py
import numpy as np
from ai.tft_predictor import predict_next_move

SEQ_LEN = 20
INPUT_DIM = 60

buffer = []  # Holds last 20 rows (each row = 60 features)

def update_brain_input(new_row: np.ndarray):
    """
    Accepts a new row of live feature data (shape = [60])
    """
    global buffer
    if new_row.shape != (INPUT_DIM,):
        raise ValueError(f"Expected shape (60,), got {new_row.shape}")
    
    buffer.append(new_row)
    if len(buffer) > SEQ_LEN:
        buffer.pop(0)

    if len(buffer) == SEQ_LEN:
        return predict_next_move(np.array(buffer))
    return None
