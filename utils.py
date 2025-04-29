import pandas as pd
import numpy as np
from typing import List

def csv_to_dataframe(file_path):
    """Read a CSV file and returns its content as a pandas DataFrame."""
    df = pd.read_csv(file_path, delimiter=';')
    return df

def pitch_from_annotation(annotation: pd.DataFrame) -> np.ndarray:
    """Extract unique pitches from annotation."""
    return np.unique(annotation["Pitch"].to_numpy())

def template_pitch(K: int, pitch: float, freq_res: float, tol_pitch: float = 0.05) -> np.ndarray:
    """Define spectral template for a pitch."""
    max_freq = K * freq_res
    pitch_freq = 2 ** ((pitch - 69) / 12) * 440
    max_order = int(np.ceil(max_freq / ((1 - tol_pitch) * pitch_freq)))
    
    template = np.zeros(K)
    for m in range(1, max_order + 1):
        min_idx = int(max(0, (1 - tol_pitch) * m * pitch_freq / freq_res))
        max_idx = int(min(K - 1, (1 + tol_pitch) * m * pitch_freq / freq_res))
        template[min_idx:max_idx + 1] = 1 / m 
    return template

def split_annotation(annotation: pd.DataFrame) -> List[pd.DataFrame]:
    """Split annotation by label."""
    return [annotation[annotation["Label"] == l] for l in annotation["Label"].unique()]