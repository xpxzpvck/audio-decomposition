import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import utils

def init_W_pitch_onset(K: int, pitch_set: np.ndarray, freq_res: float, tol_pitch: float = 0.05) -> np.ndarray:
    """Initialize template matrix with onset and sustained templates."""
    W = np.zeros((K, 2 * len(pitch_set)))
    for idx, pitch in enumerate(pitch_set):
        W[:, 2 * idx] = 0.1 
        W[:, 2 * idx + 1] = utils.template_pitch(K, pitch, freq_res, tol_pitch)
    return W

def init_H_score_onset(
    N: int, annotation: pd.DataFrame, frame_res: float,
    tol_note: List[float] = [0.2, 0.5],
    tol_onset: List[float] = [0.3, 0.1],
    pitch_set: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize activation matrix using score annotations."""
    if pitch_set is None:
        pitch_set = utils.pitch_from_annotation(annotation)
    
    pitch_to_idx = {pitch: idx for idx, pitch in enumerate(pitch_set)}
    
    H = np.zeros((2 * len(pitch_set), N))
    note_start = annotation["Start"].to_numpy()
    note_dur = annotation["Duration"].to_numpy()
    pitch_all = annotation["Pitch"].to_numpy()

    for start, dur, pitch in zip(note_start, note_dur, pitch_all):
        idx = pitch_to_idx[pitch]
        start_idx = max(0, int((start - tol_note[0]) / frame_res))
        end_idx = min(N, int((start + dur + tol_note[1]) / frame_res))
        onset_start_idx = max(0, int((start - tol_onset[0]) / frame_res))
        onset_end_idx = min(N, int((start + tol_onset[1]) / frame_res))
        
        H[2 * idx, onset_start_idx:onset_end_idx] = 1
        H[2 * idx + 1, start_idx:end_idx] = 1

    label_pitch = np.repeat(pitch_set, 2)
    
    return H, pitch_set, label_pitch

def initialize_WH(
    V: np.ndarray, R: int = None,
    score_annotations: Optional[pd.DataFrame] = None,
    pitch_set: Optional[np.ndarray] = None,
    freq_res: Optional[float] = None, frame_res: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize W and H matrices randomly or score-informed."""
    if score_annotations is None:
        W_init = np.random.rand(V.shape[0], R)
        H_init = np.random.rand(R, V.shape[1])
    else:
        W_init = init_W_pitch_onset(V.shape[0], pitch_set, freq_res)
        H_init, _, _ = init_H_score_onset(V.shape[1], score_annotations, frame_res, pitch_set)
    return W_init, H_init