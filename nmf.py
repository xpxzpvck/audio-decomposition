import numpy as np
from typing import Tuple, List, Optional
import pandas as pd

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

def init_W_pitch_onset(K: int, pitch_set: np.ndarray, freq_res: float, tol_pitch: float = 0.05) -> np.ndarray:
    """Initialize template matrix with onset and sustained templates."""
    print(pitch_set)
    W = np.zeros((K, 2 * len(pitch_set)))
    for idx, pitch in enumerate(pitch_set):
        W[:, 2 * idx] = 0.1 
        W[:, 2 * idx + 1] = template_pitch(K, pitch, freq_res, tol_pitch)
    return W

def init_H_score_onset(
    N: int, annotation: pd.DataFrame, frame_res: float,
    tol_note: List[float] = [0.2, 0.5],
    tol_onset: List[float] = [0.3, 0.1],
    pitch_set: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize activation matrix using score annotations."""
    if pitch_set is None:
        pitch_set = pitch_from_annotation(annotation)
    
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

def split_annotation(annotation: pd.DataFrame) -> List[pd.DataFrame]:
    """Split annotation by label."""
    return [annotation[annotation["Label"] == l] for l in annotation["Label"].unique()]

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
        print("pitch_set", pitch_set)
        W_init = init_W_pitch_onset(V.shape[0], pitch_set, freq_res)
        H_init, _, _ = init_H_score_onset(V.shape[1], score_annotations, frame_res, pitch_set)
    return W_init, H_init

def reconstruct_sources(
    X: np.ndarray, W: np.ndarray, H: np.ndarray,
    score_annotations: Optional[pd.DataFrame] = None,
    pitch_set: Optional[np.ndarray] = None,
    frame_res: Optional[float] = None,
    epsilon: float = 1e-10
) -> List[np.ndarray]:
    """Reconstruct sources using masks."""
    WH = np.dot(W, H)
    sources = []

    if score_annotations is None:
        for r in range(W.shape[1]):
            mask = np.outer(W[:, r], H[r, :]) / (WH + epsilon)
            sources.append(X * mask)
    else:
        for subset in split_annotation(score_annotations):
            C_m, _, _ = init_H_score_onset(WH.shape[1], subset, frame_res, pitch_set=pitch_set)
            H_m = H * C_m
            WH_m = np.dot(W, H_m)
            mask = WH_m / (WH + epsilon)
            sources.append(X * mask)
    return sources

def nmf(
    V: np.ndarray, R: int,
    max_iter: int = 1000, threshold: float = 1e-4,
    W: Optional[np.ndarray] = None, H: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Perform Non-negative Matrix Factorization (NMF)."""
    K, N = V.shape
    W = np.random.rand(K, R) if W is None else W
    H = np.random.rand(R, N) if H is None else H
    eps = np.finfo(np.float64).eps

    for _ in range(max_iter):
        W_old, H_old = W.copy(), H.copy()

        H *= (W.T @ V) / (W.T @ W @ H + eps)
        W *= (V @ H.T) / (W @ H @ H.T + eps)

        if np.linalg.norm(H - H_old) < threshold and np.linalg.norm(W - W_old) < threshold:
            break

    for r in range(R):
        v_max = np.max(W[:, r])
        if v_max > 0:
            W[:, r] = W[:, r] / v_max
            H[r, :] = H[r, :] * v_max

    V_approx = W @ H
    error = np.linalg.norm(V - V_approx, 'fro')

    return W, H, V_approx, error


def _separate_score_informed(X, V, score_annotations, freq_res, frame_res, max_iter, threshold):
    pitch_set = pitch_from_annotation(score_annotations)
    W_init, H_init = initialize_WH(V, score_annotations=score_annotations, pitch_set=pitch_set, freq_res=freq_res, frame_res=frame_res)
    W, H, _, _ = nmf(V, len(pitch_set)*2, W=W_init, H=H_init, max_iter=max_iter, threshold=threshold)
    return reconstruct_sources(X, W, H, score_annotations, pitch_set, frame_res)


def _separate_blind(X, V, R, max_iter, threshold):
    W_init, H_init = initialize_WH(V, R=R)
    W, H, _, _ = nmf(V, R=R, W=W_init, H=H_init, max_iter=max_iter, threshold=threshold)
    return reconstruct_sources(X, W, H)

def separate_sources(
    X: np.ndarray,
    R: Optional[int] = None,
    score_annotations: Optional[pd.DataFrame] = None,
    freq_res: Optional[float] = None,
    frame_res: Optional[float] = None,
    max_iter: int = 1000,
    threshold: float = 1e-4
) -> List[np.ndarray]:
    """
    Perform source separation on complex spectrogram X.
    Uses either blind NMF (random init) or score-informed NMF (from annotations).
    """
    V = np.abs(X)

    if score_annotations is not None:
        return _separate_score_informed(X, V, score_annotations, freq_res, frame_res, max_iter, threshold)
    else:
        if R is None:
            raise ValueError("Rank R must be specified for blind NMF.")
        return _separate_blind(X, V, R, max_iter, threshold)
