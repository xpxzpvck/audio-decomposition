import numpy as np
from typing import List, Optional
import pandas as pd
import utils
import initialization
import nmf


def reconstruct_sources(
    X: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    score_annotations: Optional[pd.DataFrame] = None,
    pitch_set: Optional[np.ndarray] = None,
    frame_res: Optional[float] = None,
    epsilon: float = 1e-10,
) -> List[np.ndarray]:
    """Reconstruct sources using masks."""
    WH = np.dot(W, H)
    sources = []

    if score_annotations is None:
        for r in range(W.shape[1]):
            mask = np.outer(W[:, r], H[r, :]) / (WH + epsilon)
            sources.append(X * mask)
    else:
        for subset in utils.split_annotation(score_annotations):
            C_m, _, _ = initialization.init_H_score_onset(
                WH.shape[1], subset, frame_res, pitch_set=pitch_set
            )
            H_m = H * C_m
            WH_m = np.dot(W, H_m)
            mask = WH_m / (WH + epsilon)
            sources.append(X * mask)
    return sources

def _separate_score_informed(X, V, score_annotations, freq_res, frame_res, max_iter, threshold):
    pitch_set = utils.pitch_from_annotation(score_annotations)
    W_init, H_init = initialization.initialize_WH(V, score_annotations=score_annotations, pitch_set=pitch_set, freq_res=freq_res, frame_res=frame_res)
    W, H, _, _ = nmf(V, len(pitch_set)*2, W=W_init, H=H_init, max_iter=max_iter, threshold=threshold)
    return reconstruct_sources(X, W, H, score_annotations, pitch_set, frame_res)


def _separate_blind(X, V, R, max_iter, threshold):
    W_init, H_init = initialization.initialize_WH(V, R=R)
    W, H, _, _ = nmf.nmf(V, R=R, W=W_init, H=H_init, max_iter=max_iter, threshold=threshold)
    return reconstruct_sources(X, W, H)

def separate_sources(
    X: np.ndarray,
    R: Optional[int] = None,
    score_annotations: Optional[pd.DataFrame] = None,
    max_iter: int = 1000,
    threshold: float = 1e-4,
) -> List[np.ndarray]:
    """
    Perform source separation on complex spectrogram X.
    Uses either blind NMF (random init) or score-informed NMF (from annotations).
    """
    Fs = 22050
    N_fft = 2048
    H_fft = 1024
    freq_res = Fs / N_fft
    frame_res = H_fft / Fs
    V = np.abs(X)

    if score_annotations is not None:
        return _separate_score_informed(X, V, score_annotations, freq_res, frame_res, max_iter, threshold)
    else:
        if R is None:
            raise ValueError("Rank R must be specified for blind NMF.")
        return _separate_blind(X, V, R, max_iter, threshold)