import numpy as np
from typing import Tuple, Optional

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
