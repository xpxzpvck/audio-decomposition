import numpy as np
import librosa

def audio_to_spectrogram(audio):
    '''
    Convert audio to a magnitude spectrogram using Short-Time Fourier Transform (STFT).
    Parameters:
        audio (1D array): Input audio signal.
    Returns:
        V (2D array): Nonnegative magnitude spectrogram of the audio signal.
    '''
    V = np.abs(librosa.stft(audio)).astype(np.float64)
    return V

def spectrogram_to_audio(V):
    '''
    Convert a magnitude spectrogram back to audio using the inverse Short-Time Fourier Transform (ISTFT).
    Parameters:
        V (2D array): Input magnitude spectrogram.
    Returns:
        audio (1D array): Reconstructed audio signal.
    '''
    audio = librosa.istft(V)
    return audio

def get_components(W, H):
    '''
    Get the components from W and H matrices.
    Parameters:
        W (2D array): Template matrix.
        H (2D array): Activation matrix.
    Returns:
        components (list of 1D arrays): List of components.
    '''
    components = []
    R = H.shape[0]
    for r in range(R):
        component = np.outer(W[:, r], H[r, :])
        components.append(component)
    return components

def nmf(V, R, max_iter=1000, threshold=0.0001, W=None, H=None):
    '''
    Perform Non-negative Matrix Factorization (NMF) on the input spectrogram V.
    Parameters:
        V (2D array): Input magnitude spectrogram.
        R (int): Number of components to decompose into.
        max_iter (int): Maximum number of iterations for convergence.
        threshold (float): Convergence threshold.
        W (2D array): Initial template matrix (optional).
        H (2D array): Initial activation matrix (optional).
    Returns:
        W (2D array): Template matrix after decomposition.
        H (2D array): Activation matrix after decomposition.
        V_approx (2D array): Reconstructed spectrogram from W and H.
        V_approx_error (float): Reconstruction error.
    '''
    K, N = V.shape

    if not W:
        W = np.random.rand(K, R).astype(np.float64)
    if not H:
        H = np.random.rand(R, N).astype(np.float64)

    eps = np.finfo(np.float64).eps

    for _ in range(max_iter):
        W_old = W.copy()
        H_old = H.copy()

        # Update H
        H = H * (W.T @ V) / (W.T @ W @ H + eps)  # Adding small epsilon to avoid division by zero

        # Update W
        W = W * (V @ H.T) / (W @ H @ H.T + eps)

        # Compute error
        H_diff = np.linalg.norm(H - H_old, ord=2)
        W_diff = np.linalg.norm(W - W_old, ord=2)

        # Check for convergence
        if H_diff < threshold and W_diff < threshold:
            break

    # Normalize W and H
    for r in range(R):
        v_max = np.max(W[:, r])
        if v_max > 0:
            W[:, r] = W[:, r] / v_max
            H[r, :] = H[r, :] * v_max

    V_approx = W @ H
    V_approx_error = np.linalg.norm(V - V_approx, ord=2)

    return W, H, V_approx, V_approx_error
