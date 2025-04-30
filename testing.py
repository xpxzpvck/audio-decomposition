import numpy as np

def compute_cosine_similarity(mag_clean, mag_estimate):
    """
    Compute cosine similarity between two STFT magnitude spectrograms.
    """
    mag_clean_flat = mag_clean.flatten()
    mag_estimate_flat = mag_estimate.flatten()

    dot_product = np.dot(mag_clean_flat, mag_estimate_flat)
    magnitude_A = np.linalg.norm(mag_clean_flat)
    magnitude_B = np.linalg.norm(mag_estimate_flat)
    return dot_product / (magnitude_A * magnitude_B)

def compute_normalized_cross_correlation(mag_clean, mag_estimate):
    """
    Compute normalized cross-correlation (NCC) between two spectrograms.
    Equivalent to Pearson correlation over flattened arrays.
    """
    assert mag_clean.shape == mag_estimate.shape, "STFT shapes must match"
    v1 = mag_clean.flatten()
    v2 = mag_estimate.flatten()
    v1 = (v1 - np.mean(v1)) / (np.std(v1) + 1e-10)
    v2 = (v2 - np.mean(v2)) / (np.std(v2) + 1e-10)
    return np.mean(v1 * v2)

def compute_snr_db(mag_clean, mag_estimate):
    """
    Compute Signal-to-Noise Ratio (SNR) in decibels.
    """
    assert mag_clean.shape == mag_estimate.shape, "STFT shapes must match"
    noise = mag_estimate - mag_clean
    noise = np.maximum(noise, 1e-10)
    power_clean = np.sum(mag_clean ** 2)
    power_noise = np.sum(noise ** 2)
    return 10 * np.log10(power_clean / power_noise)

def evaluate_sources(reference_sources, estimated_sources):
    metrics = []
    for ref, est in zip(reference_sources, estimated_sources):
        min_time_dim = min(ref.shape[1], est.shape[1])
        mag_clean = ref[:, :min_time_dim]
        mag_estimate = est[:, :min_time_dim]
        
        cs = compute_cosine_similarity(mag_clean, mag_estimate)
        cc = compute_normalized_cross_correlation(mag_clean, mag_estimate)
        snr = compute_snr_db(mag_clean, mag_estimate)
        metrics.append({'cosine_similarity': cs, 'cross_correlation': cc, "signal_to_noise": snr})
    return metrics
