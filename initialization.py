import numpy as np
import csv


def csv_to_list(file_path):
    """Reads a CSV file and returns its content as a list of lists
    Parameters:
        file_path (str): Path to the CSV file

    Returns:
        data (list): List of lists containing the CSV data
    """
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        data = list(reader)
    return data


def pitch_from_annotation(annotation):
    """Extract set of occurring pitches from annotation
    Parameters:
        annotation (list): Annotation data

    Returns:
        pitch_set (np.ndarray): Set of occurring pitches
    """
    pitch_all = np.array([c[2] for c in annotation])
    pitch_set = np.unique(pitch_all)
    return pitch_set


def template_pitch(K, pitch, freq_res, tol_pitch=0.05):
    """Defines spectral template for a given pitch
    Parameters:
        K (int): Number of frequency points
        pitch (float): Fundamental pitch
        freq_res (float): Frequency resolution
        tol_pitch (float): Relative frequency tolerance for the harmonics (Default value = 0.05)

    Returns:
        template (np.ndarray): Nonnegative template vector of size K
    """
    max_freq = K * freq_res
    pitch_freq = 2 ** ((pitch - 69) / 12) * 440
    max_order = int(np.ceil(max_freq / ((1 - tol_pitch) * pitch_freq)))
    template = np.zeros(K)
    for m in range(1, max_order + 1):
        min_idx = max(0, int((1 - tol_pitch) * m * pitch_freq / freq_res))
        max_idx = min(K - 1, int((1 + tol_pitch) * m * pitch_freq / freq_res))
        template[min_idx : max_idx + 1] = 1 / m
    return template


def init_W(K, pitch_set, freq_res, tol_pitch=0.05):
    """Initializes template matrix for a given set of pitches
    Parameters:
        K (int): Number of frequency points
        pitch_set (np.ndarray): Set of fundamental pitches
        freq_res (float): Frequency resolution
        tol_pitch (float): Relative frequency tolerance for the harmonics

    Returns:
        W (np.ndarray): Nonnegative matrix of size K x R with R is the number of unique pitches
    """
    R = len(pitch_set)
    W = np.zeros((K, R))
    for r in range(R):
        W[:, r] = template_pitch(K, pitch_set[r], freq_res, tol_pitch)
    return W


def init_H(N, annotation, frame_res, tol_note=[0.2, 0.5], pitch_set=None):
    """Initializes activation matrix for given score annotations
    Parameters:
        N (int): Number of frames
        annotation (list): Annotation data
        frame_res (time): Time resolution
        tol_note (list or np.ndarray): Tolerance (seconds) for beginning and end of a note (Default value = [0.2, 0.5])
        pitch_set (np.ndarray): Set of occurring pitches (Default value = None)

    Returns:
        H (np.ndarray): Nonnegative matrix of size R x N
        pitch_set (np.ndarray): Set of occurring pitches
    """
    note_start = np.array([c[0] for c in annotation])
    note_duration = np.array([c[1] for c in annotation])
    pitch_all = np.array([c[2] for c in annotation])
    if pitch_set is None:
        pitch_set = np.unique(pitch_all)
    R = len(pitch_set)
    H = np.zeros((R, N))
    for i in range(len(note_start)):
        start_idx = max(0, int((note_start[i] - tol_note[0]) / frame_res))
        end_idx = min(N, int((note_start[i] + note_duration[i] + tol_note[1]) / frame_res))
        pitch_idx = np.argwhere(pitch_set == pitch_all[i])
        H[pitch_idx, start_idx:end_idx] = 1
    return H, pitch_set
