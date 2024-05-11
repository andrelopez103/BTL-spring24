#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 16:57:03 2024

@author: andrelopez
"""
import librosa
import numpy as np

def dtw_distance(signal1, signal2):
    # Compute the Short-Time Fourier Transform (STFT) of the signals
    stft1 = librosa.stft(signal1)
    stft2 = librosa.stft(signal2)
    
    # Compute the magnitude spectrograms
    mag_stft1 = np.abs(stft1)
    mag_stft2 = np.abs(stft2)
    
    # Compute the cost matrix based on Euclidean distance between magnitude spectrogram frames
    cost_matrix = np.zeros((mag_stft1.shape[1], mag_stft2.shape[1]))
    for i in range(mag_stft1.shape[1]):
        for j in range(mag_stft2.shape[1]):
            cost_matrix[i, j] = np.linalg.norm(mag_stft1[:, i] - mag_stft2[:, j])
    
    # Initialize the accumulated cost matrix
    accumulated_cost = np.zeros((mag_stft1.shape[1], mag_stft2.shape[1]))
    
    # Compute accumulated cost matrix using dynamic programming
    accumulated_cost[0, 0] = cost_matrix[0, 0]
    for i in range(1, mag_stft1.shape[1]):
        accumulated_cost[i, 0] = accumulated_cost[i-1, 0] + cost_matrix[i, 0]
    for j in range(1, mag_stft2.shape[1]):
        accumulated_cost[0, j] = accumulated_cost[0, j-1] + cost_matrix[0, j]
    for i in range(1, mag_stft1.shape[1]):
        for j in range(1, mag_stft2.shape[1]):
            accumulated_cost[i, j] = cost_matrix[i, j] + min(accumulated_cost[i-1, j],
                                                             accumulated_cost[i, j-1],
                                                             accumulated_cost[i-1, j-1])
    
    # Compute the DTW distance
    dtw_distance = accumulated_cost[-1, -1] / (mag_stft1.shape[1] + mag_stft2.shape[1])
    """
    # Normalize the DTW distance between 0 and 1
    max_possible_distance = mag_stft1.shape[1] + mag_stft2.shape[1]
    normalized_dtw_distance = dtw_distance / max_possible_distance
    """
    return dtw_distance

# Example usage
if __name__ == "__main__":
    # Load audio signals
    signal1, sr1 = librosa.load("New-table.wav", sr=None, offset=None, duration=None)
    signal2, sr2 = librosa.load("New-table-metal.wav", sr=None, offset=4.5, duration=4)
    
    # Compute normalized DTW distance
    normalized_distance = dtw_distance(signal1, signal2)
    print("Normalized DTW Distance:", normalized_distance)

