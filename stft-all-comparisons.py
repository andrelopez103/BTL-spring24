#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:12:18 2024

@author: andrelopez
"""

import librosa
import numpy as np
from scipy.stats import pearsonr

# List of audio files
audio_files = ["New-Metal.wav", "New-table.wav", "New-table-metal.wav"]

# Specify length and offset for "Table-Metal-fc.wav" (in seconds)
table_metal_length = 4  # Example length
table_metal_offset = 4.5  # Example offset

# Dictionary to store signal lengths and offsets
signal_lengths_offsets = {}

# Iterate over each audio file
for audio_file in audio_files:
    # Specify length and offset for "Table-Metal-fc.wav" and leave other lengths and offsets as None
    length_offset = (table_metal_length, table_metal_offset) if audio_file == "New-table-metal.wav" else (None, None)
    signal_lengths_offsets[audio_file] = length_offset

# Dictionary to store similarity measures for each pair
similarity_measures = {}

# Iterate over each combination of files
for i in range(len(audio_files)):
    for j in range(i + 1, len(audio_files)):
        # Load audio signals and compute STFTs with specified length and offset
        length_i, offset_i = signal_lengths_offsets.get(audio_files[i])
        length_j, offset_j = signal_lengths_offsets.get(audio_files[j])

        y1, sr = librosa.load(audio_files[i], offset=offset_i, duration=length_i)
        y2, sr = librosa.load(audio_files[j], offset=offset_j, duration=length_j)

        # Ensure the signals have the same length
        min_len = min(len(y1), len(y2))
        y1 = y1[:min_len]
        y2 = y2[:min_len]

        # Compute STFTs
        stft_1 = librosa.stft(y1)
        stft_2 = librosa.stft(y2)

        # Flatten the STFT matrices to 1D vectors
        stft_1_flat = stft_1.flatten()
        stft_2_flat = stft_2.flatten()

        # Calculate Mean Squared Error (MSE)
        mse = np.mean((np.abs(stft_1) - np.abs(stft_2))**2)

        # Calculate Euclidean distance
        euclidean_distance = np.linalg.norm(stft_1_flat - stft_2_flat)

        # Calculate Pearson correlation coefficient
        correlation_coefficient, _ = pearsonr(stft_1_flat, stft_2_flat)

        # Calculate absolute value of correlation coefficient
        correlation_magnitude = np.abs(correlation_coefficient)

        # Store similarity measures for the pair
        pair_key = (audio_files[i], audio_files[j])
        similarity_measures[pair_key] = {
            "MSE": mse,
            "Euclidean Distance": euclidean_distance,
            "Correlation Magnitude": correlation_magnitude
        }

# Print similarity measures for each pair
for pair, measures in similarity_measures.items():
    print("Pair:", pair)
    print("MSE:", measures["MSE"])
    print("Euclidean Distance:", measures["Euclidean Distance"])
    print("Correlation Magnitude:", measures["Correlation Magnitude"])
    print()



