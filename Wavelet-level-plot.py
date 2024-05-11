#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 22:07:39 2024

@author: andrelopez
"""


import librosa
import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# List of audio files
audio_files = ["New-Metal.wav", "New-table.wav"]  # Add the second audio file here

# Define the wavelet and decomposition level
wavelet = 'db4'  # Choose the wavelet type, such as 'db4'
level = 3  # Choose the decomposition level

# Load the audio files and perform wavelet decomposition
coeffs_list = []
min_length = float('inf')
for file in audio_files:
    y, sr = librosa.load(file)
    coeffs = pywt.wavedec(y, wavelet, level=level)
    coeffs_list.append(coeffs)
    min_length = min(min_length, min(len(c) for c in coeffs))

# Calculate time values for x-axis in seconds for each level
time_values_list = []
for coeffs in coeffs_list:
    min_length = min(len(c) for c in coeffs)
    time_values_list.append(np.linspace(0, min_length / sr, min_length))

# Loop over each coefficient level
for i in range(len(coeffs_list[0])):
    # Initialize lists to store trimmed coefficients and time values
    trimmed_coeffs = []
    time_values_list = []

    # Calculate time values for x-axis in seconds for each file and level
    for coeffs in coeffs_list:
        min_length = min(len(c) for c in coeffs)
        time_values_list.append(np.linspace(0, min_length / sr, min_length))

    # Find the minimum length among the time values arrays
    min_length_time_values = min(len(t) for t in time_values_list)

    # Trim time values to match the minimum length
    time_values = time_values_list[0][:min_length_time_values]

    # Plot the wavelet packet coefficients for each audio file
    plt.figure(figsize=(10, 6))
    for j, coeffs in enumerate(coeffs_list):
        trimmed_coeffs.append(coeffs[i][:min_length_time_values])
        plt.plot(time_values, trimmed_coeffs[-1], label=f'Audio {j+1}')  # Use trimmed time_values for each audio file

    plt.title(f'Wavelet Packet Coefficients - Level {i+1}', fontsize=16)
    plt.xlabel('Time (s)', fontsize=16)  # Set x-axis label to time in seconds
    plt.ylabel('Magnitude', fontsize=16)
    plt.legend()  # Show legend with file names

    # Set x-axis limits based on the calculated time values
    plt.xlim([0, time_values[-1]])
    #tightfigure 
    plt.tight_layout()
    # Save the plot as a PNG file
    plt.savefig(f'wavelet_packet_display_level_{i+1}.png')

    plt.show()












