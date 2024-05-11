#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 22:42:02 2024

@author: andrelopez
"""

from scipy.signal import butter, filtfilt
import librosa
import matplotlib.pyplot as plt
import numpy as np

# Function to create a high-pass Butterworth filter
def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

# Function to apply a filter to an audio signal
def apply_filter(data, cutoff_freq, fs, order=5):
    b, a = butter_highpass(cutoff_freq, fs, order=order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Load audio signal
y, sr = librosa.load('Sample1-Metal.wav', sr=None)
y2, sr2 = librosa.load('Sample2-Metal.wav', sr=None)
y3, sr3 = librosa.load('Sample1-phone.wav', sr=None)
y4, sr4 = librosa.load('Sample2-phone.wav', sr=None)

# Define high-pass filter parameters
cutoff_freq = 1000  # Specify the cutoff frequency in Hz
order = 5  # Specify the filter order

# Apply high-pass filter
filtered_signal = apply_filter(y, cutoff_freq, sr, order)
filtered_signal2 = apply_filter(y2, cutoff_freq, sr2, order)
filtered_signal3 = apply_filter(y3, cutoff_freq, sr3, order)
filtered_signal4 = apply_filter(y4, cutoff_freq, sr4, order)

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
    
    return dtw_distance


    
# Compute DTW distance
distance = dtw_distance(filtered_signal4, filtered_signal)
print("DTW Distance:", distance)
    
# Plot original and filtered signals
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(y)
plt.title('Original Signal')

plt.subplot(2, 1, 2)
plt.plot(filtered_signal)
plt.title('Filtered Signal (High-pass)')
plt.tight_layout()
plt.savefig('Original-vs-trimmed.png')
plt.show()

