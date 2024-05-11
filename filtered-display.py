#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:13:34 2024

@author: andrelopez
"""

from scipy.signal import butter, filtfilt
import librosa
import matplotlib.pyplot as plt
import numpy as np

# Function to create a high-pass Butterworth filter
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

# Function to apply a filter to an audio signal
def apply_filter(data, cutoff_freq, fs, order=5):
    b, a = butter_lowpass(cutoff_freq, fs, order=order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Load audio signal
y, sr = librosa.load('Sample1-Metal.wav', sr=None)

# Define high-pass filter parameters
cutoff_freq = 1000  # Specify the cutoff frequency in Hz
order = 5  # Specify the filter order

# Apply high-pass filter
filtered_signal = apply_filter(y, cutoff_freq, sr, order)

# Create a time vector
time = np.arange(len(y)) / sr

# Plot original and filtered signals
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, y)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.xlim(0,2)

plt.subplot(2, 1, 2)
plt.plot(time, filtered_signal)
plt.title('Filtered Signal (High-pass)')
plt.xlabel('Time (s)')
plt.xlim(0,2)
plt.tight_layout()
plt.savefig('New-low-original-vs-trimmed.png')
plt.show()
