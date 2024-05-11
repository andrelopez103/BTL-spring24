#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:20:17 2024

@author: andrelopez
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import scipy

# Function to load and convert MP3 to PCM (WAV)
def load_and_convert_mp3(file_path,start_time, duration):
    y, sr = librosa.load(file_path, offset=start_time, duration=duration, sr=None)  # sr=None to keep the original sampling rate
    return y, sr

# Function to perform cross-correlation
def cross_correlation(signal1, signal2):
    return scipy.signal.correlate(signal1, signal2, mode='full')

# Load audio files
audio_path1 = 'Table.mp3'
audio_path2 = 'Metal.mp3'
audio_path3 = "Clean-signal.mp3"

# Load and convert MP3 files to PCM (WAV)
signal1, sr1 = load_and_convert_mp3(audio_path3, 0, 9)
signal2, sr2 = load_and_convert_mp3(audio_path3, 0, 5)

# Define the time range (in seconds) you want to display
start_time = 1  # Start time in seconds
end_time = 2    # End time in seconds

# Convert the time range to sample indices
start_index1 = int(start_time * sr1)
end_index1 = min(int(end_time * sr1), len(signal1))

start_index2 = int(start_time * sr2)
end_index2 = min(int(end_time * sr2), len(signal2))

# Check if the signals are empty after slicing
if start_index1 >= end_index1 or start_index2 >= end_index2:
    raise ValueError("Specified time range exceeds the length of the signals.")

# Perform cross-correlation using the sliced signals
cross_corr_result = cross_correlation(signal1[start_index1:end_index1], signal2[start_index2:end_index2])

# Calculate the lag time values in seconds
lag_times = np.arange(-len(signal1[start_index1:end_index1]) + 1, len(signal1[start_index1:end_index1])) / sr1

# Plot the cross-correlation result
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(lag_times, cross_corr_result)
plt.title('Cross-correlation Result')
plt.xlabel('Lag (seconds)')
plt.ylabel('Cross-correlation Value')
plt.grid(True)

# Plot PCM waves for the specified time range
time1 = np.arange(start_time, end_time, 1/sr1)
time2 = np.arange(start_time, end_time, 1/sr2)

plt.subplot(3, 1, 2)
plt.plot(time1, signal1[start_index1:end_index1])
plt.title('PCM Wave of Audio 1')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot(time2, signal2[start_index2:end_index2])
plt.title('PCM Wave of Audio 2')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()


