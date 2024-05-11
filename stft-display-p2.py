#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 12:08:20 2024

@author: andrelopez
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the audio file with a sampling rate of 44.1 kHz
audio_file = "Chicken_1.wav"
y, sr = librosa.load(audio_file, sr=44100)

# Define the time range of interest
start_time = 0.4
end_time = 1.0

# Compute the frame indices corresponding to the time range
start_frame = int(start_time * sr)
end_frame = int(end_time * sr)

# Extract the audio segment within the time range
y_segment = y[start_frame:end_frame]

# Adjust FFT parameters for higher time resolution
n_fft = 2048  # Increase FFT size for finer time resolution
hop_length = 128  # Decrease hop length for finer time resolution

# Compute STFT
D = librosa.stft(y_segment, n_fft=n_fft, hop_length=hop_length)

# Plot STFT spectrogram with higher time resolution
plt.figure()
librosa.display.specshow(librosa.amplitude_to_db(abs(D), ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', fmax=20000)
plt.colorbar(format='%+2.0f dB')
plt.title('Chicken Signal')
plt.xlabel('Time (s)', fontsize=12)  # Increase x-axis label font size
plt.ylabel('Frequency (Hz)', fontsize=12)  # Increase y-axis label font size
plt.tight_layout()
plt.savefig("chicken-png")
plt.show()

