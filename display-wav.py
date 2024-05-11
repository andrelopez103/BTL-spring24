#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:21:22 2024

@author: andrelopez
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt

def display_waveform(wav_file):
    # Load the WAV file
    signal, sr = librosa.load(wav_file, sr=None)
    
    # Calculate the duration of the audio in seconds
    duration = librosa.get_duration(y=signal, sr=sr)
    
    # Compute the time axis
    time = np.linspace(0, duration, len(signal))
    
    # Plot the waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal, linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Waveform Plot")
    plt.xlim(4.044, 4.08)
    plt.show()

# Example usage
if __name__ == "__main__":
    wav_file = "New-table-metal.wav"
    display_waveform(wav_file)
