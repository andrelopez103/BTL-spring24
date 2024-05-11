#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:40:33 2024

@author: andrelopez
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa

def plot_dft(audio_file, start_time, duration, max_frequency=1000):
    # Load audio file
    y, sr = librosa.load(audio_file, offset=start_time, duration=duration, sr=None)

    # Compute the Discrete Fourier Transform (DFT) using FFT
    dft = np.fft.fft(y)

    # Compute the frequency axis
    N = len(y)  # Length of the signal
    freq = np.fft.fftfreq(N, d=1/sr)
    

    # Find the index corresponding to the maximum frequency
    max_freq_index = np.argmax(freq > max_frequency)
    print(max(np.abs(dft)[:max_freq_index]))
    # Plot the magnitude spectrum up to the maximum frequency
    plt.figure(figsize=(10, 6))
    plt.plot(freq[:max_freq_index], np.abs(dft)[:max_freq_index])  # Plot only frequencies up to max_frequency
    plt.title('Magnitude Spectrum (DFT)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()
    


# Example usage
audio_file1 = 'Table.mp3'
audio_file2 = 'Metal.mp3'
audio_file3 = 'Clean-signal.mp3'


plot_dft(audio_file3, 0, 5)
plot_dft(audio_file3, 5, 3.26)
plot_dft(audio_file3, 0, 9)
