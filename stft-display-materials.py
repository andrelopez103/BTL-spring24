#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:27:55 2024

@author: andrelopez
"""
import librosa
import librosa.display
import matplotlib.pyplot as plt

# List of audio files
audio_files = ["Clean-signal.wav"]

# Loop over each audio file
for file in audio_files:
    # Load the audio file
    y, sr = librosa.load(file)

    # Compute STFT
    n_fft = 2048  # length of the FFT window
    hop_length = 512  # number of samples between successive STFT columns
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # Convert magnitude spectrogram to dB scale
    spectrogram = librosa.amplitude_to_db(abs(stft))

    # Plot spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(stft, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Wood and Metal Signal")
    plt.xlabel('Time (s)', fontsize=18)  # Increase x-axis label font size
    plt.ylabel('Frequency (Hz)', fontsize=18)  # Increase y-axis label font size
   
    plt.show()

"""
    # Save the plot as a PNG file
    plt.savefig(f'spectrogram_{file[:-4]}.png')  # Save each spectrogram with a filename based on the audio file

    plt.close()  # Close the plot to release memory
"""
#plt.title(f'Spectrogram - {file}')

