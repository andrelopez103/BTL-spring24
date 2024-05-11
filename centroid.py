#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 02:58:40 2024

@author: andrelopez
"""

import librosa
import matplotlib.pyplot as plt
import numpy as np

# Function to compute spectral centroid
def compute_spectral_centroid(audio_file, start_time, duration):
    y, sr = librosa.load(audio_file, offset=start_time, duration=duration, sr=None)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]  # Take only the first row (as it's 1D)
    return spectral_centroid

# Function to compare spectral centroids using Euclidean distance
def compare_spectral_centroids(centroid1, centroid2):
    # Ensure both vectors have the same length
    min_length = min(len(centroid1), len(centroid2))
    centroid1 = centroid1[:min_length]
    centroid2 = centroid2[:min_length]
    
    # Calculate Euclidean distance
    distance = np.linalg.norm(centroid1 - centroid2)
    return distance

# Example usage
audio_file = 'Clean-signal.mp3'

# Compute spectral centroids for different segments
centroid1 = compute_spectral_centroid(audio_file, 0, 5)
centroid2 = compute_spectral_centroid(audio_file, 0, 4)

# Compare spectral centroids
distance = compare_spectral_centroids(centroid1, centroid2)
print("Euclidean distance between spectral centroids:", distance)

# Compute spectral centroid
#spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]


