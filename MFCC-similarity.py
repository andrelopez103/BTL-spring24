#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 19:36:56 2024

@author: andrelopez
"""

import librosa
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean

# Load audio files
audio_path = 'Table.mp3'
audio_path2 = 'Metal.mp3'
audio_path3 = "Clean-signal.mp3"

y, sr = librosa.load(audio_path)
y2, sr2 = librosa.load(audio_path2)
y3, sr3 = librosa.load(audio_path3)

# Extract Mel-frequency cepstral coefficients (MFCCs)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfccs2 = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)
mfccs3 = librosa.feature.mfcc(y=y3, sr=sr3, n_mfcc=13)

# MFCCs mean calculations 
mfccs_mean1 = np.mean(mfccs, axis=1)
mfccs_mean2 = np.mean(mfccs2, axis=1)
mfccs_mean3 = np.mean(mfccs3, axis=1)

def calculate_similarity(mfccs_1, mfccs_2):
    # Calculate cosine similarity
    similarity = 1 - cosine(mfccs_1, mfccs_2)
    
    return similarity

def calculate_similarity2(mfccs_1, mfccs_2):
    # Calculate correlation coefficient
    correlation_coefficient, _ = pearsonr(mfccs_1, mfccs_2)
    
    return correlation_coefficient

def calculate_similarity3(mfccs_1, mfccs_2):
    # Calculate Euclidean distance
    euclidean_distance = euclidean(mfccs_1, mfccs_2)
    
    return 1 / (1 + euclidean_distance)  # Convert to similarity score

# Calculate similarity
similarity_score = calculate_similarity(mfccs_mean2, mfccs_mean3)

print(f"Similarity Score: {similarity_score}")

similarity_score2 = calculate_similarity2(mfccs_mean2, mfccs_mean3)

print(f"Similarity Score2: {similarity_score2}")

similarity_score3 = calculate_similarity3(mfccs_mean1, mfccs_mean2)

print(f"Similarity Score3: {similarity_score3}")

