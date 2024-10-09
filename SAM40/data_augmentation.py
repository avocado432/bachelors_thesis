"""
Implementation of the bachelor thesis:
Electroencephalogram (EEG) and machine learning-based classification of various stages of mental stress
Author: Tereza Lapčíková (xlapci03)
File: data_augmentation.py
"""

import os
import numpy as np
from scipy.io import loadmat, savemat
from scipy.ndimage import shift

def augment_signal_shift_noise(signal):
    shifted_signal = shift(signal, np.random.randint(-50, 50)) #random shift
    noisy_signal = shifted_signal + np.random.normal(0, 0.05, len(shifted_signal)) #gaussian noise
    return noisy_signal

def augment_signal_shift(signal):
    shifted_signal = shift(signal, np.random.randint(-50, 50))
    return shifted_signal

def augment_signal_noise(signal):
    noisy_signal = signal + np.random.normal(0, 0.05, len(signal))
    return noisy_signal

def augment_signal_scaling(signal):
    scale_factor = np.random.uniform(0.8, 1.2)
    return signal*scale_factor

file_list = [f for f in os.listdir('./filtered_data') if f.endswith('.mat')]

funcs = [augment_signal_shift_noise, augment_signal_shift, augment_signal_noise, augment_signal_scaling]

cnt = 0
for func in funcs:
    cnt += 1
    for file_name in file_list:
        if file_name[:9] == "augmented":
            continue
        data = loadmat(f'./filtered_data/{file_name}')
        clean_data = data['Clean_data']
        augmented_data = np.apply_along_axis(func, 1, clean_data)
        savemat(f'./filtered_data/augmented{cnt}_{file_name}', {'Clean_data': augmented_data})

#*********************** end of the data_augmentation.py file **************************