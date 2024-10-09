"""
Implementation of the bachelor thesis:
Electroencephalogram (EEG) and machine learning-based classification of various stages of mental stress
Author: Tereza Lapčíková (xlapci03)
File: feature_names.py
"""

#Imports
from enum import Enum

class ExtractedFeaturesNames(Enum):
    """
    Class enumerating names of the functions from the mne library to be used for given feature extraction.
    """
    approximate_entropy = "app_entropy"
    sampling_entropy = "samp_entropy"
    spectral_entropy = "spect_entropy"
    svd_entropy = "svd_entropy"
    mean = "mean"
    variance = "variance"
    standard_deviation = "std"
    peak_to_peak_amplitude = "ptp_amp"
    skewness = "skewness"
    kurtosis = "kurtosis"
    zero_crossings = "zero_crossings"
    hurst_exponent = "hurst_exponent"
    hjorth_mobility = "hjorth_mobility_spect"
    hjorth_complexity = "hjorth_complexity_spect"
    band_energy = "energy_freq_bands"
    phase_locking_value = "phase_lock_val"
    psd = "pow_freq_bands"

#*********************** end of the feature_names.py file **************************