"""
Implementation of the bachelor thesis:
Electroencephalogram (EEG) and machine learning-based classification of various stages of mental stress
Author: Tereza Lapčíková (xlapci03)
File: freq_bands.py
"""

#Imports
from enum import Enum

class BandFrequency(Enum):
    """
    Class enumerating pairs of frequencies that border a given frequency band (in Hz).
    """
    alpha = (8, 13)
    beta = (13, 30)
    gamma = (30, 60)
    delta = (0.5, 4)
    theta = (4, 8)

#*********************** end of the freq_bands.py file **************************