"""
Implementation of the bachelor thesis:
Electroencephalogram (EEG) and machine learning-based classification of various stages of mental stress
Author: Tereza Lapčíková (xlapci03)
File: activities.py
"""

#Imports
from enum import Enum

class Activities(Enum):
    """
    Brief: Enumerator of the SAM40 test activities.
    """
    arithmetic = "Arithmetic"
    mirror = "Mirror_image"
    stroop = "Stroop"
    relax = "Relax"

#*********************** end of the activities.py file **************************