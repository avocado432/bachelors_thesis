"""
Implementation of the bachelor thesis:
Electroencephalogram (EEG) and machine learning-based classification of various stages of mental stress
Author: Tereza Lapčíková (xlapci03)
File: feature_extraction.py
"""

# Imports
from objects.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mne_features.feature_extraction import FeatureExtractor
from mne.viz import plot_topomap
import matplotlib.pyplot as plt
import imageio
from mne import filter
from objects.freq_bands import BandFrequency
import matplotlib.pyplot as plt
import mne.io as mne_i
import glob
from natsort import natsorted
import scipy.io as sp

class FeatureExtraction:
    """
    Class representing and implementing the extraction of features.
    """    
    def __init__(self) -> None:
        pass
            
    def filter_frequency_bands(self, data, sampling_frequency):
        alpha_band = filter.filter_data(data, sfreq=sampling_frequency, l_freq=BandFrequency.alpha.value[0], h_freq=BandFrequency.alpha.value[1])
        beta_band = filter.filter_data(data, sfreq=sampling_frequency, l_freq=BandFrequency.beta.value[0], h_freq=BandFrequency.beta.value[1])
        gamma_band = filter.filter_data(data, sfreq=sampling_frequency, l_freq=BandFrequency.gamma.value[0], h_freq=BandFrequency.gamma.value[1])
        delta_band = filter.filter_data(data, sfreq=sampling_frequency, l_freq=BandFrequency.delta.value[0], h_freq=BandFrequency.delta.value[1])
        theta_band = filter.filter_data(data, sfreq=sampling_frequency, l_freq=BandFrequency.theta.value[0], h_freq=BandFrequency.theta.value[1])
        return np.array((alpha_band, beta_band, gamma_band, delta_band, theta_band))

    def extractor(self, dataset : Dataset, extracted_features_func):
        extracted_feature = []
        for trial in range(len(dataset.data)):
            extracted_feature.append([])
            feature = FeatureExtractor(sfreq=dataset.sampling_frequency, selected_funcs=extracted_features_func)
            result = feature.fit_transform(dataset.data[trial])
            extracted_feature[trial].extend(result)
        extracted_feature = np.array(extracted_feature)
        extracted_feature = np.reshape(extracted_feature, (extracted_feature.shape[0], extracted_feature.shape[1]*extracted_feature.shape[2]))
        return extracted_feature

    def plot_psd(self, dataset : Dataset, file_name : str, lobe_info, band, state):
        file = natsorted(glob.glob(file_name))
        loaded_file = sp.loadmat(file[0])
        raw = mne_i.RawArray(loaded_file['Clean_data'], dataset.info)
        raw_channels = raw.pick_channels(lobe_info['channels'])
        psd = raw_channels.compute_psd(method='welch', fmin=band.value[0], fmax=band.value[1], tmin=0, tmax=50)
        plt.savefig('./beta_'+lobe_info['name']+"_"+state+'.png')
        plt.close()

    def save_trial_topomaps(self, psd, file_num : int, dataset : Dataset):
        psd = np.reshape(psd, (psd.shape[0], dataset.trial_time, 5, 32))
        for time_window in range(dataset.trial_time):
            plot_topomap(psd[file_num][time_window][2], dataset.info, names=dataset.channel_names, sphere=0.099, ch_type='eeg', outlines='head', cmap=plt.get_cmap('jet'), show=False, contours=6, size=10, sensors=True)
            plt.savefig(f'./alpha_relaxed_{time_window}.png', transparent=False, facecolor="white")
            plt.close()
    
    def create_topomaps_gif(self, dataset : Dataset):
        frames = []
        for time_window in range(dataset.trial_time):
            image = imageio.v2.imread(f'./alpha_relaxed_{time_window}.png')
            frames.append(image)
        imageio.mimsave('./alpha_relaxed.gif', frames, fps=3, loop=0)

#************************ end of the feature_extraction.py file **************************