"""
Implementation of the bachelor thesis:
Electroencephalogram (EEG) and machine learning-based classification of various stages of mental stress
Author: Tereza Lapčíková (xlapci03)
File: data_loader.py
"""

# Imports
import glob
import scipy.io as sp
import numpy as np
import pandas
from natsort import natsorted
from mne import create_info
from scipy import interpolate
import mne.io
import matplotlib.pyplot as plt

class Dataset:
    """
    Brief: Class representing a dataset.
    """
    def __init__(self, sampling_frequency : int, trial_time : int, channel_count : int, channel_names : list):
        self.sampling_frequency = sampling_frequency
        self.trial_time = trial_time
        self.size = 3200//self.trial_time
        self.channel_count = channel_count
        self.channel_names = channel_names
        self.info = create_info(ch_names=channel_names,sfreq=sampling_frequency, ch_types='eeg')
        self.info.set_montage('standard_1020')

    def load_dataset(self, data_path : str, labels_path : str):
        self.labels = []
        self.data = []
        labels_file = pandas.read_excel(labels_path, skiprows = [0])
        data_files = natsorted(glob.glob(data_path + "*.mat"))
        file_counter = 0
        for file in data_files:
            before, match, after = file.partition("_sub_")
            bef, match, activity = before.partition(data_path)
            if activity[:9].lower() == "augmented":
                bef, match, activity = activity.partition("_")
            subject, match, trial_mat = after.partition("_trial")
            trial_number, match, af = trial_mat.partition(".mat")
            label = self.load_label(labels_file, activity, int(subject), int(trial_number))
            self.labels.append(self.evaluate_stress_level(label))
            self.data.append(sp.loadmat(file)['Clean_data'])
            file_counter += 1
        self.labels = np.array(self.labels)

    def evaluate_stress_level(self, label : int):
        if label > 2 and label < 6:
            return 1
        elif label >=6:
            return 2
        else:
            return 0

    def load_label(self, labels_file, activity : str, subject : int, trial : int):
        column_name = activity
        if(trial > 1):
            column_name += "." + str(trial - 1)
        column = labels_file[column_name]
        return int(column[subject-1])
    
    def segment_dataset(self):
        segmented_data = np.zeros((len(self.data), self.trial_time, self.channel_count, self.size))
        recording_count = 0
        for recording in self.data:
            for time_window in range(self.trial_time):
                start_time = time_window * self.size # Start of the time window.
                end_time = (time_window + 1) * self.size # End of the time window.
                segmented_data[recording_count][time_window] = self.data[recording_count][:, start_time:end_time]
            recording_count += 1
        self.data = segmented_data

    def interpolation(self, interpolation_rate : int):
        """
        Interpolate the windowed data using spline.
        """
        period = 1/(self.sampling_frequency)
        time_stamps = np.arange(period, (1+period), period)
        self.sampling_frequency = interpolation_rate
        time_axis = np.zeros((*self.data.shape[:3], interpolation_rate))
        interpolated_data = np.zeros((*self.data.shape[:3], interpolation_rate))
        for trial in range(len(self.data)):
            for window in range(int(self.trial_time)):
                for electrode in range(self.channel_count):
                    time = time_stamps*(window+1)
                    time_space = np.linspace(time[0], time[-1], interpolation_rate)
                    B_spline_coefficients = interpolate.make_interp_spline(time, self.data[trial, window, electrode])
                    interpolated_data[trial, window, electrode] = B_spline_coefficients(time_space)
                    time_axis[trial, window, electrode] = time_space
        self.data = interpolated_data
        self.interpolated_time_axis = time_axis

    def derivatives(self, derivative_order : int):
        """
        Calculate the given order time derivative of the data.
        """
        for trial in range(len(self.data)):
            for window in range(self.trial_time):
                for electrode in range(self.channel_count):
                    derivative = self.data[trial, window, electrode]
                    for order in range(derivative_order):
                        derivative = np.gradient(derivative, self.interpolated_time_axis[trial, window, electrode])
                    self.data[trial, window, electrode] = derivative

    def plot_raw_eeg(self, file_path : str):
        file = glob.glob(file_path)
        raw = []
        loaded_file = sp.loadmat(file[0])
        raw = mne.io.RawArray(loaded_file['Clean_data'], self.info)
        raw.plot(n_channels = self.channel_count, scalings = 40, title = 'Clean EEG data - stressed', block = True)
        plt.savefig('./plots/stressed_sub20_arithmetic1/raw_eeg.png')
        plt.close()

#*********************** end of the data_loader.py file **************************