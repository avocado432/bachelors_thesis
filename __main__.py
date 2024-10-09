"""
Implementation of the bachelor thesis:
Electroencephalogram (EEG) and machine learning-based classification of various stages of mental stress
Author: Tereza Lapčíková (xlapci03)
File: __main__.py
"""

# Imports
from objects.dataset import Dataset
from objects.feature_extraction import FeatureExtraction
from objects.SVM import SVM
from objects.LSTM import LSTMModel
import torch
import torch.optim
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import StratifiedKFold
import random
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import SubsetRandomSampler
from objects.feature_names import ExtractedFeaturesNames
from objects.freq_bands import BandFrequency

def main():
    # Information about the SAM40 dataset.
    dataset_path = "SAM40/filtered_da/"
    labels_path = "SAM40/scales.xls"
    sampling_frequency = 128
    trial_time = 25
    channel_count = 32
    channel_names = ["Cz", "Fz", "Fp1", "F7", "F3", "FC1", "C3", "FC5", "FT9", "T7", "CP5", "CP1", "P3", "P7", "PO9", "O1", "Pz", "Oz", "O2", "PO10", "P8", "P4", "CP2", "CP6", "T8", "FT10", "FC6", "C4", "FC2", "F4", "F8", "Fp2"]

    # Load info for plot of the given lobe PSD (important for recognition of a stress phase)
    frontal_lobe = {"name" : "frontal",
                    "channels" : ["Fz", "Fp1", "F7", "F3", "FC1","FC5", "FT9", "FT10", "FC6", "FC2", "F4", "F8", "Fp2"]}
    parietal_lobe = {"name" : "parietal",
                     "channels" : ["Pz","P3","P4","P7","P8","CP1","CP2","CP5","CP6", "PO9", "PO10"]}
    # Channel placement in location of the central sulcus (between frontal and a parietal lobe)
    central_lobe = {"name" : "central",
                     "channels" : ["Cz", "C3", "C4"]} 
    temporal_lobe = {"name" : "temporal",
                     "channels" : ["T7", "T8", "FT9", "FT10"]}
    occipital_lobe = {"name" : "occipital",
                     "channels" : ["O1", "Oz", "O2"]}
    # Set parameters for plot of the power spectrum
    band = BandFrequency.beta
    state = "stressed" # Or "relaxed", only to set the name of the plot
    file_name = dataset_path + "Arithmetic_sub_20_trial1.mat"#"Relax_sub_1_trial1.mat"
    lobe = parietal_lobe

    # Set how to prepare the dataset
    interpolation_rate = 300
    derivative_order = 2

    # Prepare the dataset
    dataset = Dataset(sampling_frequency, trial_time, channel_count, channel_names)
    dataset.load_dataset(dataset_path, labels_path)
    dataset.segment_dataset()
    dataset.interpolation(interpolation_rate)
    dataset.derivatives(derivative_order)

    # Extract features
    extracted_features_list = [ExtractedFeaturesNames.phase_locking_value.value] # Select feature to be extracted using the enumerator
    feature_name = "PLV"
    extractor = FeatureExtraction()
    extracted_features = extractor.extractor(dataset, extracted_features_list)
    # Set number of folds for stratified k-fold
    num_folds = 5

    # Set SVM hyperparameters
    c_params = [0.1,1,10,100,1000]
    kernel_function = ['sigmoid', 'linear', 'poly', 'rbf']
    

    if(True):
        # SVM
        SVM(dataset.labels, extracted_features, feature_name, c_params, kernel_function)

    #LSTM
    # Set LSTM layer parameters
    output_dimension = 3 # This parameter is always 3, because we are classifying into three classes
    input_dimension = int(extracted_features.shape[1]/trial_time)
    lstm_layers_count = 3
    lstm_hidden_dimension = 40
    dropout = 0.3

    # Set LSTM hyperparameters
    batch_size = 50
    epoch_count = 300
    learning_rate = 0.0007
    lr_decrease = 0.999

    # Device Configuration
    device = torch.device('cpu') 
    # Split dataset to train and test part according to holdout validation method
    features_train, features_test, labels_train, labels_test = train_test_split(extracted_features, dataset.labels, test_size=0.2, random_state=random.randint(1, 100))
    labels_train = Variable(torch.Tensor(labels_train))
    labels_test = Variable(torch.Tensor(labels_test))
    features_train = Variable(torch.Tensor(features_train))
    features_test = Variable(torch.Tensor(features_test))
    features_train_final = torch.reshape(features_train, (features_train.shape[0], trial_time, features_train.shape[1]//trial_time))
    features_test_final = torch.reshape(features_test, (features_test.shape[0], trial_time, features_test.shape[1]//trial_time))
    train_data = TensorDataset(features_train_final, labels_train)
    test_data = TensorDataset(features_test_final, labels_test)

    # Stratified K-fold cross-validation
    if len(features_train_final) < num_folds:
        k_folds = len(features_train_final)
    else:
        k_folds = num_folds # k% of the training data is for validation
    skf = StratifiedKFold(n_splits=k_folds, shuffle=False)
    skf.get_n_splits(features_train_final, labels_train)
    best_model_accuracy = 0
    best_model = None
    for fold, (train_index, validation_index) in enumerate(skf.split(features_train_final, labels_train)):
        model = LSTMModel(input_dimension, lstm_hidden_dimension, lstm_layers_count, output_dimension, device, learning_rate, dropout)
        model.to(device)
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Learning rate scheduler
        scheduler = ExponentialLR(optimizer, gamma=lr_decrease)
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, sampler=SubsetRandomSampler(train_index))
        validation_loader = DataLoader(dataset=train_data, batch_size=batch_size, sampler=SubsetRandomSampler(validation_index))
        print(f"Fold {fold+1}")
        print("----------")
        model.train(epoch_count, optimizer, scheduler, train_loader)
        print(f"Fold {fold+1} test")
        accuracy = model.test(validation_loader)
        # Selection of the best model
        if accuracy > best_model_accuracy:
            best_model_accuracy = accuracy
            best_model = model

    # Test the model
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    print("LSTM FINAL TEST")
    best_model.test(test_loader)
    print("\nHyperparameters: ")
    print(f"Extracted feature: {feature_name}")
    print(f"Derivative order: {derivative_order}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {epoch_count}")
    print(f"Learning rate: {learning_rate}")
    print(f"Learning rate decrease: {str(lr_decrease)}\n")
    print(f"LSTM hidden dimension: {lstm_hidden_dimension}")
    print(f"LSTM layer dimension: {lstm_layers_count}")
    print(f"Dropout: {dropout}")

if __name__ == '__main__':
    main()

#*********************** end of the __main__.py file **************************