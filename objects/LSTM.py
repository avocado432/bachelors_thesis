"""
Implementation of the bachelor thesis:
Electroencephalogram (EEG) and machine learning-based classification of various stages of mental stress
Author: Tereza Lapčíková (xlapci03)
File: LSTM.py
"""

# Imports
import torch
from torch import nn
import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

class LSTMModel(nn.Module):
    """
    @Brief: Class representing LSTM based classifier.
    @cite: Code inspired by https://wandb.ai/sauravmaheshkar/LSTM-PyTorch/reports/Using-LSTM-in-PyTorch-A-Tutorial-With-Examples--VmlldzoxMDA2NTA5
    """
    
    def __init__(self, input_dim : int, hidden_dim : int, layer_dim : int, output_dim : int, device : str, learning_rate : int, dropout : int):
        super(LSTMModel, self).__init__()
        # Initialize attributes
        self.learning_rate = learning_rate
        self.device = device
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.input_dim = input_dim
        # Loss function
        self.loss_function = nn.CrossEntropyLoss()
        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        #self.linear_first = nn.Linear(hidden_dim, 20)
        #self.sigmoid = nn.Sigmoid()
        #self.relu = nn.ReLU()
        self.linear_final = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, data.size(0), self.hidden_dim).to(self.device)
        # Initialize cell state with zeros
        c0 = torch.zeros(self.layer_dim, data.size(0), self.hidden_dim).to(self.device)
        lstm_out, (hn, cn) = self.lstm(data, (h0.detach(), c0.detach()))
        first_dropout = self.dropout(lstm_out[:, -1, :])
        #linear_first_out = self.linear_first(first_dropout)
        #sigmoid_out = self.sigmoid(linear_first_out)
        #second_dropout = self.dropout(sigmoid_out)
        #relu_out = self.relu(second_dropout)
        out = self.linear_final(first_dropout)
        return out
    
    def train(self, num_epochs : int, optimizer, scheduler, train_loader):
        training_loss = []
        for epoch in range(num_epochs):
            for batch, (data, labels) in enumerate(train_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)
                outputs = self.forward(data) # Model makes prediction
                loss = self.loss_function(outputs, labels.long()) # Loss function compares the model's prediction to the true labels
                optimizer.zero_grad() # Set existing gradients to zero
                loss.backward() # Compute the gradient of the loss with respect to each model parameter - model learns from its mistakes
                optimizer.step() # Model parameters are updated, optimizer uses computed gradients
                # Calculate training loss
                if (epoch % 20) == 0:
                    print(f'Epoch: {epoch + 1}/{num_epochs}\t Training loss: {loss.item():.4f}', end="")
                    print(" Learning rate: ", end = " ")
                    for param_group in optimizer.param_groups:
                        self.learning_rate = param_group['lr']
                    print(self.learning_rate)
            scheduler.step()
            #training_loss = training_loss.extend(loss.item())
            #x_axis = np.arange(1, num_epochs, 1)
            #plt.plot()

    def test(self, data_loader):
        with torch.no_grad():
            total_predicted = []
            total_labels = []
            progress = tqdm.tqdm(data_loader, total=len(data_loader))
            # Iterate through test dataset
            for data, labels in progress:
                labels = labels.to(self.device)
                total_labels.extend(labels)
                outputs = self.forward(data)
                predicted = torch.argmax(outputs, 1)
                total_predicted.extend(predicted)
            print(classification_report(total_labels, total_predicted))
            print(confusion_matrix(total_labels, total_predicted))
            return classification_report(total_labels, total_predicted, output_dict=True)['accuracy']

#*********************** end of the LSTM.py file **************************