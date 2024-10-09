# Electroencephalogram (EEG) and machine learning-based classification of various stages of mental stress

## Author
Tereza Lapčíková

## Description
These scripts provide implementation of a ternary classifier of varying mental stress stages. This final solution contains one SVM model and a three-layer LSTM model. This solution is based on an object oriented programming paradigm.

Hyperparameters, extracted features and a way of data preparation can be easily changed in script \_\_main\_\_.py. Classes implemented within this solution are in directory objects/. In directory SAM40/ is scales.xls file with dataset labels and data_augmentation.py script used for data augmentation. In directory SAM40/filtered_data/ are the SAM40 dataset .mat files. Directory docs/ contains .tex and other resources of the thesis.pdf document. Document thesis.pdf is in the root directory same as \_\_main\_\_.py script, Makefile and this README.md.

This script creates results.txt file with printed results. 

## Installation

Before running the Python script, ensure you have the necessary packages installed. You can install them using following pip command:

```bash
make install
```

## Running the Scripts

When using the script for the first time, the augmentation of the dataset has to be done, which can be achieved with following command:

```bash
make augmentation
```
Classification scripts can be run using the following command:
```bash
make run
```