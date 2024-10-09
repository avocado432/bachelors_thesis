"""
Implementation of the bachelor thesis:
Electroencephalogram (EEG) and machine learning-based classification of various stages of mental stress
Author: Tereza Lapčíková (xlapci03)
File: classifier.py
"""

#Imports
from sklearn.model_selection import train_test_split
import random
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

class Classifier:
    """
    Brief: Class representing a classifier.
    """
    def __init__(self, labels, features):

        features_train, features_test, labels_train, labels_test = train_test_split(
            features, labels, test_size=0.2, random_state=random.randint(0, 100)
        )

        skf = StratifiedKFold(n_splits=5)

        return features_train, labels_train, features_test, labels_test, skf

    def print_test_results(self, feature_name, labels_test, predicted_labels):
        print("TEST RESULTS")
        print(feature_name + "\n")
        print(metrics.classification_report(labels_test, predicted_labels))
        print(metrics.confusion_matrix(labels_test, predicted_labels))

#************************* end of the classifier.py file **************************