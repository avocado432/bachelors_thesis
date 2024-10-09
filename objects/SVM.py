"""
Implementation of the bachelor thesis:
Electroencephalogram (EEG) and machine learning-based classification of various stages of mental stress
Author: Tereza Lapčíková (xlapci03)
File: SVM.py
"""

from objects.classifier import Classifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

class SVM(Classifier):
    """
    @Brief: Class representing SVM classifier.
    @Cite: Inspired by https://www.kaggle.com/code/prashant111/svm-classifier-tutorial
    """
    def __init__(self, labels, features, feature_name : str, c_params : list, kernel_function : str):
        features_train, labels_train, features_test, labels_test, skf = super().__init__(labels, features)

        svm_parameters = {
            'C': c_params,
            'kernel': kernel_function
        }

        svm_clf = GridSearchCV(SVC(), svm_parameters, cv=skf, refit=True)
        svm_clf.fit(features_train, labels_train)

        print("VALIDATION RESULTS")
        for i in range(skf.n_splits):
            print(f"Fold {i+1}:")
            print(svm_clf.cv_results_['split{}_test_score'.format(i)].mean())

        svm_clf.fit(features_test, labels_test)
        predicted_labels = svm_clf.predict(features_test)

        self.print_test_results(feature_name, labels_test, predicted_labels)
        
#************************* end of the SVM.py file **************************