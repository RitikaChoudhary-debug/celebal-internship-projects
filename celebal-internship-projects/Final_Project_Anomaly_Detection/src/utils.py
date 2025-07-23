# src/utils.py

import os
import pickle
import gc
from sklearn.metrics import roc_auc_score

def check_and_load_cache(cache_paths):
    all_files_exist = True
    for key, path in cache_paths.items():
        if not os.path.exists(path):
            all_files_exist = False
            break
    if all_files_exist:
        X_train_cached = pickle.load(open(cache_paths['X_train'], 'rb'))
        X_test_cached = pickle.load(open(cache_paths['X_test'], 'rb'))
        y_train_binary_cached = pickle.load(open(cache_paths['y_train_binary'], 'rb'))
        y_test_binary_cached = pickle.load(open(cache_paths['y_test_binary'], 'rb'))
        return X_train_cached, X_test_cached, y_train_binary_cached, y_test_binary_cached
    else:
        return None

def iso_forest_roc_auc_scorer(estimator, X, y):
    """
    Custom Scorer for Isolation Forest ROC AUC.
    Isolation Forest's decision_function returns LOWER scores for anomalies (positive class).
    This scorer negates the decision_function output to align with roc_auc_score's expectation,
    ensuring correct ROC AUC calculation.
    """
    anomaly_scores = estimator.decision_function(X)
    return roc_auc_score(y, -anomaly_scores)

def clean_memory(*args):
    """
    General utility function to delete variables and force garbage collection.
    """
    for arg in args:
        del arg
    gc.collect()