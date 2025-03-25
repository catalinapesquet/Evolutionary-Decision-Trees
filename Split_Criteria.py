# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 11:17:23 2025

@author: Catalina
"""

# Split criterion
import numpy as np


def gini(y):
    # get the unique class and their counts
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    # calculate gini impurity: 1 - sum of squared probabilities
    return 1 - np.sum(probabilities ** 2)

def _gini_gain(y, X_column, threshold):
    left_indices = X_column <= threshold
    right_indices = X_column > threshold
    # check if one group is empty
    if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
        return 0
    # calculate proportion of data in left group
    p = float(len(y[left_indices])) / len(y)
    return gini(y) - p * gini(y[left_indices]) - (1 - p) * gini(y[right_indices])

def entropy(y):
    # get the unique class and their counts
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    # Calculate entropy: -sum of p * log2(p)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))

def information_gain(y, left_indices, right_indices):
    # proportion of samples in the left split 
    p = float(len(left_indices)) / (len(left_indices) + len(right_indices))
    # return entropy reduction
    return entropy(y) - p * entropy(y[left_indices]) - (1 - p) * entropy(y[right_indices])
