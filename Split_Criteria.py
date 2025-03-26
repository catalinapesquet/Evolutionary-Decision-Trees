# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 11:17:23 2025

@author: Catalina
"""

# Split criterion
import numpy as np
from scipy.stats import chi2_contingency
from scipy.stats import hypergeom
from math import *

# Gene 0: Information Gain 
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

# Gene 1: Gini
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

# Gene 3: G Statistic 
def g_stat(y, X_column, threshold):
    left_indices = X_column <= threshold
    right_indices = X_column > threshold

    left_prob = np.bincount(y[left_indices]) / len(y[left_indices])
    right_prob = np.bincount(y[right_indices]) / len(y[right_indices])

    G = 2 * np.sum(left_prob * np.log(left_prob / np.mean(left_prob))) + \
        2 * np.sum(right_prob * np.log(right_prob / np.mean(right_prob)))
    return G

# Gene 4: Mantaras
def mantaras(y, X_column, threshold):
    left_indices = X_column <= threshold
    right_indices = X_column > threshold
    
    left_counts = np.bincount(y[left_indices], minlength=np.max(y)+1)
    right_counts = np.bincount(y[right_indices], minlength=np.max(y)+1)
    observed = np.array([left_counts, right_counts])# contengency table, how many times each class appears in subgroups after split

    # Margin frequencies
    total_left = np.sum(observed[0])
    total_right = np.sum(observed[1])
    total = np.sum(observed)

    # expected frequencies 
    expected_left = np.outer(np.sum(observed, axis=1), np.sum(observed, axis=0)) / total
    expected_right = expected_left 

    # Mutual info for left group
    P_left = observed[0] / total_left 
    P_left = np.clip(P_left, 1e-10, 1)
    P_left_marginal = np.sum(P_left)
    MI_left = np.sum(P_left * np.log(P_left / (P_left_marginal)))  # handle division by zero

    # Mutual info for right group
    P_right = observed[1] / total_right 
    P_right = np.clip(P_right, 1e-10, 1)
    P_right_marginal = np.sum(P_right)
    MI_right = np.sum(P_right * np.log(P_right / (P_right_marginal)))  # handle division by zero
    distance = MI_left + MI_right

    return distance

# Gene 5: Hypergeometric Distribution

def hg_distribution(y, X_column, threshold):
    left_indices = X_column <= threshold
    right_indices = X_column > threshold
    
    left_counts = np.bincount(y[left_indices], minlength=np.max(y) + 1)
    right_counts = np.bincount(y[right_indices], minlength=np.max(y) + 1)
    
    total_left = np.sum(left_counts)
    total_right = np.sum(right_counts)
    total = total_left + total_right
    
    gains = []
    
    for class_label in range(np.max(y) + 1):
        N = np.sum(y == class_label)
        k = left_counts[class_label]
        p0 = hypergeom.cdf(k, total, N, total_left + total_right)
        gain = 1 - p0
        gains.append(gain)
    
    average_gain = np.mean(gains)
    
    return average_gain

def hg_distribution0(y, X_column, threshold):
    C = len(np.unique(y)) # number of classes 
    V = 2 # number of subgroup
    n_c = np.bincount(y) # number of appearance of each class
    N = len(X_column)
    
    left_indices = X_column <= threshold
    right_indices = X_column > threshold
    
    m_v = np.bincount(y[left_indices], minlength=np.max(y) + 1) + np.bincount(y[right_indices], minlength=np.max(y) + 1)
    f_cv = ([] for c in range(C))
    for c in range(C):
        for v in range(V):
            f_cv[C].append(m_v[v][c])
    p1 = 1 # first product
    p2 = 1 # second product
    for c in range(C):
        p1 = p1*factorial(n_c[c])
    p1/=factorial(N)
    for v in range(V):
        p3 = 1
        for c in range(C):
            p3 = p3*factorial(f_cv[c][v])
        p2 = p2*(factorial(m_v[v])/p3)
    P0 = p1*p2
    gain = 1 - P0
    return gain

# Gene 8: Chi Square
def chi_square(y, X_column, threshold):
    left_indices = X_column <= threshold
    right_indices = X_column > threshold
    
    # counts classes occurences 
    left_counts = np.bincount(y[left_indices], minlength=np.max(y) + 1)
    right_counts = np.bincount(y[right_indices], minlength=np.max(y) + 1)
    observed = np.array([left_counts, right_counts]) # contengency table, how many times each class appears in subgroups after split
    observed = np.maximum(observed, 1e-6)  # Replace 0 by small values to manage calculation problems 

    # calculate expected frequencies 
    expected = np.outer(np.sum(observed, axis=1), np.sum(observed, axis=0)) / np.sum(observed)
    chi2_stat, p_value, dof, expected = chi2_contingency(observed, correction=False) # calculates chi-quare on contingency table
    return chi2_stat

