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
from scipy.special import gammaln
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Gene 0: Information Gain 
def entropy(y):
    # get the unique class and their counts
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    # Calculate entropy: -sum of p * log2(p)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(y, left_indices, right_indices):
    # proportion of samples in the left split 
    p = float(len(left_indices)) / (len(left_indices) + len(right_indices))
    # return entropy reduction
    return entropy(y) - (p * entropy(y[left_indices]) + (1 - p) * entropy(y[right_indices]))

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

# Gene 2: G Statistic 
# def g_stat(y, X_column, threshold):

#     y = np.asarray(y).ravel().astype(int)
    
#     C = len(np.unique(y))  # Number of classes

#     n_c = np.bincount(y)  # Count of each class in the total group
#     N = len(X_column)  # Total number of elements

#     left_indices = X_column <= threshold
#     right_indices = X_column > threshold
#     m_v = np.array([np.sum(left_indices), np.sum(right_indices)])  # Number of elements in each group
#     f_cv = np.zeros((C, 2), dtype=int)  # Correct initialization of the f_cv matrix
#     for c in range(C):
#         f_cv[c, 0] = np.sum(y[left_indices] == c)  # Count of instances of class c in the left group
#         f_cv[c, 1] = np.sum(y[right_indices] == c)  # Count of instances of class c in the right group

#     # Using logarithms to avoid overflow in the numerator and denominator
#     log_numerator = np.sum(gammaln(n_c + 1)) + np.sum(gammaln(m_v + 1))
#     log_denominator = gammaln(N + 1) + np.sum(gammaln(f_cv + 1))
    
#     # If the denominator is infinity, return 1.0 (indicating a problem)
#     if np.isinf(log_denominator):
#         return 1.0
#     else:
#         log_ratio = np.clip(log_numerator - log_denominator, -700, 700)
#         P0 = np.exp(log_ratio) # Calculate the final probability
#         gain = 1 - P0  # Compute the gain
#         return gain
    
def g_stat(y, X_column, threshold):

    y = np.asarray(y).ravel().astype(int)
    classes = np.unique(y)
    C = len(classes)  # Number of classes
    V = 2 # split into two groups
    # print("👀 y dans g_stat:", y)
    # print("🧾 type:", type(y), "| shape:", np.shape(y), "| dtype:", np.array(y).dtype)
    # print("🔢 valeurs uniques:", np.unique(y))

    n_c = np.bincount(y)  # Count of each class in the total group
    N = len(X_column)  # Total number of elements

    left_indices = X_column <= threshold
    right_indices = X_column > threshold
    m_v = np.array([np.sum(left_indices), np.sum(right_indices)])  # Number of elements in each group
    f_cv = np.zeros((C, V), dtype=int)
    for i, c in enumerate(classes):
        f_cv[i, 0] = np.sum(y[left_indices] == c)
        f_cv[i, 1] = np.sum(y[right_indices] == c)

    # Using logarithms to avoid overflow in the numerator and denominator
    log_numerator = np.sum(gammaln(n_c + 1)) + np.sum(gammaln(m_v + 1))
    log_denominator = gammaln(N + 1) + np.sum(gammaln(f_cv + 1))
    
    # If the denominator is infinity, return 1.0 (indicating a problem)
    if np.isinf(log_denominator):
        return 1.0
    else:
        log_ratio = np.clip(log_numerator - log_denominator, -700, 700)
        P0 = np.exp(log_ratio) # Calculate the final probability
        gain = 1 - P0  # Compute the gain
        return gain
    
    
# Gene 3: Mantaras
def safe_log2(x):
    """ Computes log2, returning 0 for log2(0) """
    return np.log2(x, out=np.zeros_like(x), where=(x > 0))

def mantaras(y, X_column, threshold):
    # Ensure y is 1D array of integers
    y = np.array(y)
    if y.ndim > 1:
        y = y.ravel()
    if not np.issubdtype(y.dtype, np.integer):
        y = LabelEncoder().fit_transform(y)
        
    X_column = np.asarray(X_column).ravel()

    left_indices = X_column <= threshold
    right_indices = X_column > threshold

    left_counts = np.bincount(y[left_indices], minlength=np.max(y)+1)
    right_counts = np.bincount(y[right_indices], minlength=np.max(y)+1)
    observed = np.array([left_counts, right_counts])  # contingency table

    # Margin frequencies
    total_left = np.sum(observed[0])
    total_right = np.sum(observed[1])
    total = np.sum(observed)

    class_counts = np.bincount(y, minlength=np.max(y)+1)
    class_probabilities = class_counts / total
    
    pv_counts = np.array([total_left, total_right])
    pv_probabilities = pv_counts / total
    
    I_PC = -np.sum(class_probabilities * safe_log2(class_probabilities))
    I_PV = -np.sum(pv_probabilities * safe_log2(pv_probabilities))
    joint_probabilities = observed / total
    I_PC_PV = -np.sum(joint_probabilities * safe_log2(joint_probabilities))
    I_PC__PV = I_PC_PV - I_PV
    I_PV__PC = I_PC_PV - I_PC
    
    distance = I_PC__PV + I_PV__PC
    distance = distance / I_PC_PV # normalization
    gain = 1 - distance # to minimize
    
    return gain

# Gene 4: Hypergeometric Distribution
def hg_distribution(y, X_column, threshold):
    classes = np.unique(y)
    C = len(classes)  # Number of classes
    V = 2  # Binary split
    N = len(y)

    left_indices = X_column <= threshold
    right_indices = X_column > threshold
    if not left_indices.any() or not right_indices.any():
        return 0.0

    # f_cv: [classes, branches]
    f_cv = np.zeros((C, V), dtype=int)
    for i, c in enumerate(classes):
        f_cv[i, 0] = np.sum(y[left_indices] == c)
        f_cv[i, 1] = np.sum(y[right_indices] == c)
    
    # print("f_cv:", f_cv)

    n_c = f_cv.sum(axis=1) 
    m_v = f_cv.sum(axis=0)  

    log_numerator = np.sum(gammaln(n_c + 1)) + np.sum(gammaln(m_v + 1))
    # print("log_numerator: ", log_numerator)
    log_denominator = gammaln(N + 1) + np.sum(gammaln(f_cv + 1))
    # print("log_denominator: ", log_denominator)
    log_P0 = log_numerator - log_denominator
    log_P0 = np.clip(log_P0, -700, 700)
    # P0 = np.exp(log_P0) # original gain, but due to realy small values we prefer using lop_P0 directly

    gain = -log_P0 # à maximiser
    return gain
    
# Gene 5: Chv
def chv_criterion(y, X_column, threshold):
    if isinstance(X_column, pd.Series):
        X_column = X_column.values
    if isinstance(y, pd.Series):
        y = y.values

    left_indices = X_column <= threshold
    right_indices = X_column > threshold

    left = y[left_indices]
    right = y[right_indices]

    # Check if one group is empty
    if len(left) == 0 or len(right) == 0:
        return float('inf')

    left_indices = X_column <= threshold
    right_indices = X_column > threshold
    
    left = y[left_indices]
    right = y[right_indices]
    C = len(np.unique(y)) # number of classes 
    R = len(y)
    class_counts = pd.Series(y).value_counts().to_dict()
    
    CS1 = len(np.unique(left)) # number of classes in left group
    S1 = len(left) # number of instances in left group
    class_counts_left = pd.Series(left).value_counts().to_dict()
    
    # Statistiques pour la partition S2
    CS2 = len(np.unique(right))
    S2 = len(right)
    class_counts_right = pd.Series(right).value_counts().to_dict()
    
    # Calculate first term of gain
    term1 = 0
    if CS1 > 0 and C > 0 and R > 0: # check for potential division by zero
        sum_ratio_left = 0
        for c in class_counts_left:
            n_i_left = class_counts_left.get(c, 0)
            n_i_y = class_counts.get(c, 0)
            if n_i_y > 0:
                sum_ratio_left += n_i_left / n_i_y
        term1 = (S1 / R) * (CS1 / C) * sum_ratio_left

    
    # Calculate second term of gain 
    term2 = 0
    if CS2 > 0 and C > 0 and R > 0:
        sum_ratio_right = 0
        for class_label in class_counts_right:
            n_i_S2 = class_counts_right.get(class_label, 0)
            n_i_R = class_counts.get(class_label, 0)
            if n_i_R > 0:
                sum_ratio_right += n_i_S2 / n_i_R
        term2 = (S2 / R) * (CS2 / C) * sum_ratio_right
    
    # Calculate gain
    measure = term1 + term2
    measure = measure / C # to normalize
    gain = 1 - measure
    return gain

# Gene 6: DCSM
def dcsm(y, X_column, threshold):
    y = np.array(y)
    X_column = np.array(X_column)

    left_indices = X_column <= threshold
    right_indices = X_column > threshold
    left = y[left_indices]
    right = y[right_indices]
    N = len(y) # total number of instances in node
    C = np.unique(y) # number of classes distinctes présentes dans le nœud parent
    D_u = len(C)

    # Check if one group is empty
    if len(left) == 0 or len(right) == 0:
        return float('inf')

    # Calculate for left group
    N_left = len(left)
    C_left = np.unique(left) # number of distinc classes in left group
    D_left = len(C_left)
    if D_u > 0: # handle division by zero
        d_left = D_left / D_u
    else:
        d_left = 0

    left_sum = 0
    for c in C_left:
        N_c_left = np.sum(left == c)
        a_c_left = N_c_left / N_left
        left_sum += a_c_left * exp(d_left * (1 - a_c_left)**2)

    left_term = (N_left / N) * (D_left * exp(D_left)) * left_sum

    # Calculte for right group
    N_right = len(right)
    C_right = np.unique(right) # number of distinct classes in right group
    D_right = len(C_right)
    if D_u > 0: # handle division by zero
        d_right = D_right/ D_u
    else:
        d_right = 0

    right_sum = 0
    for c in C_right:
        N_c_right = np.sum(right == c)
        a_c_right = N_c_right / N_right
        right_sum += a_c_right * exp(d_right * (1 - a_c_right)**2)

    right_term = (N_right / N) * (D_right * exp(D_right)) * right_sum

    # calculate gain
    dcsm = left_term + right_term
    gain =  1000 - dcsm
    return gain

# Gene 7: Chi Square
def chi_square(y, X_column, threshold):
    # Ensure y is 1D array of integers
    y = np.array(y)
    if y.ndim > 1:
        y = y.ravel()
    if not np.issubdtype(y.dtype, np.integer):
        y = LabelEncoder().fit_transform(y)
    
    X_column = np.asarray(X_column).ravel()

    left_indices = X_column <= threshold
    right_indices = X_column > threshold

    # counts classes occurrences 
    left_counts = np.bincount(y[left_indices], minlength=np.max(y) + 1)
    right_counts = np.bincount(y[right_indices], minlength=np.max(y) + 1)
    observed = np.array([left_counts, right_counts])  # contingency table, how many times each class appears in subgroups after split
    observed = np.maximum(observed, 1e-6)  # Replace 0 by small values to manage calculation problems 

    # calculate expected frequencies 
    expected = np.outer(np.sum(observed, axis=1), np.sum(observed, axis=0)) / np.sum(observed)
    chi2_stat, p_value, dof, expected = chi2_contingency(observed, correction=False)  # calculates chi-square on contingency table

    return chi2_stat

# Gene 8: Mean Posterior Improvement 
def mpi(y, X_column, threshold):
    L_indices = X_column <= threshold
    R_indices = X_column > threshold
    # Number of individuals on each side
    n_L = np.sum(L_indices)
    n_R = np.sum(R_indices)
    n_tot = n_L + n_R
    # Proportion of individuals on each side
    if n_tot == 0:
        return 0.0
    pL = n_L / n_tot
    pR = 1 - pL
    # Unique classes
    classes = np.unique(y)

    # Calculate MPI
    mpi_sum = 0.0
    for c in classes:
        # Proportion of class c in the parent node
        n_k = np.sum(y == c)
        pk = n_k / n_tot if n_tot > 0 else 0.0

        # Proportion of class c in the left group
        nL_k = np.sum(y[L_indices] == c)
        p_tLk = nL_k / n_k if n_k > 0 else 0.0

        # Proportion of class c in the right group
        nR_k = np.sum(y[R_indices] == c)
        p_tRk = nR_k / n_k if n_k > 0 else 0.0

        mpi_sum += pk * p_tLk * p_tRk

    # Calculate MPI value
    mpi_value = pL * pR - mpi_sum
    return mpi_value
    
# Gene 9: ORT 
def ORT(y, X_column, threshold):
    L_indices = X_column <= threshold
    R_indices = X_column > threshold
    L = y[L_indices]
    R = y[R_indices]
    
    # Check if one group is empty
    if len(L) == 0 or len(R) == 0:
        return float('inf')
    
    # Unique classes
    classes = np.unique(y)
    # Calculate class vector for both sides
    V_L = np.array([np.sum(L == c) for c in classes])
    V_R = np.array([np.sum(R == c) for c in classes])
    # Calculate dot product of two vectors
    dot_product = np.dot(V_L, V_R)
    # Calculate magnitude of 
    magnitude_L = np.linalg.norm(V_L)
    magnitude_R = np.linalg.norm(V_R) 
    # Calculate cosine between vectors
    if magnitude_L == 0 or magnitude_R == 0:
        cosine = 0.0
    else:
        cosine = dot_product / (magnitude_L * magnitude_R) 
    
    # Calculate ORT
    ort_value = np.abs(1-cosine)
    
    return ort_value

# Gene 10: Twoing
def twoing(y, X_column, threshold):
    left_indices = X_column <= threshold
    right_indices = X_column > threshold
    left = y[left_indices]
    right = y[right_indices]

    # Check if one group is empty. If so, PL or PR will be 0, making the twoing criterion 0.
    if len(left) == 0 or len(right) == 0:
        return 0.0

    n_total = len(y)
    n_L = len(left)
    n_R = len(right)

    # Calculate PL and PR
    PL = n_L / n_total
    PR = n_R / n_total

    # Unique classes
    classes = np.unique(y)

    # Calculate Pj,L and Pj,R for each class and the sum of absolute differences
    diff_sum = 0
    for c in classes:
        Pj_L = np.sum(left == c) / n_L if n_L > 0 else 0
        Pj_R = np.sum(right == c) / n_R if n_R > 0 else 0
        diff_sum += np.abs(Pj_L - Pj_R)

    # Calculate the twoing criterion: PLPR * [Σⱼ |Pⱼ,L - Pⱼ,R|]²
    twoing_value = PL * PR * (diff_sum ** 2)

    return twoing_value

# Gene 11: CAIR
# def cai_after_split(y, X_column, threshold):
#     left_indices = X_column <= threshold
#     right_indices = X_column > threshold
#     # Check if one side is empty 
#     if len(left_indices) == 0 or len(right_indices) == 0:
#         return 0  

#     cai_value = information_gain(y, left_indices, right_indices)
#     return cai_value

# def redundancy_measure(y, X_column, threshold):
#     left_indices = X_column <= threshold
#     right_indices = X_column > threshold
    
#     left = y[left_indices]
#     right = y[right_indices]

#     n_y = len(y)
#     n_left = len(left)
#     n_right = len(right)

#     if len(left) == 0 or len(right) == 0:
#         return 0  

#     class_counts_parent = pd.Series(y).value_counts(normalize=True).to_dict()
#     class_counts_left = pd.Series(left).value_counts(normalize=True).to_dict()
#     class_counts_right = pd.Series(right).value_counts(normalize=True).to_dict()

#     inconsistency = 0
#     all_classes = set(class_counts_parent.keys()) | set(class_counts_left.keys()) | set(class_counts_right.keys())

#     for cls in all_classes:
#         prob_parent = class_counts_parent.get(cls, 0)
#         prob_left = class_counts_left.get(cls, 0)
#         prob_right = class_counts_right.get(cls, 0)

#         # Calculer la différence pondérée des probabilités par la taille des partitions
#         inconsistency += abs(prob_left - prob_parent) * (n_left / n_y) + abs(prob_right - prob_parent) * (n_right / n_y)

#     return inconsistency

# def cair(y, X_column, threshold):
#     if len(y) != len(X_column):
#         raise ValueError("y and X_column must have the same length")

#     left_indices = X_column <= threshold
#     right_indices = X_column > threshold

#     n_total = len(y)
#     n_left = np.sum(left_indices)
#     n_right = np.sum(right_indices)

#     if n_left == 0 or n_right == 0:
#         print("Warning: One split is empty. Returning a large value.") 
#         return np.inf 

#     gain = information_gain(y, left_indices, right_indices)

#     num_classes = np.max(y) + 1
#     left_counts = np.bincount(y[left_indices], minlength=num_classes)
#     right_counts = np.bincount(y[right_indices], minlength=num_classes)
#     observed = np.vstack([left_counts, right_counts])  

#     total = observed.sum()
#     expected = np.outer(observed.sum(axis=0), observed.sum(axis=1)) / total
#     expected = expected.T  # Ajuster pour être aligné avec observed

#     chi2_statistic = np.sum((observed - expected) ** 2 / (expected + 1e-9))

#     conditional_term = chi2_statistic / (2 * n_total)

#     r_ca = gain - conditional_term

#     return r_ca

def cair(y, X_column, threshold):
    # Convert inputs to numpy arrays for easier manipulation
    y = np.array(y)
    X_column = np.array(X_column)

    # Discretize the attribute column based on the threshold
    A_d = np.where(X_column <= threshold, 0, 1)

    # Calculate the probabilities and entropies
    classes, class_counts = np.unique(y, return_counts=True)
    P_c = class_counts / len(y)

    # Calculate H(C)
    H_C = -np.sum(P_c * np.log2(P_c + 1e-10))  # Adding a small value to avoid log(0)

    # Calculate joint probabilities P(c_i, a_{dj})
    P_ca = np.zeros((len(classes), 2))
    for i, c in enumerate(classes):
        for j in range(2):
            P_ca[i, j] = np.sum((y == c) & (A_d == j)) / len(y)

    # Calculate P(a_{dj})
    P_a = np.sum(P_ca, axis=0)

    # Calculate I(C; A_d)
    I_CA = 0
    for i in range(len(classes)):
        for j in range(2):
            if P_ca[i, j] > 0 and P_c[i] > 0 and P_a[j] > 0:
                I_CA += P_ca[i, j] * np.log2(P_ca[i, j] / (P_c[i] * P_a[j]))

    # Calculate CAIR
    R_CA = 1 - I_CA / H_C

    return 1-R_CA

# Gene 12: Gain Ratio
def gain_ratio(y, X_column, threshold):
    initial_entropy = entropy(y)

    left_indices = X_column <= threshold
    right_indices = X_column > threshold
    y_left = y[left_indices]
    y_right = y[right_indices]

    n_total = len(y)
    n_left = len(y_left)
    n_right = len(y_right)

    if n_left == 0 or n_right == 0:
        return 0  

    entropy_left = entropy(y_left)
    entropy_right = entropy(y_right)

    information_gain = initial_entropy - (n_left / n_total) * entropy_left - (n_right / n_total) * entropy_right

    split_info = - (n_left / n_total) * log2(n_left / n_total) - (n_right / n_total) * log2(n_right / n_total)

    if split_info == 0:
        return 0
    gain_ratio_value = information_gain / split_info

    return gain_ratio_value

class SplitCriterion:
    def __init__(self, criterion='gini'):
        self.criterion = criterion
    
    def calculate(self, y, X_column, threshold):
        if self.criterion == 'information_gain':
            return self._information_gain(y, X_column, threshold) 
        elif self.criterion == 'gini':
            return self._gini_gain(y, X_column, threshold)
        elif self.criterion == 'g_stat':
            return self._g_stat(y, X_column, threshold)
        elif self.criterion == 'mantaras':
            return self._mantaras(y, X_column, threshold)
        elif self.criterion == 'hg_distribution':
            return self._hg_distribution(y, X_column, threshold)
        elif self.criterion == 'chv_criterion':
            return self._chv_criterion(y, X_column, threshold)
        elif self.criterion == 'dcsm':
            return self._dscm(y, X_column, threshold)
        elif self.criterion == 'chi_square':
            return self._chi_square(y, X_column, threshold)
        elif self.criterion == 'mpi':
            return self._mpi(y, X_column, threshold)
        elif self.criterion == 'ort':
            return self._mpi(y, X_column, threshold)
        elif self.criterion == 'twoing':
            return self._twoing(y, X_column, threshold)
        elif self.criterion == 'cair':
            return self._cair(y, X_column, threshold)
        elif self.criterion == 'gain_ratio':
            return self._gain_ratio(y, X_column, threshold)
        else:
            raise ValueError(f"Unsupported criterion: {self.criterion}")
            
    # Gene 0: Information Gain
    def _information_gain(self, y, X_column, threshold):
        left_mask = X_column <= threshold
        right_mask = X_column > threshold
    
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return 0
    
        left_indices = np.where(left_mask)[0]
        right_indices = np.where(right_mask)[0]
    
        return information_gain(y, left_indices, right_indices)

    # Gene 1: Gini
    def _gini_gain(self, y, X_column, threshold):
        left_indices = X_column <= threshold
        right_indices = X_column > threshold
        # check if one group is empty
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0
        # calculate proportion of data in left group
        p = float(len(y[left_indices])) / len(y)
        return gini(y) - p * gini(y[left_indices]) - (1 - p) * gini(y[right_indices])
    
    # Gene 3: G statistic
    def _g_stat(self, y, X_column, threshold):
        return g_stat(y, X_column, threshold)

    # Gene 4: Mandaras
    def _mantaras(self, y, X_column, threshold):
        return mantaras(y, X_column, threshold)
    
    # Gene 5: Hypergeometric Distribution
    def _hg_distribution(self, y, X_column, threshold):
        return hg_distribution(y, X_column, threshold)
    
    # Gene 6: Chandra-Varghese
    def _chv_criterion(self, y, X_column, threshold):
        return chv_criterion(y, X_column, threshold)
    
    # Gene 7: DCSM
    def _dscm(self, y, X_column, threshold):
        return dcsm(y, X_column, threshold)
    
    # Gene 8: Chi-square
    def _chi_square(self, y, X_column, threshold):
        return chi_square(y, X_column, threshold)
    
    # Gene 9: MPI
    def _mpi(self, y, X_column, threshold):
        return mpi(y, X_column, threshold)
    
    # Gene 11: ORT
    def _ort(self, y, X_column, threshold):
        return ORT(y, X_column, threshold)
    
    # Gene 12: Twoing
    def _twoing(self, y, X_column, threshold):
        return twoing(y, X_column, threshold)
    
    # Gene 13: CAIR
    def _cair(self, y, X_column, threshold):
        return cair(y, X_column, threshold)
    
    # Gene 14: Gain Ratio
    def _gain_ratio(self, y, X_column, threshold):
        return gain_ratio(y, X_column, threshold)
    
    
