# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 11:22:30 2025

@author: Catalina
"""

from Split_Criteria import (gini,
                            information_gain,
                            g_stat,
                            mantaras, 
                            hg_distribution,
                            chv_criterion,
                            dcsm,
                            chi_square,
                            mpi,
                            ORT,
                            twoing,
                            cair,
                            gain_ratio,
                            SplitCriterion)
from Stopping_Criteria import StoppingCriterion
from Missing_Values import MissingValues

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DecisionTree:
    """
    Implements a Classification Decision Tree
    
    Attributes:
        criterion : str
            Split criterion.
        param : int
            Parameter used by the stopping criterion. For example maximum depth of the decision tree. 
        stopping_criteria : StoppingCriterion
            Stopping criteria to stop the tree from growing. 
        mv_handler : MissingValues
            Handler to treat missing values in data.
    
    Methods:
        fit(X, y):
            Train decision tree on training data.
        _grow_tree(X, y, n_tot_samples, depth=0):
            Recursively grows the tree by splitting the data according to the best split criteria. 
        _best_split(X, y, n_samples, n_features):
            Find best split according to split criterion.
        ...
    """
    def __init__(self, stopping_criteria, criterion='gini', mv_split='ignore_all', param=3):
        self.criterion = criterion
        self.param = param
        self.stopping_criteria = stopping_criteria
        self.mv_handler = MissingValues(mv_split)
        
    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        # size of total dataset
        n_tot_samples = len(y)
        self.stopping_criteria = StoppingCriterion(n_tot_samples = n_tot_samples, 
                                                   criterion=self.stopping_criteria, 
                                                   param=self.param)
        self.tree_ = self._grow_tree(X, y, n_tot_samples)

    def _grow_tree(self, X, y, n_tot_samples, depth=0):
        # size of parent node
        n_samples, n_features = X.shape 
        n_labels = len(np.unique(y))
    
        # Check if stopping condition is reached
        if self.stopping_criteria.stop(n_tot_samples, y, depth):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # Find best split
        split_idx, split_threshold, left_idxs, right_idxs = self._best_split(X, y, n_samples, n_features)
        if split_idx is None:
            # If no good split is found, create leaf node
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        
        left_idxs, right_idxs = self._split(X[:, split_idx], split_threshold)
    
        # Check if group is empty
        if len(left_idxs) == 0:  
            leaf_value = self._most_common_label(y[right_idxs])
            return Node(value=leaf_value)
        if len(right_idxs) == 0: 
            leaf_value = self._most_common_label(y[left_idxs])
            return Node(value=leaf_value)
        
    
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], n_tot_samples, depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], n_tot_samples, depth + 1)
    
        return Node(split_idx, split_threshold, left, right)

    def _best_split(self, X, y, n_samples, n_features):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            X_df = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])

            if self.mv_handler.mv_split == "weight_split":
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    gain = self.mv_handler.calculate_weighted_information_gain(X_df, y, str(feat_idx), criterion=self.criterion)
                    if gain > best_gain:
                        best_gain = gain
                        split_idx = feat_idx
                        split_threshold = threshold 
            else:
                X_filtered, y_filtered, X_feature_column = self.mv_handler.handle_split(X_df, y, feature=str(feat_idx))
                if X_filtered.empty:
                    continue

                X_column_filtered = X_feature_column.to_numpy()
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    left_indices_temp, right_indices_temp = self._split(X_column, threshold)
                    if len(left_indices_temp) == 0 or len(right_indices_temp) == 0:
                        continue

                    gain = self._calculate_criterion(y, X_column, threshold)
                    
                    if gain > best_gain:
                        best_gain = gain
                        split_idx = feat_idx
                        split_threshold = threshold

        if split_idx is not None:
            X_original_column = pd.Series(X[:, split_idx])
            left_idxs, right_idxs = self.mv_handler.apply_split_with_mv(
                X_original_column, split_threshold, self.mv_handler.mv_split
            )
            return split_idx, split_threshold, left_idxs, right_idxs

        return None, None, None, None

    def _calculate_criterion(self, y, X_column, threshold):
        criterion = SplitCriterion(self.criterion)
        return criterion.calculate(y, X_column, threshold)

    def _most_common_label(self, y):
        if len(y) == 0:
            raise ValueError("The array 'y' is empty, cannot determine the most common label.")
        most_common = np.argmax(np.bincount(y))
        return most_common

    def _split(self, X_column, split_threshold):
        left_indices = np.argwhere(X_column <= split_threshold).flatten()
        right_indices = np.argwhere(X_column > split_threshold).flatten()
        return left_indices, right_indices
    
    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feat_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

class Node:
    def __init__(self, feat_idx=None, threshold=None, left=None, right=None, *, value=None):
        self.feat_idx = feat_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Testing on a well known dataset
iris= load_iris()
X, y = iris.data, iris.target
# Introduce missing values 
X[1, 2] = np.nan
X[3, 2] = np.nan
X[5, 1] = np.nan
X[10, 0] = np.nan
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree = DecisionTree(criterion='gini', stopping_criteria='max_depth', param=3, mv_split='weight_split')
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
    
        