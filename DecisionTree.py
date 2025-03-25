# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 11:22:30 2025

@author: Catalina
"""

# DecisionTree 
from Split_Criteria import gini, entropy, information_gain
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris= load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


class DecisionTree:
    def __init__(self, criterion='gini', max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # stopping criteria: max depth reached, pure node or too few samples
        if (depth >= self.max_depth or
            n_labels == 1 or
            n_samples < 2):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # find best split
        feat_idx, threshold = self._best_split(X, y, n_samples, n_features)
        if feat_idx is None:
            # if no good split found, create leaf
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        # split data, grow subtrees
        left_idxs, right_idxs = self._split(X[:, feat_idx], threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(feat_idx, threshold, left, right)

    def _best_split(self, X, y, n_samples, n_features):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx] # column of specific features
            thresholds = np.unique(X_column) 
            for threshold in thresholds: # go through all possible split values of the feature
                #Calculate gain for the current split
                gain = self._calculate_criterion(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold
        return split_idx, split_threshold

    def _calculate_criterion(self, y, X_column, threshold):
        # Choose criterion function based on selected criterion
        if self.criterion == 'gini':
            return self._gini_gain(y, X_column, threshold)
        elif self.criterion == 'entropy':
            return self._information_gain(y, X_column, threshold)

    def _gini_gain(self, y, X_column, threshold):
        left_indices = X_column <= threshold
        right_indices = X_column > threshold
        # check if one group is empty
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0
        # calculate proportion of data in left group
        p = float(len(y[left_indices])) / len(y)
        return gini(y) - p * gini(y[left_indices]) - (1 - p) * gini(y[right_indices])

    def _information_gain(self, y, X_column, threshold):
        left_indices = X_column <= threshold
        right_indices = X_column > threshold
        # check if one group is empty
        if len(y[left_indices]) == 0 or len(y[right_indices]) == 0:
            return 0
        return information_gain(y, left_indices, right_indices)

    def _most_common_label(self, y):
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

tree = DecisionTree(criterion='gini', max_depth=3)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
accuracy
    
        