# -*- coding: utf-8 -*-More actions
"""
Created on Tue Mar 25 11:22:30 2025

@author: Catalina
"""

from Split_Criteria import SplitCriterion
from Stopping_Criteria import StoppingCriterion
from Missing_Values import MissingValues
from Pruning_Strategies import Pruning

import numpy as np
import pandas as pd
import random

random.seed(42)
np.random.seed(42)

class DecisionTree:
    """
    A flexible implementation of a classification decision tree that supports:
    - Multiple split criteria 
    - Multiple stopping criteria 
    - Missing value handling (at split time, during distribution and classification)
    - Optional post-pruning strategies 

    Attributes:
        criterion : str
            Name of the split criterion (e.g., 'gini', 'information_gain', etc.)
        split_criterion_obj : SplitCriterion
            Object used to compute the quality of splits.
        stopping_criteria : str or StoppingCriterion
            Name of the stopping criterion or an initialized object later.
        param : int
            Gene-controlled parameter passed to the stopping criterion.
        mv_dis: str
            Strategy for handling missing values during evaluation of the split.
        mv_dis: str
            Strategy for handling missing values during distribution.
        mv_classif : str
            Strategy for handling missing values during classification.
        pruning_method : str or None
            Name of the pruning strategy to apply after training.
        pruning_param : int or None
            Hyperparameter associated with the pruning method.
        min_samples_split : int
            Minimum number of samples required to perform a split. (not a gene)
    """
    def __init__(self, 
                 criterion='gini',
                 stopping_criteria='max_depth', 
                 param=3,
                 mv_split='ignore_all', 
                 mv_dis='ignore_all', 
                 mv_classif='explore_all',
                 pruning_method=None,
                 pruning_param=None,
                 min_samples_split=1):
        self.criterion = criterion
        self.split_criterion_obj = SplitCriterion(criterion)
        self.param = param
        self.stopping_criteria = stopping_criteria
        self.mv_classif = mv_classif
        self.mv_handler = MissingValues(mv_split, mv_dis, mv_classif, split_criterion=self.split_criterion_obj)
        self.pruning_method = pruning_method
        self.pruning_param = pruning_param
        self.min_samples_split = min_samples_split


    def fit(self, X, y):
        """
        Trains the decision tree on the provided dataset.

        - Initializes the number of classes and features.
        - Configures the stopping criterion.
        - Builds the tree recursively with `_grow_tree`.
        - Simplifies redundant branches using `_simplify_tree`.
        - Optionally prunes the tree if a pruning method is specified.
        
        Parameters:
            X : Training features (n_samples, n_features).
            y : Corresponding class labels.
        """
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1] 
        
        # size of total dataset
        n_tot_samples = len(y)
        
        self.stopping_criteria = StoppingCriterion(n_tot_samples = n_tot_samples, 
                                                   criterion=self.stopping_criteria, 
                                                   param=self.param)
        # Train the tree recursively
        self.tree_ = self._grow_tree(X, y, n_tot_samples)

        # Post-process the tree to remove useless splits
        self.tree_ = self._simplify_tree(self.tree_)
        
        # Post process the tree with the chosen pruning strategy
        if self.pruning_method is not None:
            pruner = Pruning(self.pruning_method, self.pruning_param)
            self.tree_ = pruner.prune(self.tree_, X, y)
            

    def _grow_tree(self, X, y, n_tot_samples, depth=0):
        """
        Recursively builds the decision tree by selecting the best splits.
    
        Base cases:
        - All labels are identical or only one sample remains → create leaf.
        - Stopping condition is met → create leaf.
    
        Steps:
        - Finds the best split using `_best_split`.
        - If no valid split is found → create leaf.
        - Recursively grows left and right subtrees.
        - Returns an internal node with split information and child nodes.
    
        Parameters:
            X : Subset of input features at this node.
            y : Corresponding labels.
            n_tot_samples : Total number of samples in the original dataset.
            depth : Current depth in the tree.
    
        Returns:
            Node: A tree node (leaf or internal).
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        majority_class = self._most_common_label(y)
        value_distribution = np.bincount(y, minlength=self.n_classes_)
        
        # If all labels are identical or only one sample remains: create a leaf
        if n_labels == 1 or len(y) <= 1:
            return Node(value=majority_class, class_distribution=value_distribution, 
                        is_leaf=True, samples=len(y))

        # Check if Stopping condition is complete
        if self.stopping_criteria.stop(n_tot_samples, y, depth):
            # print(f"🛑 Stopping split at depth {depth}")
            # print(f"  Samples: {len(y)}, Classes: {np.unique(y)}")
            return Node(value=majority_class, class_distribution=value_distribution,
                        is_leaf=True, samples=len(y))

        # Search for Best possible split
        split_idx, split_threshold, left_idxs, right_idxs, best_gain = self._best_split(X, y, n_samples, n_features)
        
        # If no valid split is found : create a leaf
        if split_idx is None:
            return Node(value=majority_class, is_leaf=True, samples=len(y[right_idxs]), class_distribution=value_distribution)

        left_idxs, right_idxs = self._split(X[:, split_idx], split_threshold)

        # Check if split doesnt create an empty group, else: create a leaf
        if len(left_idxs) == 0:
            leaf_value = self._most_common_label(y[right_idxs])
            return Node(value=leaf_value, is_leaf=True, samples=len(y[right_idxs]), class_distribution=value_distribution)
        if len(right_idxs) == 0:
            leaf_value = self._most_common_label(y[left_idxs])
            return Node(value=leaf_value, is_leaf=True, samples=len(y[right_idxs]), class_distribution=value_distribution)
        
        # Recursive growth
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], n_tot_samples, depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], n_tot_samples, depth + 1)
        
        # Create a tree node (internal or leaf)
        node = Node(
            feat_idx=split_idx,
            threshold=split_threshold,
            left=left,
            right=right,
            value=majority_class,
            is_leaf=False,
            gain=best_gain,
            samples=len(y),
            class_distribution=value_distribution
        )
        
        node.left_weight = len(left_idxs) / (len(left_idxs) + len(right_idxs))
        node.right_weight = len(right_idxs) / (len(left_idxs) + len(right_idxs))

        return node

    def _simplify_tree(self, node):
        """
        Recursively builds the decision tree by selecting the best splits.
    
        Base cases:
        - If the node is a leaf then no simplification is needed.
        - If both childs are leaves and predict the same class -> merge them into one node.
    
        Parameters:
            node (Node): Current node.
    
        Returns:
            node:a simplified version of the node.
        """
        # If the node is a leaf, no simplification needed
        if node.is_leaf:
            return node
        
        # Recursevely simplify left and right subtrees
        node.left = self._simplify_tree(node.left)
        node.right = self._simplify_tree(node.right)

        # If both children are leaves and predict the same class, collapse this node
        if node.left.is_leaf and node.right.is_leaf:
            if node.left.leaf_value == node.right.leaf_value:
                return Node(
                    is_leaf=True,
                    value=node.left.leaf_value,
                )
        # Otherwise, keep the current node
        return node

    def _best_split(self, X, y, n_samples, n_features):
        """
        Finds the best feature and threshold to split the data.
    
        For each feature:
        - Handles missing values according to the configured strategy.
        - Computes candidate thresholds (midpoints between unique sorted values).
        - Evaluates the quality of each split using the split criterion.
        - Keeps the split with the highest gain.
    
        Special case:
        - If mv_split is 'weight_split', adjusts gain based on missing value proportion.
    
        Parameters:
            X : Input features.
            y : Class labels.
            n_samples (int): Number of current samples.
            n_features (int): Total number of features.
    
        Returns:
            tuple: (best feature index, best threshold, left indices, right indices, best gain),
                   or (None, None, None, None, None) if no valid split is found.
        """
        # Initialize gain to -1
        best_gain = -1
        split_idx, split_threshold = None, None

        # Prepares data frame
        X_df = pd.DataFrame(X, columns=list(range(X.shape[1])))
        
        
        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]

            # Special case for weight split: adjusts gain based on missing value proportion.
            if self.mv_handler.mv_split == 'weight_split':
                thresholds = np.sort(np.unique(X_column))
                thresholds = (thresholds[:-1] + thresholds[1:]) / 2
                thresholds = np.sort(thresholds)
                for threshold in thresholds:
                    left_indices_temp, right_indices_temp = self._split(X_column, threshold)
                    if len(left_indices_temp) == 0 or len(right_indices_temp) == 0:
                        continue

                    gain = self.mv_handler.weight_split(X_df, y, feat_idx, threshold)
                    
                    if gain <= 0:
                        continue
                    
                    # A better gain has been found:
                    if gain > best_gain:
                        best_gain = gain
                        split_idx = feat_idx
                        split_threshold = threshold
                        left_idxs = left_indices_temp
                        right_idxs = right_indices_temp

                    # Handle case where we have the same gain for multiple thresholds
                    elif gain == best_gain:
                        if split_idx is None or (feat_idx, threshold) < (split_idx, split_threshold):
                            split_idx = feat_idx
                            split_threshold = threshold
                            left_idxs = left_indices_temp
                            right_idxs = right_indices_temp


            # All other cases             
            else:
                try:
                    X_filtered, y_filtered, X_feature_column = self.mv_handler.handle_split(
                        X_df, y, feature=feat_idx
                    )
                except Exception as e:
                    print(f"Error with handle_split on feature {feat_idx}: {e}")
                    continue

                if X_filtered.empty or len(y_filtered) == 0:
                    continue

                X_column_filtered = X_feature_column.to_numpy()
                vals = np.sort(np.unique(X_column_filtered))
                if len(vals) < 2:
                    continue 

                thresholds = (vals[:-1] + vals[1:]) / 2  # midpoints between values

                for threshold in thresholds:
                    left_mask = X_column_filtered <= threshold
                    right_mask = X_column_filtered > threshold
                    
                    if left_mask.sum() == 0 or right_mask.sum() == 0:
                        continue

                    if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                        continue

                    y_left = y_filtered[left_mask]
                    y_right = y_filtered[right_mask]
                    
                    # Check if split doesnt create any empty group
                    if len(y_left) == 0 or len(y_right) == 0:
                        continue
                    gain = self._calculate_criterion(y_filtered, X_column_filtered, threshold)

                    if gain <= 0:
                        continue

                    # A better gain has been found:
                    if gain > best_gain:
                        best_gain = gain
                        split_idx = feat_idx
                        split_threshold = threshold

                        mask = X_column_filtered <= threshold
                        left_idxs = np.sort(X_filtered[mask].index.to_numpy())
                        right_idxs = np.sort(X_filtered[~mask].index.to_numpy())

                    elif gain == best_gain:
                        # print(f"✳️ Tie detected: feature {feat_idx}, threshold {threshold}, gain {gain}")
                        if split_idx is None or (feat_idx, threshold) < (split_idx, split_threshold):
                            split_idx = feat_idx
                            split_threshold = threshold
                            left_idxs = left_indices_temp
                            right_idxs = right_indices_temp

        if split_idx is not None:
            return split_idx, split_threshold, left_idxs, right_idxs, best_gain

        return None, None, None, None, None
    

    def _calculate_criterion(self, y, X_column, threshold):
        """
        Calculate the chosen splitting criterion
        """
        criterion = SplitCriterion(self.criterion)
        return criterion.calculate(y, X_column, threshold)

    def _most_common_label(self, y):
        """
        Calculate most common value in a list of label
        """
        if len(y) == 0:
            raise ValueError("The array 'y' is empty, cannot determine the most common label.")
        most_common = np.argmax(np.bincount(y))
        return most_common

    def _split(self, X_column, split_threshold):
        """
        Splits indices of samples based on a threshold on a single feature.
    
        """
        left_indices = np.sort(np.argwhere(X_column <= split_threshold).flatten())
        right_indices = np.sort(np.argwhere(X_column > split_threshold).flatten())
        return left_indices, right_indices

    def predict(self, X):
        y_pred = []
        for i, inputs in enumerate(X):
            pred = self._predict(inputs)
            if pred is None:
                print(f"⚠️ Exemple {i} : prediction = None → inputs = {inputs}")
            y_pred.append(pred)
        return y_pred


    def _predict(self, inputs):
        if self.mv_classif == 'explore_all':
            result = self.mv_handler.predict_explore_all(inputs, self.tree_, weight=1.0)
            return max(result.items(), key=lambda x: x[1])[0]
        elif self.mv_classif == 'most_probable_path':
            return self.mv_handler._predict_most_probable(inputs, self.tree_)
        else:
            raise ValueError(f"Unknown mv_classification strategy: {self.mv_classif}")


class Node:
    """
    Represents a node in the decision tree.

    Attributes:
        feat_idx (int): Index of the feature used for splitting (None if leaf).
        threshold (float): Threshold value used for splitting (None if leaf).
        left (Node): Left child node.
        right (Node): Right child node.
        value (int): Majority class of the samples at this node.
        is_leaf (bool): True if the node is a leaf.
        leaf_value (int): Predicted class at a leaf (same as value).
        left_weight (float): Proportion of data going to the left (used for MV distribution).
        right_weight (float): Proportion of data going to the right.
    """
    def __init__(self, feat_idx=None, threshold=None, left=None, right=None, 
                 value=None, is_leaf=False, leaf_value=None, gain=None,
                 samples=None, class_distribution=None):
        self.feat_idx = feat_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.value = value
        self.leaf_value = value if is_leaf else None
        self.majority_class = value  
        self.gain = gain
        self.samples = samples
        self.class_distribution = class_distribution
