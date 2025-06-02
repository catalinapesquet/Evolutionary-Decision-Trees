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

        if self.pruning_method is not None:
            pruner = Pruning(self.pruning_method, self.pruning_param)
            self.tree_ = pruner.prune(self.tree_, X, y)

    def _grow_tree(self, X, y, n_tot_samples, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        majority_class = self._most_common_label(y)
        value_distribution = np.bincount(y, minlength=self.n_classes_)
        
        if n_labels == 1 or len(y) <= 1:
            return Node(value=majority_class, class_distribution=value_distribution, 
                        is_leaf=True, samples=len(y))

        # Stopping condition
        if self.stopping_criteria.stop(n_tot_samples, y, depth):
            # print(f"🛑 Stopping split at depth {depth}")
            # print(f"  Samples: {len(y)}, Classes: {np.unique(y)}")
            return Node(value=majority_class, class_distribution=value_distribution,
                        is_leaf=True, samples=len(y))

        # Best split
        split_idx, split_threshold, left_idxs, right_idxs, best_gain = self._best_split(X, y, n_samples, n_features)
        if split_idx is None:
            return Node(value=majority_class, is_leaf=True, samples=len(y[right_idxs]), class_distribution=value_distribution)

        left_idxs, right_idxs = self._split(X[:, split_idx], split_threshold)

        if len(left_idxs) == 0:
            leaf_value = self._most_common_label(y[right_idxs])
            return Node(value=leaf_value, is_leaf=True, samples=len(y[right_idxs]), class_distribution=value_distribution)
        if len(right_idxs) == 0:
            leaf_value = self._most_common_label(y[left_idxs])
            return Node(value=leaf_value, is_leaf=True, samples=len(y[right_idxs]), class_distribution=value_distribution)
        
        # Recursive growth
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], n_tot_samples, depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], n_tot_samples, depth + 1)
            
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
        # If the node is a leaf, no simplification needed
        if node.is_leaf:
            return node
        # Simplify left and right subtrees
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
        best_gain = -1
        split_idx, split_threshold = None, None

        # Prepares data frame
        X_df = pd.DataFrame(X, columns=list(range(X.shape[1])))

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]

            # Special case: weight split
            if self.mv_handler.mv_split == 'weight_split':
                thresholds = np.sort(np.unique(X_column))
                thresholds = (thresholds[:-1] + thresholds[1:]) / 2
                thresholds = np.sort(thresholds)
                for threshold in thresholds:
                    left_indices_temp, right_indices_temp = self._split(X_column, threshold)
                    if len(left_indices_temp) == 0 or len(right_indices_temp) == 0:
                        continue

                    gain = self.mv_handler.weight_split(X_df, y, feat_idx, threshold)
                    # DEBUGGING
                    # print(f"Tested split - Feature {feat_idx} | Threshold: {threshold:.2f} | Gain: {gain:.4f}")

                    if gain <= 0:
                        continue

                    if gain > best_gain:
                        best_gain = gain
                        split_idx = feat_idx
                        split_threshold = threshold
                        left_idxs = left_indices_temp
                        right_idxs = right_indices_temp

                    # Handle case where we have the same gain for multiple thresholds
                    elif gain == best_gain:
                        # DEBUGGING
                        # print(f"✳️ Tie detected: feature {feat_idx}, threshold {threshold}, gain {gain}")
                        # We choose to give priority to the smallest feat_idx and smallest threshold
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

                    if len(y_left) == 0 or len(y_right) == 0:
                        continue
                    gain = self._calculate_criterion(y_filtered, X_column_filtered, threshold)

                    # DEBUGGING
                    # if feat_idx == 4 or feat_idx == 11 :
                        # print(f"Tested split - Feature {feat_idx} | Threshold: {threshold:.2f} | Gain: {gain:.4f}")
                    # print(f"Tested split - Feature {feat_idx} | Threshold: {threshold:.2f} | Gain: {gain:.4f}")

                    if gain <= 0:
                        continue

                    # When we find a better split
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
            # DEBUGGING
            # print(f"✅ Selected split: Feature {split_idx} | Threshold: {split_threshold} | Gain: {best_gain:.4f}")
            # print(f"Left size: {len(left_idxs)}, Contains classes: {np.unique(y[left_idxs])}// Right size: {len(right_idxs)}, Contains classes: {np.unique(y[right_idxs])}")
            return split_idx, split_threshold, left_idxs, right_idxs, best_gain

        # DEBUGGING
        # print("⛔️ No valid split found.")
        return None, None, None, None, None



    def _calculate_criterion(self, y, X_column, threshold):
        criterion = SplitCriterion(self.criterion)
        return criterion.calculate(y, X_column, threshold)

    def _most_common_label(self, y):
        if len(y) == 0:
            raise ValueError("The array 'y' is empty, cannot determine the most common label.")
        most_common = np.argmax(np.bincount(y))
        return most_common

    def _split(self, X_column, split_threshold):
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
        # elif self.mv_classif == 'stop_and_vote':
        #     return self.mv_handler._predict_stop_and_vote(inputs, self.tree_)
        else:
            raise ValueError(f"Unknown mv_classification strategy: {self.mv_classif}")


class Node:
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

def print_tree(node, depth=0, prefix=""):
    indent = "|   " * depth
    if node.is_leaf:
        print(f"{indent}|--- class: {node.leaf_value}")
    else:
        print(f"{indent}|--- feature_{node.feat_idx} <= {node.threshold:.2f}")
        print_tree(node.left, depth + 1)
        print(f"{indent}|--- feature_{node.feat_idx} >  {node.threshold:.2f}")
        print_tree(node.right, depth + 1)
