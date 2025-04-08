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
from Pruning_Strategies import Pruning

import numpy as np
import pandas as pd
from collections import defaultdict
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
    def __init__(self, 
                 criterion='gini',
                 stopping_criteria='max_depth', 
                 param=3,
                 mv_split='ignore_all', 
                 mv_dis='ignore_all', 
                 mv_classif='explore_all',
                 pruning_method=None,
                 pruning_param=None):
        self.criterion = criterion
        self.split_criterion_obj = SplitCriterion(criterion)
        self.param = param
        self.stopping_criteria = stopping_criteria
        self.mv_classif = mv_classif
        self.mv_handler = MissingValues(mv_split, mv_dis, mv_classif, split_criterion=self.split_criterion_obj)
        self.pruning_method = pruning_method
        self.pruning_param = pruning_param

        
    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        # size of total dataset
        n_tot_samples = len(y)
        self.stopping_criteria = StoppingCriterion(n_tot_samples = n_tot_samples, 
                                                   criterion=self.stopping_criteria, 
                                                   param=self.param)
        self.tree_ = self._grow_tree(X, y, n_tot_samples)
        
        
        if self.pruning_method is not None:
            pruner = Pruning(self.pruning_method, self.pruning_param)
            self.tree_ = pruner.prune(self.tree_, X, y)
    
    def _grow_tree(self, X, y, n_tot_samples, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
    
        majority_class = self._most_common_label(y)
    
        # Stopping condition
        if self.stopping_criteria.stop(n_tot_samples, y, depth):
            return Node(value=majority_class, is_leaf=True)
    
        # Best split
        split_idx, split_threshold, left_idxs, right_idxs = self._best_split(X, y, n_samples, n_features)
        if split_idx is None:
            return Node(value=majority_class, is_leaf=True)
    
        left_idxs, right_idxs = self._split(X[:, split_idx], split_threshold)
    
        if len(left_idxs) == 0:
            leaf_value = self._most_common_label(y[right_idxs])
            return Node(value=leaf_value, is_leaf=True)
        if len(right_idxs) == 0:
            leaf_value = self._most_common_label(y[left_idxs])
            return Node(value=leaf_value, is_leaf=True)
    
        # Recursive growth
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], n_tot_samples, depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], n_tot_samples, depth + 1)
    
        node = Node(
            feat_idx=split_idx,
            threshold=split_threshold,
            left=left,
            right=right,
            value=majority_class,
            is_leaf=False
        )
        node.left_weight = len(left_idxs) / (len(left_idxs) + len(right_idxs))
        node.right_weight = len(right_idxs) / (len(left_idxs) + len(right_idxs))
    
        return node

    # def _best_split(self, X, y, n_samples, n_features):
    #     best_gain = -1
    #     split_idx, split_threshold = None, None

    #     for feat_idx in range(n_features):
    #         X_column = X[:, feat_idx]
    #         # X_df = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    #         X_df = pd.DataFrame(X, columns=list(range(X.shape[1]))) #####
    #         X_filtered, y_filtered, X_feature_column = self.mv_handler.handle_split(X_df, y, feature=str(feat_idx))
    #         # Check if all instances have missing value
    #         if X_filtered.empty:
    #             continue

    #         X_column_filtered = X_feature_column.to_numpy()
    #         thresholds = np.unique(X_column)

    #         for threshold in thresholds:
    #             left_indices_temp, right_indices_temp = self._split(X_column, threshold)
    #             if len(left_indices_temp) == 0 or len(right_indices_temp) == 0:
    #                 continue

    #             gain = self._calculate_criterion(y, X_column, threshold)
                    
    #             if gain > best_gain:
    #                 best_gain = gain
    #                 split_idx = feat_idx
    #                 split_threshold = threshold
    #     if split_idx is not None:
    #         X_original_column = pd.Series(X[:, split_idx])
    #         left_idxs, right_idxs = self.mv_handler.apply_split_with_mv(
    #             X_df, y, feature=str(split_idx), split_value=split_threshold
    #         )
    #         return split_idx, split_threshold, left_idxs, right_idxs
    #     return None, None, None, None
    
    def _best_split(self, X, y, n_samples, n_features):
        best_gain = -1
        split_idx, split_threshold = None, None
    
        # Prépare un DataFrame avec colonnes en int
        X_df = pd.DataFrame(X, columns=list(range(X.shape[1])))
    
        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
    
            # ⚠️ Cas spécial : weight_split (retourne un float)
            if self.mv_handler.mv_split == 'weight_split':
                thresholds = np.unique(X_column)
    
                for threshold in thresholds:
                    left_indices_temp, right_indices_temp = self._split(X_column, threshold)
                    if len(left_indices_temp) == 0 or len(right_indices_temp) == 0:
                        continue
    
                    gain = self.mv_handler.weight_split(X_df, y, feat_idx)
    
                    if gain > best_gain:
                        best_gain = gain
                        split_idx = feat_idx
                        split_threshold = threshold
    
            else:
                # 🎯 Cas général : autres stratégies
                try:
                    X_filtered, y_filtered, X_feature_column = self.mv_handler.handle_split(
                        X_df, y, feature=feat_idx
                    )
                except Exception as e:
                    print(f"Erreur avec handle_split sur feature {feat_idx}: {e}")
                    continue
    
                # Skip si données trop manquantes
                if X_filtered.empty or len(y_filtered) == 0:
                    continue
    
                X_column_filtered = X_feature_column.to_numpy()
                thresholds = np.unique(X_column_filtered)
    
                for threshold in thresholds:
                    # 💡 Crée des masques booléens
                    left_mask = X_column_filtered <= threshold
                    right_mask = X_column_filtered > threshold
    
                    if left_mask.sum() == 0 or right_mask.sum() == 0:
                        continue
    
                    y_left = y_filtered[left_mask]
                    y_right = y_filtered[right_mask]
    
                    if len(y_left) == 0 or len(y_right) == 0:
                        continue
    
                    gain = self._calculate_criterion(y_filtered, X_column_filtered, threshold)
    
                    if gain > best_gain:
                        best_gain = gain
                        split_idx = feat_idx
                        split_threshold = threshold
    
        # 🔁 Applique le meilleur split trouvé
        if split_idx is not None:
            left_idxs, right_idxs = self.mv_handler.apply_split_with_mv(
                X_df, y, feature=split_idx, split_value=split_threshold
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
        y_pred = []
        for i, inputs in enumerate(X):
            pred = self._predict(inputs)
            if pred is None:
                print(f"⚠️ Exemple {i} : prédiction = None → inputs = {inputs}")
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
    def __init__(self, feat_idx=None, threshold=None, left=None, right=None, value=None, is_leaf=False):
        self.feat_idx = feat_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.leaf_value = value if is_leaf else None
        self.majority_class = value  # toujours défini

# Testing on a well known dataset
iris= load_iris()
X, y = iris.data, iris.target
# Introduce missing values 
X[1, 2] = np.nan
X[3, 2] = np.nan
X[5, 1] = np.nan
X[10, 0] = np.nan
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# With REP pruning
tree = DecisionTree(criterion='g_stat', 
                    stopping_criteria='max_depth',
                    param=40, 
                    mv_split='ignore_all',
                    mv_dis='most_probable_partition',
                    mv_classif='explore_all',
                    pruning_method='MEP',
                    pruning_param=20)  
tree.fit(X_train, y_train)
y_pred_rep = tree.predict(X_test)
acc_rep = accuracy_score(y_test, y_pred_rep)
print(f"Accuracy: {acc_rep:.4f}")
