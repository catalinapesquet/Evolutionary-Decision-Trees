# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 17:09:52 2025

@author: Catalina
"""
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split

class Pruning:
    def __init__(self, method=None, param=None):
        self.method = method
        self.param = param
        
    def prune(self, tree, X_train, y_train):
        """ Select the pruning strategy based on self.method."""
        if self.method == 'REP':
            return self._REP(tree, X_train, y_train)
        elif self.method == 'PEP':
            return self._PEP(tree, X_train, y_train)
        elif self.method == 'MEP':
            return self._MEP(tree, X_train, y_train)       
        elif self.method == 'CCP':
            return self._CCP(tree, X_train, y_train)
        elif self.method == 'EBP':
            return self._EBP(tree, X_train, y_train)
        
    def _predict_from_node(self, node, x):
        """Predict the class label for a single sample x starting from node."""
        while not node.is_leaf:
            if x[node.feat_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.leaf_value
    
    def _evaluate_error(self, tree, X_val, y_val):
        """ Evaluate total number of misclassification on validation set."""
        y_pred = [self._predict_from_node(tree, x) for x in X_val]
        return np.sum(y_pred != y_val)  

    def _collect_reachable_examples(self, root, target_node, X, y):
        """
        Collect all examples (X, y) that reach the given target_node during tree traversal.
        This is useful for localized pruning decisions.
        """
        reachable_X = []
        reachable_y = []

        for xi, yi in zip(X, y):
            node = root
            while node is not None and not node.is_leaf:
                if node == target_node:
                    reachable_X.append(xi)
                    reachable_y.append(yi)
                    break
                if xi[node.feat_idx] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            if node == target_node:
                reachable_X.append(xi)
                reachable_y.append(yi)

        return np.array(reachable_X), np.array(reachable_y)
    
    # Gene 0: REP
    def _REP(self, tree, X, y):
        """
        Reduced-Error Pruning on the tree.
        Split the dataset into training and pruning sets, and prune bottom-up.
        """
        # Split data based on self.param
        param = (self.param * 0.4 + 10)/100 # Clamp between 10% and 50%
        val_size = param
        X_train, X_prune, y_train, y_prune = train_test_split(X, y, test_size=val_size, stratify=y)
        tree = self._prune_node_rep(tree, tree, X_prune, y_prune)
        return tree
    
    def _prune_node_rep(self, node, tree, X_val, y_val):
        """
        Recursive pruning method for REP.
        If replacing a subtree with a leaf improves or maintains validation accuracy, prune it.
        """
        if node is None or node.is_leaf:
            return node

        node.left = self._prune_node_rep(node.left, tree, X_val, y_val)
        node.right = self._prune_node_rep(node.right, tree, X_val, y_val)

        error_before = self._evaluate_error(tree, X_val, y_val)

        original_left = node.left
        original_right = node.right
        original_leaf = node.is_leaf
        original_leaf_value = node.leaf_value

        node.left = None
        node.right = None
        node.is_leaf = True
        node.leaf_value = node.majority_class

        error_after = self._evaluate_error(tree, X_val, y_val)

        if error_after > error_before:
            node.left = original_left
            node.right = original_right
            node.is_leaf = original_leaf
            node.leaf_value = original_leaf_value

        return node
    
    # Gene 1: PEP
    def _PEP(self, tree, X, y):
        """ Pessimistic Error Pruning"""
        return self._prune_node_pep(tree, tree, X, y)

    def _prune_node_pep(self, node, tree, X_val, y_val):
        """
        Recursive pruning method for PEP.
        Uses standard error multiplier (self.param) to adjust the pessimistic error.
        """
        if node is None or node.is_leaf:
            return node

        node.left = self._prune_node_pep(node.left, tree, X_val, y_val)
        node.right = self._prune_node_pep(node.right, tree, X_val, y_val)

        X_node, y_node = self._collect_reachable_examples(tree, node, X_val, y_val)
        if len(y_node) == 0:
            return node
        
        errors_subtree = np.sum([self._predict_from_node(node, x) != y for x, y in zip(X_node, y_node)])
        def count_leaves(n):
            if n is None:
                return 0
            if n.is_leaf:
                return 1
            return count_leaves(n.left) + count_leaves(n.right)

        nb_leaves = count_leaves(node)

        # Apply SE correction dynamically based on self.param
        mapping = [0.5, 1.0, 1.5, 2.0]
        index = min(int(self.param / 25), 3)
        se = mapping[index]
        pessimistic_error_subtree = errors_subtree + se * 0.5 * nb_leaves

        majority_class = max(set(y_node), key=list(y_node).count)
        errors_leaf = np.sum(y_node != majority_class)
        pessimistic_error_leaf = errors_leaf + se * 0.5

        if pessimistic_error_leaf <= pessimistic_error_subtree:
            node.left = None
            node.right = None
            node.is_leaf = True
            node.leaf_value = majority_class

        return node
    
    # Gene 2: MEP
    def _MEP(self, tree, X, y):
        """
        Apply Minimum Error Pruning using a Bayesian m-estimate.
        Parameter `self.param` is mapped directly as `m` ∈ [0, 100].
        """
        return self._prune_node_mep(tree, tree, X, y)
    
    def _prune_node_mep(self, node, tree, X_val, y_val):
        """
        Recursive pruning method for MEP (Minimum Error Pruning).
        Uses Bayesian m-estimate to compare subtree error vs leaf error.
        """
        if node is None or node.is_leaf:
            return node
    
        # First recursively prune the children
        node.left = self._prune_node_mep(node.left, tree, X_val, y_val)
        node.right = self._prune_node_mep(node.right, tree, X_val, y_val)
    
        # Get examples reaching this node
        X_node, y_node = self._collect_reachable_examples(tree, node, X_val, y_val)
        if len(y_node) == 0:
            return node
    
        # Estimate errors with m-estimate
        m = max(1, self.param)  # avoid division by zero
    
        # Compute expected error of subtree
        errors_subtree = np.sum([self._predict_from_node(node, x) != y for x, y in zip(X_node, y_node)])
        n_subtree = len(y_node)
        est_error_subtree = (errors_subtree + m * 0.5) / (n_subtree + m)
    
        # Compute expected error of replacing the node with a leaf
        majority_class = max(set(y_node), key=list(y_node).count)
        errors_leaf = np.sum(y_node != majority_class)
        est_error_leaf = (errors_leaf + m * 0.5) / (n_subtree + m)
    
        # If pruning improves expected error → prune
        if est_error_leaf <= est_error_subtree:
            node.left = None
            node.right = None
            node.is_leaf = True
            node.leaf_value = majority_class
    
        return node

    def CCP(self, tree, X, y):
        pass
    def EBP(self, tree, X, y):
        pass
    