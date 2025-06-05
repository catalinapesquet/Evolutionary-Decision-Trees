# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 17:09:52 2025

@author: Catalina
"""
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split
from scipy.stats import norm

class Pruning:
    """
    Implements multiple decision tree pruning strategies.
    
    The pruning method is selected via `method` and parameterized by `param`.
    Available methods:
    - REP: Reduced Error Pruning
    - PEP: Pessimistic Error Pruning
    - MEP: Minimum Error Pruning (Bayesian)
    - CCP: Cost-Complexity Pruning
    - EBP: Error-Based Pruning with confidence bounds
    """
    def __init__(self, method=None, param=None):
        """
        Initializes the pruning object with a strategy and its parameter.
        
        Parameters:
            method (str): Pruning method identifier (e.g., 'REP', 'PEP', etc.)
            param (int): Gene-mapped hyperparameter controlling pruning aggressiveness.
        """
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
    
    def _count_leaves(self, n):
        """
        Recursively counts the number of leaf nodes in the subtree rooted at n.
        """
        if n is None:
            return 0
        if n.is_leaf:
            return 1
        return self._count_leaves(n.left) + self._count_leaves(n.right)
    
    # Gene 0: REP
    def _REP(self, tree, X, y):
        """
        Reduced-Error Pruning on the tree.
        Split the dataset into training and pruning sets, and prune bottom-up.
        """
        # Split data based on self.param
        param = (self.param * 0.4 + 10)/100 # Clamp between 10% and 50%
        val_size = param
        X_train, X_prune, y_train, y_prune = train_test_split(X, y, test_size=val_size, stratify=y, random_state=42)
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
        nb_leaves = self._count_leaves(node)

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
    
    # Gene 3: CCP
    def _CCP(self, tree, X, y):
        """
        Iteratively generates a pruning sequence by minimizing the 
        cost-complexity trade-off (error vs. number of leaves).
        Selects the best pruned subtree on a validation set.
        """
        val_size = min(max(self.param / 100.0, 0.1), 0.5)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, stratify=y, random_state=42)
        sequence = self._generate_ccp_sequence(tree, X_train, y_train)
        best_tree = self._select_best_ccp_tree(sequence, X_val, y_val)
        return best_tree

    def _generate_ccp_sequence(self, tree, X, y):
        sequence = [deepcopy(tree)]
        while True:
            candidates = []
            self._collect_alpha(tree, tree, X, y, candidates)
            if not candidates:
                break
            node_to_prune = min(candidates, key=lambda x: x['alpha'])
            node = node_to_prune['node']
            X_node, y_node = self._collect_reachable_examples(tree, node, X, y)
            majority_class = max(set(y_node), key=list(y_node).count)
            node.left = None
            node.right = None
            node.is_leaf = True
            node.leaf_value = majority_class
            sequence.append(deepcopy(tree))
        return sequence
    
    def _collect_alpha(self, root, node, X, y, candidates):
        if node is None or node.is_leaf:
            return
        X_node, y_node = self._collect_reachable_examples(root, node, X, y)
        if len(y_node) == 0:
            return
        error_Tt = np.sum([self._predict_from_node(node, x) != y for x, y in zip(X_node, y_node)])
        majority_class = max(set(y_node), key=list(y_node).count)
        error_t = np.sum(y_node != majority_class)
        leaves_Tt = self._count_leaves(node)
        if leaves_Tt > 1:
            alpha = (error_t - error_Tt) / (leaves_Tt - 1)
            candidates.append({'node': node, 'alpha': alpha})
        self._collect_alpha(root, node.left, X, y, candidates)
        self._collect_alpha(root, node.right, X, y, candidates)
        
    def _select_best_ccp_tree(self, sequence, X_val, y_val):
        mapping = [0.5, 1.0, 1.5, 2.0]
        index = min(int(self.param / 25), 3)
        se = mapping[index]
        errors = [self._evaluate_error(t, X_val, y_val) for t in sequence]
        best_error = min(errors)
        std_error = np.std(errors)
        for tree, error in zip(sequence, errors):
            if error <= best_error + se * std_error:
                return tree
        return sequence[-1]  # fallback
    
    # Gene 4: EBP
    def _EBP(self, tree, X, y):
        """
        Uses statistical bounds on classification error (based on binomial distribution)
        to prune nodes that are not significantly better than a single leaf.
        """
        val_size = min(max(self.param / 100.0, 0.1), 0.5)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, stratify=y, random_state=42)

        cf = max(min(self.param / 100, 0.5), 0.01)  # CF in [0.01, 0.5]
        z = norm.ppf(1 - cf)  # critical value for binomial upper bound

        self._prune_node_ebp(tree, tree, X_val, y_val, z)
        return tree

    def _prune_node_ebp(self, node, tree, X_val, y_val, z):
        if node is None or node.is_leaf:
            return node

        node.left = self._prune_node_ebp(node.left, tree, X_val, y_val, z)
        node.right = self._prune_node_ebp(node.right, tree, X_val, y_val, z)

        X_node, y_node = self._collect_reachable_examples(tree, node, X_val, y_val)
        if len(y_node) == 0:
            return node

        majority_class = max(set(y_node), key=list(y_node).count)
        errors_leaf = np.sum(y_node != majority_class)
        n = len(y_node)
        p_hat = errors_leaf / n

        err_leaf = p_hat + z * np.sqrt(p_hat * (1 - p_hat) / n)

        errors_subtree = np.sum([self._predict_from_node(node, x) != y for x, y in zip(X_node, y_node)])
        p_hat_sub = errors_subtree / n
        err_subtree = p_hat_sub + z * np.sqrt(p_hat_sub * (1 - p_hat_sub) / n)

        # Décision
        if err_leaf <= err_subtree:
            node.left = None
            node.right = None
            node.is_leaf = True
            node.leaf_value = majority_class

        return node
    