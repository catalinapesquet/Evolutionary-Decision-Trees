# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 18:38:33 2025

@author: Catalina
"""

from sklearn.metrics import recall_score, f1_score, confusion_matrix, multilabel_confusion_matrix
import numpy as np

# Complexity: counting nodes
def count_nodes(node):
    if node is None:
        return 0
    if node.is_leaf:
        return 1
    return 1 + count_nodes(node.left) + count_nodes(node.right)

# Evaluates tree with chosen metric (f1, specificity, recall)
def evaluate_tree(tree, X_train, y_train, X_test, y_test):
    try:
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
    
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    
        specificity = compute_macro_specificity(y_test, y_pred)
    
        n_nodes = count_nodes(tree.tree_)
        
        return {
            "f1": f1,
            "recall": recall,
            "specificity": specificity,
            "n_nodes": n_nodes
        }

    
    except ValueError as e:
        print(f"⚠️ Error during evaluation: {e}")
        # Pénaliser l'arbre
        return {
            "f1": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "n_nodes": 9999  # grand arbre mal noté`
            }


def compute_macro_specificity(y_true, y_pred):
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    specificities = []

    for conf in mcm:
        tn, fp, fn, tp = conf.ravel()
        spec = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        specificities.append(spec)

    return np.mean(specificities)
