# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 18:38:33 2025

@author: Catalina
"""

from pymoo.core.problem import ElementwiseProblem
from encode_decode import decode  # à adapter selon ton code
from DecisionTree import DecisionTree
from sklearn.metrics import recall_score, f1_score, confusion_matrix
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
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    else:
        specificity = 0.0

    n_nodes = count_nodes(tree.tree_)

    return {
        "f1": f1,
        "recall": recall,
        "specificity": specificity,
        "n_nodes": n_nodes
    }


            
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from encode_decode import generate_individual

iris= load_iris()
X, y = iris.data, iris.target
# Introduce missing values 
# X[1, 2] = np.nan
# X[3, 2] = np.nan
# X[5, 1] = np.nan
# X[10, 0] = np.nan
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# indiv = [ 2,  4, 63,  2,  1,  1,  1, 21]
# print(indiv)
# tree = decode(indiv)
# tree.fit(X_train, y_train)
# print(tree.tree_)
# metrics = evaluate_tree(tree, X_train, y_train, X_test, y_test)
# print(metrics)
