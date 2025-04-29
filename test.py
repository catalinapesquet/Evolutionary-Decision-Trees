# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 16:33:36 2025

@author: Aurora
"""
import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score
from DecisionTree import DecisionTree, print_tree
from DT_genes import SPLIT_CRITERIA, STOPPING_CRITERIA, MV_SPLIT, MV_DISTRIBUTION, MV_CLASSIFICATION, PRUNNING_GENES
from decode_small import decode_small
from encode_decode import decode
import matplotlib.pyplot as plt

iris= load_iris()
X, y = iris.data, iris.target
# # Introduce missing values 
# X[1, 2] = np.nan
# X[3, 2] = np.nan
# X[5, 1] = np.nan
# X[10, 0] = np.nan
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# indiv = generate_individual()
# print(indiv)

# y_pred = tree.predict(X_test)
# acc_rep = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {acc_rep:.4f}")

sk_tree = DecisionTreeClassifier(
    criterion='entropy',
    min_samples_split=26)
sk_tree.fit(X_train, y_train)
# print("\n Sklearn")
# print(export_text(sk_tree))

tree_sk = decode_small([1, 1, 2, 0, 55])
tree_sk.fit(X_train, y_train)
# print("\n Custom DT like Sklearn")
# print_tree(tree_sk.tree_)

tree = decode([1, 2, 8, 3, 2, 1, 4, 98])
tree.fit(X_train, y_train)
print("\n Custom DT Result")
print_tree(tree.tree_)

from metrics import count_nodes, evaluate_tree

print(evaluate_tree(tree, X_train, y_train, X_test, y_test))