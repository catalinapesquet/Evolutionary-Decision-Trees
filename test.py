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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# indiv = generate_individual()
# print(indiv)

# sk_tree = DecisionTreeClassifier(
#     criterion='entropy',
#     min_samples_split=26)
# sk_tree.fit(X_train, y_train)
# # print("\n Sklearn")
# # print(export_text(sk_tree))

# tree_sk = decode_small([1, 1, 2, 0, 55])
# tree_sk.fit(X_train, y_train)
# # print("\n Custom DT like Sklearn")
# # print_tree(tree_sk.tree_)

from metrics import count_nodes, evaluate_tree
indiv_1 = [12, 3, 50, 0, 1, 1, 3, 20]
tree = decode(indiv_1)
print(f"indiv_1: {indiv_1}\n")
print(f"\n Metrics_1 : {evaluate_tree(tree, X_train, y_train, X_test, y_test)}")
print("\n DT_1 Result")
print_tree(tree.tree_)


indiv_2 = [3, 3, 0, 0, 0, 0, 0, 0]
print(f"\n indiv_2: {indiv_2} \n")
tree = decode(indiv_2)
print(f"\n Metrics_2 : {evaluate_tree(tree, X_train, y_train, X_test, y_test)}")
print("\n DT_2 Result")
print_tree(tree.tree_)

