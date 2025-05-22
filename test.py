# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 16:33:36 2025

@author: Aurora
"""
from DecisionTree import print_tree
from encode_decode import decode
from decode_small import decode_small
from Extract_data import extract_data
from metrics import evaluate_tree
from sklearn.tree import DecisionTreeClassifier, export_text
import numpy as np
import random

# random.seed(42)
# np.random.seed(42)


dataset = "iris"
X_train, X_test, y_train, y_test = extract_data(dataset)

# sk_tree = DecisionTreeClassifier(
#     criterion='entropy',
#     max_depth=2)
# sk_tree.fit(X_train, y_train)
# print("\n Sklearn")
# print(export_text(sk_tree))

# tree_sk = decode_small([2, 1, 0, 0, 55])
# print("\n Custom DT like Sklearn")
# print(f"\n Metrics : {evaluate_tree(tree_sk, X_train, y_train, X_test, y_test)}")
# print_tree(tree_sk.tree_)

indiv_1 =  [1, 0, 1, 1, 4, 1, 3, 66]
tree = decode(indiv_1)
# print(f"indiv_1: {indiv_1}\n")
print(f"\n Metrics_1 : {evaluate_tree(tree, X_train, y_train, X_test, y_test)}")
print("\n DT_1 Result")
print_tree(tree.tree_)

# indiv_2 =  [2, 1, 1, 3, 0, 1, 4, 53]
# tree = decode(indiv_1)
# # print(f"indiv_1: {indiv_1}\n")
# print(f"\n Metrics_2 : {evaluate_tree(tree, X_train, y_train, X_test, y_test)}")
# print("\n DT_2 Result")
# print_tree(tree.tree_)
