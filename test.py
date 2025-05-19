# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 16:33:36 2025

@author: Aurora
"""
from DecisionTree import print_tree
from encode_decode import decode
from decode_small import decode_small
from Extract_data import extract_data

dataset = "anneal"
X_train, X_test, y_train, y_test = extract_data(dataset)


# sk_tree = DecisionTreeClassifier(
#     criterion='entropy',
#     min_samples_split=26)
# sk_tree.fit(X_train, y_train)
# print("\n Sklearn")
# print(export_text(sk_tree))

# tree_sk = decode_small([1, 2, 2, 0, 55])
# tree_sk.fit(X_train, y_train)
# print("\n Custom DT like Sklearn")
# print_tree(tree_sk.tree_)

from metrics import evaluate_tree
indiv_1 =  [7, 1, 0, 1, 0, 1, 4, 32]
tree = decode(indiv_1)
print(f"indiv_1: {indiv_1}\n")
print(f"\n Metrics_1 : {evaluate_tree(tree, X_train, y_train, X_test, y_test)}")
print("\n DT_1 Result")
print_tree(tree.tree_)


# indiv_2 =  [4, 3, 43, 2, 1, 0, 4, 1] 
# print(f"\n indiv_2: {indiv_2} \n")
# tree = decode(indiv_2)
# print(f"\n Metrics_2 : {evaluate_tree(tree, X_train, y_train, X_test, y_test)}")
# print("\n DT_2 Result")
# print_tree(tree.tree_)