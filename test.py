# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 16:33:36 2025

@author: Aurora
"""
from encode_decode import decode
from Extract_data import extract_data
from metrics import evaluate_tree
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix
from Plot_tree import print_tree, export_tree_dot
import matplotlib.pyplot as plt

dataset = "iris"
X_train, X_test, y_train, y_test = extract_data(dataset)
    
# # Cart
# cart = DecisionTreeClassifier(max_depth=3)
# cart.fit(X_train, y_train)
# y_pred_cart = cart.predict(X_test)
# f1 = f1_score(y_test, y_pred_cart, average='macro')
# recall = recall_score(y_test, y_pred_cart, average='macro')
# accuracy = accuracy_score(y_test, y_pred_cart)
# print(f"CART: F1 = {f1:.4f}, Recall = {recall:.4f}, Accuracy = {accuracy:.4f}")
# print(f"Number of nodes: {cart.tree_.node_count}")
# fig = plt.figure(figsize=(25,20))
# _ = tree.plot_tree(cart,
#                    filled=True)
# # print(export_text(cart))

# # C4.5 approx with 'entropy'
# c45 = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=3)
# c45.fit(X_train, y_train)
# y_pred_c45 = c45.predict(X_test)
# f1_c45 = f1_score(y_test, y_pred_c45, average='macro')
# recall_c45 = recall_score(y_test, y_pred_c45, average='macro')
# print(f"C4.5: F1 = {f1_c45:.4f}, Recall = {recall_c45:.4f}")
# print(f"Number of nodes: {c45.tree_.node_count}")
# # print(export_text(c45))
# fig = plt.figure(figsize=(25,20))
# _ = tree.plot_tree(c45,
#                    filled=True)

# Our solution
indiv_1 =  [2, 1, 1, 1, 1, 1, 4, 16]
tree1 = decode(indiv_1)
# print(f"indiv_1: {indiv_1}\n")
print(f"\n Metrics_1 : {evaluate_tree(tree1, X_train, y_train, X_test, y_test)}")
print("\n DT_1 Result")
print_tree(tree1.tree_)
export_tree_dot(tree1.tree_)