# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 13:24:51 2025

@author: Catalina
"""

from sklearn.tree import DecisionTreeClassifier, export_text
from DecisionTree_V2 import DecisionTree, print_tree  
from Plot_Tree import plot_custom_tree
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import numpy as np
import matplotlib.pyplot as plt

wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scikit learn decision tree
sk_tree = DecisionTreeClassifier(
    criterion='gini',
    max_depth=10,
    min_samples_split=2,
    random_state=42
)
sk_tree.fit(X_train, y_train)

# Custom Decision Tree
my_tree = DecisionTree(
    criterion='gini',
    stopping_criteria='max_depth',
    param=100,  # maps to 4 
    mv_split='ignore_all',
    mv_dis='ignore_all',
    mv_classif='most_probable_path',
    pruning_method=None
)
my_tree.fit(X_train, y_train)

# Compare performance
y_pred_sk = sk_tree.predict(X_test)
y_pred_custom = my_tree.predict(X_test)

print("Accuracy sklearn:", accuracy_score(y_test, y_pred_sk))
print("Accuracy custom :", accuracy_score(y_test, y_pred_custom))

# Print tree
print("\n Sklearn DT:\n")
print(export_text(sk_tree, feature_names=wine.feature_names))

print("\n Custom DT:\n")
print_tree(my_tree.tree_)

# Visualize it 
plt.figure(figsize=(12, 8))
plot_tree(
    sk_tree,
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    filled=True,
    rounded=True
)

plt.title("Scikit-learn Decision Tree")
plt.show()
