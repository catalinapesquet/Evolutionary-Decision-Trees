# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 13:24:51 2025

@author: Catalina
"""

from sklearn.tree import DecisionTreeClassifier, export_text
from DecisionTree import DecisionTree, print_tree  # ta classe + print_tree() défini plus haut
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scikit learn decision tree
sk_tree = DecisionTreeClassifier(
    criterion='gini',
    max_depth=4,
    min_samples_split=2,
    random_state=42
)
sk_tree.fit(X_train, y_train)

# Custom Decision Tree
my_tree = DecisionTree(
    criterion='gini',
    stopping_criteria='max_depth',
    param=25,  # maps to 4 
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
print("\n🌲 Arbre sklearn:\n")
print(export_text(sk_tree, feature_names=iris.feature_names))

print("\n🌿 Arbre custom:\n")
print_tree(my_tree.tree_)
