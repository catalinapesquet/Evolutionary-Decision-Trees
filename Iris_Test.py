# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 13:23:42 2025

@author: Catalina
"""

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from DecisionTree_V2 import DecisionTree, print_tree  
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target

sk_tree = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=10,
    min_samples_split=2,
    random_state=42
)
sk_tree.fit(X, y)

my_tree = DecisionTree(
    criterion='information_gain',
    stopping_criteria='max_depth',
    param=100,  # maps to 10
    mv_split='ignore_all',
    mv_dis='ignore_all',
    mv_classif='most_probable_path',
    pruning_method=None
)
my_tree.fit(X,y)

# print("\n Sklearn DT:\n")
# print(export_text(sk_tree))

# print("\n Custom DT:\n")
# print_tree(my_tree.tree_)