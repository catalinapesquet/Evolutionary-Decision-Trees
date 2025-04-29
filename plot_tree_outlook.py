# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 13:14:12 2025

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
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Reload dataset with proper delimiter
df = pd.read_csv("C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\DT\\Play(Sheet1).csv", sep=";")

le = LabelEncoder()
df["Weather"] = le.fit_transform(df["Weather"])
df["Temp"] = le.fit_transform(df["Temp"])
df["Play"] = df["Play"].str.strip()
df["Play_binary"] = df["Play"].map({"No":0, "Yes":1})

# Split into features and target
X = df[["Weather", "Temp"]].to_numpy()
y = df["Play_binary"].to_numpy().astype(int)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# indiv = generate_individual()
# print(indiv)

# y_pred = tree.predict(X_test)
# acc_rep = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {acc_rep:.4f}")

sk_tree = DecisionTreeClassifier(
    criterion='entropy',
    min_samples_split=26)
sk_tree.fit(X_train, y_train)
print("\n Sklearn")
print(export_text(sk_tree))

tree_sk = decode_small([1, 1, 2, 0, 55])
tree_sk.fit(X_train, y_train)
# print("\n Custom DT like Sklearn")
# print_tree(tree_sk.tree_)

tree = decode([11,1])
tree.fit(X_train, y_train)
print("\n Custom DT Result")
print_tree(tree.tree_)

from metrics import count_nodes, evaluate_tree

print(evaluate_tree(tree, X_train, y_train, X_test, y_test))