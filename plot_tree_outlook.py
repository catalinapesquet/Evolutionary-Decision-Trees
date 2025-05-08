# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 13:14:12 2025

@author: Aurora
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from DecisionTree import print_tree
from encode_decode import decode
from decode_small import decode_small
from metrics import evaluate_tree

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

sk_tree = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=10)
sk_tree.fit(X_train, y_train)
print("\n Sklearn")
print(export_text(sk_tree))

print("\n Custom DT like Sklearn")
tree_sk = decode_small([0, 1, 100, 0, 55])
print(evaluate_tree(tree_sk, X_train, y_train, X_test, y_test))
print_tree(tree_sk.tree_)

print("\n Custom DT Result")
tree = decode([6, 3, 0, 3, 0, 0, 0, 0])
print(evaluate_tree(tree, X_train, y_train, X_test, y_test))
print_tree(tree.tree_)

