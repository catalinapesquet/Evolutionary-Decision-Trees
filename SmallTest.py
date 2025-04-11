# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 15:20:54 2025

@author: Catalina
"""

import pandas as pd 
df = pd.read_excel("D:\downloads\Play.xlsx")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from DecisionTree_V2 import DecisionTree, print_tree  
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import numpy as np
import matplotlib.pyplot as plt

le = LabelEncoder()
df["Weather"] = le.fit_transform(df["Weather"])
df["Temp"] = le.fit_transform(df["Temp"])
df["Play"] = df["Play"].str.strip()
df["Play_binary"] = df["Play"].map({"No": 0, "Yes": 1})


X = df[["Weather", "Temp"]].to_numpy()
y = df["Play_binary"].to_numpy().astype(int) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scikit learn decision tree
sk_tree = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=10,
    min_samples_split=2,
    random_state=42
)
sk_tree.fit(X, y)

# Custom Decision Tree
my_tree = DecisionTree(
    criterion='information_gain',
    stopping_criteria='max_depth',
    param=100,  # maps to 4 
    mv_split='ignore_all',
    mv_dis='ignore_all',
    mv_classif='most_probable_path',
    pruning_method=None
)
my_tree.fit(X,y)

# Compare performance
y_pred_sk = sk_tree.predict(X_test)
y_pred_custom = my_tree.predict(X_test)

# print("Accuracy sklearn:", accuracy_score(y_test, y_pred_sk))
# print("Accuracy custom :", accuracy_score(y_test, y_pred_custom))

# Print tree
# print("\n Sklearn DT:\n")
# print(export_text(sk_tree))

print("\n Custom DT:\n")
print_tree(my_tree.tree_)

# Visualize it 
# plt.figure(figsize=(12, 8))
# plot_tree(
#     sk_tree,
#     filled=True,
#     rounded=True
# )

# plt.title("Scikit-learn Decision Tree")
# plt.show()

from Split_Criteria import information_gain
import numpy as np

# --- Les indices restants dans le sous-arbre après le split x[0] ≤ 0.5 ---
node1_indices = [0, 1, 3, 4, 5, 7, 8, 9, 10, 13]
node1 = y[node1_indices]  # Ce sont les y associés à ce sous-ensemble

# --- Split à x[1] ≤ 0.5 ---
left_global = [3, 7]  # indices (dans le dataset complet) où x[1] ≤ 0.5
right_global = [0, 1, 4, 5, 8, 9, 10, 13]  # reste

# --- Conversion en indices locaux dans node1 ---
left = [i for i, idx in enumerate(node1_indices) if idx in left_global]
right = [i for i, idx in enumerate(node1_indices) if idx in right_global]

# --- Calcul du gain d'information ---
gain = information_gain(node1, left, right)
print(f"Gain d'information : {gain:.4f}")

