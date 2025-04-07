# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 12:14:19 2025

@author: Catalina
"""

import random
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from DecisionTree import DecisionTree
from DT_genes import SPLIT_CRITERIA, STOPPING_CRITERIA, MV_SPLIT, MV_DISTRIBUTION, MV_CLASSIFICATION, PRUNNING_GENES

def generate_individual():
    """
    Generates a random list of 9 genes representing the configuration of an evolutionary decision tree.
    Each gene is an integer within a defined range corresponding to a specific component of the decision tree.
    """
    gene_bounds = [
        (0, 15),   # split criterion
        (0, 1),    # split type
        (0, 6),    # stopping criterion
        (0, 100),  # stopping value
        (0, 3),    # missing value strategy
        (0, 5),    # pruning strategy
        (0, 100),  # pruning value
        (0, 1),    # attribute weighting
        (0, 1)     # post-pruning
    ]
    return [random.randint(low, high) for (low, high) in gene_bounds]

# [split, split_type, stopping_crit, stopping_param, mv_split, mv_dis, mv_classif, pruning_tech, pruning_param]

def generate_population(n):
    """
    Generate a population of n individuals (lists of 9 genes each).
    """
    return [generate_individual() for _ in range(n)]

def create_tree(gene_list):
    """
    Create a DecisionTree instance from a gene list using standardized dictionaries.
    """

    # Décodage des gènes
    split_criterion = SPLIT_CRITERIA.get(gene_list[0], "gini")
    # split_type = BINARY_SPLIT.get(gene_list[1], "binary")  # non utilisé pour l’instant
    stopping_criterion = STOPPING_CRITERIA.get(gene_list[2], "max_tree_depth")
    stopping_param = gene_list[3]

    mv_split_strategy = MV_SPLIT.get(gene_list[4], "ignore_all")
    mv_dis_strategy = MV_DISTRIBUTION.get(gene_list[5], "ignore_all")
    mv_classif_strategy = MV_CLASSIFICATION.get(gene_list[6], "explore_all")

    pruning_method = PRUNNING_GENES.get(gene_list[7], "none")
    pruning_param = gene_list[8]

    # Creating the DT
    tree = DecisionTree(
        criterion=split_criterion,
        stopping_criteria=stopping_criterion,
        param=stopping_param,
        mv_split=mv_split_strategy,
        mv_dis=mv_dis_strategy,
        mv_classif=mv_classif_strategy,
        pruning_method=None if pruning_method == "none" else pruning_method.upper(),
        pruning_param=None if pruning_method == "none" else pruning_param
    )

    return tree

iris= load_iris()
X, y = iris.data, iris.target
# Introduce missing values 
X[1, 2] = np.nan
X[3, 2] = np.nan
X[5, 1] = np.nan
X[10, 0] = np.nan
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

indiv = generate_individual()
tree = create_tree(indiv)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
acc_rep = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc_rep:.4f}")