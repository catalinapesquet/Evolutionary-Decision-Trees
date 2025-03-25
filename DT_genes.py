# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 17:42:31 2025

@author: Catalina
"""
# Genes

SPLIT_CRITERIA = {
    0: "entropy",
    1: "gain_ratio",
    2: "gini",
    3: "symmetrical_uncertainty",
    4: "g_statistic",
    5: "chi_squared",
    6: "gain_ratio_p",
    7: "kullback_leibler",
    8: "hellinger_distance",
    9: "mdlp",
    10: "distance_measure",
    11: "modified_gini",
    12: "intrinsic_value",
    13: "weighted_information_gain",
    14: "log_gain"
}

BINARY_SPLIT = {
    0: "binary",
    1: "multi-way"
    }

STOPPING_CRITERIA = {
    0: "class_homogeneity",
    1: "max_tree_depth",
    2: "min_num_instances_node",
    3: "min_percentage_instances_node",
    4: "predictive_accuracy"
    }

MV_SPLIT = {
    0: "ignore_all",
    1: "impute_mode_mean",
    2: "weighted_by_proportion",
    3: "impute_mode_mean_same_class"
    }

MV_DISTRIBUTION = {
    0: "ignore_all",
    1: "impute_mode_mean",
    2: "impute_mode_mean_same_class",
    3: "assign_all_partition",
    4: "assign_large_partition",
    5: "weighted_by_probability",
    6: "weighted_by_probability_class"
    }

MV_CLASSIFICATION = {
    0: "combine_results",
    1: "most_probable",
    2: "assign_to_majority",
    }

PRUNNING_GENES = {
    0: "none",
    1: "rep",
    2: "pep",
    3: "mep",
    4: "ccp",
    5: "ebp"
    }

def interpret_stopping_parameter(index: int, method: str):
    if method == "min_instances":
        return max(1, index % 10)
    elif method == "max_depth":
        return max(2, index % 9 + 2)
    elif method == "no_gain":
        return index / 100.0  # seuil
    elif method == "error_threshold":
        return index / 100.0
    else:
        return None  # homogénéité ne nécessite pas de paramètre

def interpret_pruning_parameter(index: int, method: str):
    return None