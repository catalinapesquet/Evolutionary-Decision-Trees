# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 17:42:31 2025

@author: Catalina
"""
# Genes

SPLIT_CRITERIA = {
    0: "information_gain",
    1: "gini",
    # 2: "global_mutual",  NOT READY
    2: "g_stat",
    3: "mantaras",
    4: "hg_distribution",
    5: "chv_criterion",
    6: "dcsm",
    7: "chi_square",
    8: "mpi",
    # 10: "norm_gain", # NOT READY
    9: "ort",
    10: "twoing",
    11: "cair",
    12: "gain_ratio"
}

BINARY_SPLIT = {
    0: "binary",
    1: "multi-way" # NOT READY
    }

STOPPING_CRITERIA = {
    0: "homogeneity",
    1: "max_depth",
    2: "min_samples_split",
    3: "min_portion_split",
    4: "predictive_accuracy"
    }

MV_SPLIT = {
    0: "ignore_all",
    1: "impute_mv",
    2: "weight_split",
    3: "impute_mv_class"
    }

MV_DISTRIBUTION = {
    0: "ignore_all",
    1: "most_common",
    2: "class_specific_common",
    3: "assign_all",
    4: "largest_partition",
    5: "weight_dis",
    6: "most_probable_partition"
    }

MV_CLASSIFICATION = {
    0: "explore_all",
    1: "most_probable_path",
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

