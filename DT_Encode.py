# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 17:24:53 2025

@author: Catalina
"""

from typing import List
from sklearn.tree import DecisionTreeClassifier
from DT_genes import SPLIT_CRITERIA, BINARY_SPLIT, STOPPING_CRITERIA, MV_SPLIT, MV_DISTRIBUTION, MV_CLASSIFICATION, PRUNNING_GENES
from DT_genes import interpret_stopping_parameter, interpret_pruning_parameter

class DecisionTreeChromosome:
    def __init__(self, chromosome: List[str]):
        self.chromosome = chromosome
        self._validate_genes()
        self.decoded = self.decode()
        self.clf = self.build_classifier()
        
    def _validate_genes(self):
        assert len(self.chromosome) == 9 # a chromosome must have 9 genes
        
    def decode(self):
        decoded = {}
        chromosome = self.chromosome
        
        decoded["split_criterion"] = SPLIT_CRITERIA[chromosome[0]]
        decoded["split_type"] = BINARY_SPLIT[chromosome[1]]
        
        stop_crit = STOPPING_CRITERIA[chromosome[2]]
        decoded["stopping_criterion"] = stop_crit
        decoded["stopping_parameter"] = interpret_stopping_parameter(chromosome[3], stop_crit)
        
        prune_crit = PRUNNING_GENES[chromosome[4]]
        decoded["pruning_method"] = prune_crit
        decoded["pruning_parameter"] = interpret_pruning_parameter(chromosome[5], prune_crit)
        
        decoded["mv_split"] = MV_SPLIT[chromosome[6]]
        decoded["mv_distribution"] = MV_DISTRIBUTION[chromosome[7]]
        decoded["mv_classification"] = MV_CLASSIFICATION[chromosome[8]]
        
        return decoded
    
    def build_classifier(self):
        params = {
            "criterion": self.decoded["split_criterion"],
            "random_state": 42
        }
        # Splitting criteria
        
        
        # Stopping criteria 
        if self.decoded["stopping_criterion"] == "max_depth":
            params["max_depth"] = self.decoded["stopping_param"]
        elif self.decoded["stopping_criterion"] == "min_samples_leaf":
            params["min_samples_leaf"] = self.decoded["stopping_param"]
            
        # Pruning
        if self.decoded["pruning_method"] == "ccp_alpha":
            params["ccp_alpha"] = self.decoded["pruning_param"]
        
        return DecisionTreeClassifier(**params)
            
    def __repr__(self):
        return f"DecisionTreeChromosome({self.genes})"
    
chromosomeX = [3, 0, 2, 1, 1, 0, 2, 1, 0]