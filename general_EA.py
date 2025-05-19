# -*- coding: utf-8 -*-
"""
Created on Mon May 19 16:25:46 2025

@author: Aurora
"""

from Extract_data import extract_data
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from choice_sampling import ChoiceRandomSampling
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.mutation.rm import ChoiceRandomMutation
from pymoo.optimize import minimize
from pymoo.core.variable import Choice
from pymoo.termination import get_termination
from encode_decode import decode
from metrics import evaluate_tree

import numpy as np

datasets = ["iris", "anneal", "audiology"]
X_train, y_train, X_test, y_test = [], [], [], []

data = [[] for _ in datasets]

for i in range(len(datasets)):
    data[i].append(X_train, y_train, X_test, y_test)
    
class TreeProblem(ElementwiseProblem):
    def __init__(self, X_train, y_train, X_test, y_test, objectives):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.selected_objectives = objectives
        
        # How to create an individual
        vars = {
                    "gene_0": Choice(options=np.arange(13)),  # split criteria
                    "gene_1": Choice(options=np.arange(5)),   # stopping criteria
                    "gene_2": Choice(options=np.arange(101)), # stopping parameter
                    "gene_3": Choice(options=np.arange(4)),   # missing value split 
                    "gene_4": Choice(options=np.arange(7)),   # missing value distribution
                    "gene_5": Choice(options=np.arange(2)),   # missing value classification
                    "gene_6": Choice(options=np.arange(6)),   # pruning strategy
                    "gene_7": Choice(options=np.arange(101))  # pruning parameter
                }
        super().__init__(vars=vars, n_obj=2)
    
    def _evaluate_specific(self, x, out, *args, **kwargs):
        try:
            # Decode tree from genotype
            # print("Evaluating x :", x.tolist())
            tree = decode(x.tolist())
 
            metrics = evaluate_tree(tree, self.X_train, self.y_train, self.X_test, self.y_test)
            # print("Nodes :", metrics["n_nodes"])
            # Extract chosen objectives
            values = []
            for obj in self.selected_objectives:
                val = metrics[obj]
                # Check if the metrics is to be maximized 
                if obj in ['f1', 'recall', 'specificity']:
                    values.append(1 - val) # need to maximise those
                else: 
                    values.append(val) # need to minimize number of nodes
            
            # Objectives to minimize
            out["F"] = values
            
        except Exception as e:
            print(f"Error for x = {x} : {e}", flush=True)
            # In case of fail in the construction of the tree
            out["F"] = [1.0, 999]
