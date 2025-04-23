# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 15:26:37 2025

@author: Catalina
"""

from pymoo.core.problem import ElementwiseProblem
from encode_decode import decode
from metrics import evaluate_tree 
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.operators.mutation.rm import ChoiceRandomMutation
from pymoo.core.variable import Choice
from choice_sampling import ChoiceRandomSampling

class TreeProblem(ElementwiseProblem):
    def __init__(self, X_train, y_train, X_test, y_test, objectives):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.selected_objectives = objectives
        
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
        super().__init__(vars=vars, n_obj=3)
    
    def _evaluate(self, x, out, *args, **kwargs):
        try:
            # Decode tree from genotype
            # print("Évaluation de :", x.tolist())
            tree = decode(x.tolist())
 
            metrics = evaluate_tree(tree, self.X_train, self.y_train, self.X_test, self.y_test)
            # print(f'metrics = {metrics}')
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

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(X, y, train_size=0.9)
X_train, X_test, y_train, y_test = train_test_split(X_meta_train, y_meta_train, train_size=0.7)
problem = TreeProblem(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    objectives=["recall", "f1", "n_nodes"]  
)



termination = get_termination("n_gen", 20)

choices = [
    list(range(13)),  # SPLIT_CRITERIA
    list(range(5)),   # STOPPING_CRITERIA
    list(range(101)),  # STOPPING_PARAM
    list(range(4)),   # MV_SPLIT
    list(range(7)),   # MV_DISTRIBUTION
    list(range(2)),   # MV_CLASSIF
    list(range(6)),   # PRUNING_STRATEGY
    list(range(101))  # PRUNING_PARAM
]

algorithm = NSGA2(pop_size=100,
                  sampling=ChoiceRandomSampling(),
                  crossover=UniformCrossover(),
                  mutation=ChoiceRandomMutation(),
                  eliminate_duplicates=True)

res = minimize(problem,
               algorithm,
               termination,
               seed=40,
               verbose=True)
