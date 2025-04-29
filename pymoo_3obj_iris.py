# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 17:58:08 2025

@author: Catalina
"""
from pymoo.core.problem import ElementwiseProblem
from encode_decode import decode
from metrics import evaluate_tree 
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from choice_sampling import ChoiceRandomSampling
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.crossover.pntx import SinglePointCrossover
from pymoo.operators.mutation.rm import ChoiceRandomMutation
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.variable import Choice


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
        super().__init__(vars=vars, n_obj=2)
    
    def _evaluate(self, x, out, *args, **kwargs):
        try:
            # Decode tree from genotype
            # print("Evaluating x :", x.tolist())
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
    objectives=["recall", "f1",]  
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
                  crossover=SinglePointCrossover(prob=0.9),
                  mutation=ChoiceRandomMutation(prob=0.3),
                  eliminate_duplicates=True)

res = minimize(problem,
               algorithm,
               termination,
               seed=40,
               verbose=True,
               save_history=True)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pymoo.indicators.hv import HV

# 1. Extract Pareto fronts from each generation
fronts_per_gen = []
for algo in res.history:
    if hasattr(algo, 'pop') and algo.pop is not None:
        fronts_per_gen.append(algo.pop.get("F"))

print(f"Number of extracted Pareto fronts: {len(fronts_per_gen)}")

# 2. Plot Pareto fronts every 10 generations
step = 10  # Plot every 10 generations
gens_to_plot = list(range(0, len(fronts_per_gen), step))

n_cols = 4
n_rows = int(np.ceil(len(gens_to_plot) / n_cols))

fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

for idx, gen_idx in enumerate(gens_to_plot):
    front = fronts_per_gen[gen_idx]
    ax = fig.add_subplot(n_rows, n_cols, idx+1, projection='3d')
    sc = ax.scatter(front[:, 0], front[:, 1], front[:, 2], c=front[:, 2], cmap='viridis', s=20)
    ax.set_title(f"Generation {gen_idx+1}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("F1 Score")
    ax.set_zlabel("Number of Nodes")

fig.suptitle("3D Pareto Fronts (1 plot every 10 generations)", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 3. Plot only the 4 first generations
gens_to_plot = list(range(0, min(4, len(fronts_per_gen))))

n_cols = 4
n_rows = int(np.ceil(len(gens_to_plot) / n_cols))

fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

for idx, gen_idx in enumerate(gens_to_plot):
    front = fronts_per_gen[gen_idx]
    ax = fig.add_subplot(n_rows, n_cols, idx+1, projection='3d')
    sc = ax.scatter(front[:, 0], front[:, 1], front[:, 2], c=front[:, 2], cmap='viridis', s=20)
    ax.set_title(f"Generation {gen_idx+1}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("F1 Score")
    ax.set_zlabel("Number of Nodes")

fig.suptitle("3D Pareto Fronts - First 4 Generations", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 4. Compute and plot Hypervolume evolution
# Find the reference point
all_points = np.vstack(fronts_per_gen)
ref_point = np.max(all_points, axis=0) + 0.05  # Safety margin

print(f"Reference point for HV: {ref_point}")

# Compute hypervolume for each generation
hv_indicator = HV(ref_point=ref_point)
hypervolumes = [hv_indicator.do(front) for front in fronts_per_gen]

# Plot hypervolume over generations
plt.figure(figsize=(10,6))
plt.plot(range(1, len(hypervolumes)+1), hypervolumes, marker='o', linestyle='-', color='royalblue')
plt.xlabel('Generation')
plt.ylabel('Hypervolume')
plt.title('Hypervolume Evolution Across Generations')
plt.grid(True)
plt.show()

# 5. Plot the first 10 generations if available
gens_to_plot = list(range(min(10, len(fronts_per_gen))))

n_cols = 5
n_rows = int(np.ceil(len(gens_to_plot) / n_cols))

fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows))

for idx, gen_idx in enumerate(gens_to_plot):
    front = fronts_per_gen[gen_idx]
    ax = fig.add_subplot(n_rows, n_cols, idx+1, projection='3d')
    sc = ax.scatter(front[:, 0], front[:, 1], front[:, 2], c=front[:, 2], cmap='viridis', s=20)
    ax.set_title(f"Generation {gen_idx+1}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("F1 Score")
    ax.set_zlabel("Number of Nodes")

fig.suptitle("3D Pareto Fronts - First 10 Generations", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
