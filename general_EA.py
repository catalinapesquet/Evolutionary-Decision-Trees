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

# PARAMETERS
dataset_names = ["audiology", "car", "glass"]
pop_size = 20 
n_gen = 20

datasets = [extract_data(name) for name in dataset_names]
    
class MultiDatasetProblem(ElementwiseProblem):
    def __init__(self, datasets, objectives):
        self.datasets = datasets
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
    
    def _evaluate(self, x, out, *args, **kwargs):
        try:
            # Decode tree from genotype
            # print("Evaluating x :", x.tolist())
            tree = decode(x.tolist())
            
            all_metrics = []
            # pour chaque dataset evaluer l'individu 
            for (X_train, X_test, y_train, y_test) in self.datasets:
                met = evaluate_tree(tree, X_train, y_train, X_test, y_test)
                all_metrics.append(met)
                
            # Extract chosen objectives
            averaged_values = []
            # iterating through every objective
            for obj in self.selected_objectives:
                obj_values = []
                # iterating through every dataset
                for met in all_metrics:
                    val = met[obj]
                    if obj in ['f1', 'recall', 'specificity']:  # à maximiser
                        val = 1 - val  # inversion pour minimisation
                    obj_values.append(val)
                avg_val = np.mean(obj_values)
                averaged_values.append(avg_val)
    
            out["F"] = averaged_values
            
        except Exception as e:
            print(f"Error for x = {x} : {e}", flush=True)
            out["F"] = [1.0 for _ in self.selected_objectives]  # valeur de pénalité

algorithm = NSGA2(pop_size=pop_size,
                  sampling=ChoiceRandomSampling(),
                  crossover=SinglePointCrossover(prob=0.9),
                  mutation=ChoiceRandomMutation(prob=0.3),
                  eliminate_duplicates=True)

problem = MultiDatasetProblem(
    datasets=datasets,
    objectives=["recall", "specificty"]  
)

termination = get_termination("n_gen", n_gen)

res = minimize(problem,
               algorithm,
               termination,
               seed=40,
               verbose=True,
               save_history=True)

# PLOTS
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV

# Extract Pareto fronts from each generation
fronts_per_gen = []
X_per_gen = []
for algo in res.history:
    if hasattr(algo, 'pop') and algo.pop is not None:
        fronts_per_gen.append(algo.pop.get("F"))
        X_per_gen.append(algo.pop.get("X"))

print(f"Number of extracted Pareto fronts: {len(fronts_per_gen)}")

# 2. Plot Pareto fronts every 10 generations (2D plot)
step = 10  # Plot every 10 generations
gens_to_plot = list(range(0, len(fronts_per_gen), step))

n_cols = 4
n_rows = int(np.ceil(len(gens_to_plot) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for idx, gen_idx in enumerate(gens_to_plot):
    front = fronts_per_gen[gen_idx]
    ax = axes[idx]
    sc = ax.scatter(front[:, 0], front[:, 1], color='red', s=20)
    ax.set_title(f"Generation {gen_idx+1}")
    ax.set_xlabel("f1")
    ax.set_ylabel("nodes")
    ax.grid(True)

# Remove empty plots
for j in range(len(gens_to_plot), len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Pareto Fronts across generations (2 objectives)", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# Plot only the 4 first generations
gens_to_plot = list(range(0, min(4, len(fronts_per_gen))))

fig, axes = plt.subplots(1, len(gens_to_plot), figsize=(5 * len(gens_to_plot), 4))

for idx, gen_idx in enumerate(gens_to_plot):
    front = fronts_per_gen[gen_idx]
    ax = axes[idx]
    sc = ax.scatter(front[:, 0], front[:, 1], color='r', s=20)
    ax.set_title(f"Generation {gen_idx+1}")
    ax.set_xlabel("F1 Score")
    ax.set_ylabel("Nodes")
    ax.grid(True)

fig.suptitle("First 4 Generations (2D Pareto Front)", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# Compute and plot Hypervolume evolution
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


# Plot the first 10 generations if available
gens_to_plot = list(range(min(10, len(fronts_per_gen))))

n_cols = 5
n_rows = int(np.ceil(len(gens_to_plot) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for idx, gen_idx in enumerate(gens_to_plot):
    front = fronts_per_gen[gen_idx]
    ax = axes[idx]
    sc = ax.scatter(front[:, 0], front[:, 1], color='r', s=20)
    ax.set_title(f"Generation {gen_idx+1}")
    ax.set_xlabel("1-F1")
    ax.set_ylabel("Nmbr of nodes")
    ax.grid(True)

# Remove empty plots
for j in range(len(gens_to_plot), len(axes)):
    fig.delaxes(axes[j])
    
fig.suptitle("First 10 Generations (2D Pareto Front)", fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

mean_nodes = np.mean(res.F[:,1])
mean_f1 = np.mean(res.F[:,0])

print(f"mean nodes : {mean_nodes}")
print(f"mean f1 : {mean_f1}")

def knee_point(res, W=[0.5, 0.5]):
    U = [np.dot(W, obj_vec) for obj_vec in res.F]
    best_idx = np.argmin(U)

    return res.X[best_idx], best_idx



