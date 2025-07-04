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
import matplotlib.pyplot as plt
from pymoo.indicators.hv import HV

import datetime
from collections import Counter
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# PARAMETERS
dataset_names = ["winequality-red", "car", "segment"] # dermatology
pop_size = 100
n_gen = 100
objectives = ["specificity", "recall", "n_nodes"]

algorithm = NSGA2(pop_size=pop_size,
                  sampling=ChoiceRandomSampling(),
                  crossover=SinglePointCrossover(prob=1),
                  mutation=ChoiceRandomMutation(prob=0.1),
                  eliminate_duplicates=True)

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
        super().__init__(vars=vars, n_obj=len(objectives))
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluates one individual (tree configuration) by:
        1. Decoding the genotype.
        2. Training and evaluating the tree on each dataset.
        3. Averaging the objective values (recall, specificity, n_nodes).
        4. Inverting metrics (1 - value) to allow minimization.
        
        On failure, assigns penalty values.
        """
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
                    if obj in ['f1', 'recall', 'specificity']:  # to maximise
                        val = 1 - val  # inversion for minimisation
                    obj_values.append(val)
                avg_val = np.mean(obj_values)
                averaged_values.append(avg_val)
    
            out["F"] = averaged_values
            
        except Exception as e:
            print(f"Error for x = {x} : {e}", flush=True)
            out["F"] = [1.0 for _ in self.selected_objectives]  # valeur de pénalité

problem = MultiDatasetProblem(
    datasets=datasets,
    objectives=objectives
)

termination = get_termination("n_gen", n_gen)

res = minimize(problem,
               algorithm,
               termination,
               seed=40,
               verbose=True,
               save_history=True)

# PLOTS

# Extract Pareto fronts from each generation
fronts_per_gen = []
X_per_gen = []
for algo in res.history:
    if hasattr(algo, 'pop') and algo.pop is not None:
        fronts_per_gen.append(algo.pop.get("F"))
        X_per_gen.append(algo.pop.get("X"))
    
print(f"Number of extracted Pareto fronts: {len(fronts_per_gen)}")

if fronts_per_gen[0].shape[1] == 2:
    # Plot Pareto fronts every 10 generations (2D plot)
    step = 10  # Plot every 10 generations
    gens_to_plot = list(range(0, len(fronts_per_gen), step))
    
    n_cols = 4
    n_rows = int(np.ceil(len(gens_to_plot) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, gen_idx in enumerate(gens_to_plot):
        front = fronts_per_gen[gen_idx]
        ax = axes[idx]
        sc = ax.scatter(front[:, 0], front[:, 1], color='mediumaquamarine', s=20)
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
        sc = ax.scatter(front[:, 0], front[:, 1], color='mediumaquamarine', s=20)
        ax.set_title(f"Generation {gen_idx+1}")
        ax.set_xlabel("F1 Score")
        ax.set_ylabel("Nodes")
        ax.grid(True)
    
    fig.suptitle("First 4 Generations (2D Pareto Front)", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = f"4gen_2obj_{timestamp}.png"
    plt.savefig(f"C:/Users/Aurora/Desktop/DecisionTreesEA/DT/log/{filename}", dpi=300, bbox_inches='tight')
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
        sc = ax.scatter(front[:, 0], front[:, 1], color='mediumaquamarine', s=20)
        ax.set_title(f"Generation {gen_idx+1}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("F1 Score")
        ax.grid(True)
    
    # Remove empty plots
    for j in range(len(gens_to_plot), len(axes)):
        fig.delaxes(axes[j])
        
    
    fig.suptitle("First 10 Generations (2D Pareto Front)", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = f"plot_2obj_{timestamp}.png"
    plt.savefig(f"C:/Users/Aurora/Desktop/DecisionTreesEA/DT/log/{filename}", dpi=300, bbox_inches='tight')
    plt.show()

# 3 Objectives
if fronts_per_gen[0].shape[1] == 3:
    step = 10  # Affiche 1 génération sur 5
    gens_to_plot = list(range(0, len(fronts_per_gen), step))

    n_cols = 5
    n_rows = int(np.ceil(len(gens_to_plot) / n_cols))

    fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))

    for idx, gen_idx in enumerate(gens_to_plot):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        front = fronts_per_gen[gen_idx]
        ax.scatter(front[:, 0], front[:, 1], front[:, 2], color='teal', s=20)
        ax.set_title(f"Generation {gen_idx+1}")
        ax.set_xlabel(objectives[0])
        ax.set_ylabel(objectives[1])
        ax.set_zlabel(objectives[2])
        ax.grid(True)

    fig.suptitle("3D Pareto Fronts across Generations (3 Objectives)", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Compute and plot Hypervolume evolution
# Find the reference point
all_points = np.vstack(fronts_per_gen)
ref_point = np.max(all_points, axis=0) + 0.05  # Safety margin

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
filename = f"HV_2obj_{timestamp}.png"
plt.savefig(f"C:/Users/Aurora/Desktop/DecisionTreesEA/DT/log/{filename}", dpi=300, bbox_inches='tight')
plt.show()

# SAVING RESULTS

mean_nodes = np.mean(res.F[:,1])
mean_f1 = np.mean(res.F[:,0])

filename = f"log/results_2obj_{timestamp}.txt"
with open(filename, "w") as f:
    f.write(f"Solutions:\n{res.X}\n")
    f.write(f"Objective functions results:\n{res.F}\n")
    f.write(f"Mean number of nodes:\n{mean_nodes}\n")
    f.write(f"Mean f1:\n{mean_f1}\n")


population = res.X  # ou np.array([...])
gene_titles = [
    "Splitting criteria",    # gene[0]
    "Stopping criteria",     # gene[1]
    "Stopping parameter",    # gene[2]
    "Mv Split",              # gene[3]
    "Mv_Distrib",            # gene[4]
    "Mv Class",              # gene[5]
    "Pruning",               # gene[6]
    "Pruning param"          # gene[7]
]
filename = f"log/gene_statistics_{timestamp}.txt"
with open(filename, "w") as f:
    for i, title in enumerate(gene_titles):
        f.write(f"{title}:\n")
        values = [indiv[i] for indiv in population]
        counter = Counter(values)
        for key, count in sorted(counter.items()):
            f.write(f"{key}: {count} times\n")
        f.write("\n")


# SAVE RESULTS
for i in range(res.F.shape[1]):
    mean_val = np.mean(res.F[:, i])
    print(f"mean Objective {i+1} : {mean_val}")


def knee_point(res, objectives):
    n_obj = len(objectives)

    W = np.full(n_obj, 1.0 / n_obj)

    U = [np.dot(W, obj_vec) for obj_vec in res.F]

    best_idx = np.argmin(U)
    return res.X[best_idx], best_idx