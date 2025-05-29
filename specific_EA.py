# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 11:28:46 2025

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
from sklearn.model_selection import train_test_split
import os

import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
 
# PARAMETERS
dataset="abalone"
pop_size = 100
n_gen = 500
objectives = ["specificity", "recall", "n_nodes"] 

n_nodes_min, n_nodes_max = None, None
first_gen_bornes_done = False

# LOAD DATA 

# Split into meta training set / meta test set (80-20)
X_meta_train, X_meta_test, y_meta_train, y_meta_test = extract_data(dataset)


# DEFINE PROBLEM
class TreeProblem(ElementwiseProblem):
    def __init__(self, X_meta_train, y_meta_train, X_meta_test, y_meta_test, objectives):
        self.X_meta_train = X_meta_train
        self.y_meta_train = y_meta_train
        self.X_meta_test = X_meta_test
        self.y_meta_test = y_meta_test
        self.selected_objectives = objectives
        self.evaluation_counter = 0
        
        random_seed = np.random.randint(0, 100000)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_meta_train, self.y_meta_train, test_size=0.3, random_state=random_seed, stratify=self.y_meta_train)

        
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
        
    def resplit_data(self):
        random_seed = np.random.randint(0, 100000)  # New seed
        X_train_new, X_val, y_train_new, y_val = train_test_split(
            self.X_meta_train, self.y_meta_train, test_size=0.3, random_state=random_seed, stratify=self.y_meta_train)
        self.X_train = X_train_new
        self.X_test = X_val
        self.y_train = y_train_new
        self.y_test = y_val

    def _evaluate(self, x, out, *args, **kwargs):
        global n_nodes_min, n_nodes_max, first_gen_bornes_done
        try:
            self.evaluation_counter += 1
    
            # Decode tree from genotype
            tree = decode(x.tolist())
    
            metrics = evaluate_tree(tree, self.X_train, self.y_train, self.X_test, self.y_test)
            # print(f"Metrics : {metrics}")
    
            # Calculate node bounds
            if not first_gen_bornes_done:
                if 'n_nodes' in metrics:
                    val = metrics['n_nodes']
                    if n_nodes_min is None or val < n_nodes_min:
                        n_nodes_min = val
                    if n_nodes_max is None or val > n_nodes_max:
                        n_nodes_max = val
    
          
            if not first_gen_bornes_done and self.evaluation_counter >= pop_size:
                first_gen_bornes_done = True
    
            # Extract chosen objectives
            values = []
            for obj in self.selected_objectives:
                val = metrics[obj]
                if obj in ['f1', 'recall', 'specificity']:
                    val = 1 - val  # inverser pour minimiser
                elif obj == 'n_nodes' and n_nodes_min is not None and n_nodes_max is not None:
                    val = val
                values.append(val)
    
            out["F"] = values
    
        except Exception as e:
            print(f"Error for x = {x} : {e}", flush=True)
            # Cas d'erreur
            if len(objectives) == 2:
                out["F"] = [1.0, 999]
            elif len(objectives) == 3:
                out["F"] = [1.0, 999, 1.0]
                

algorithm = NSGA2(pop_size=pop_size,
                  sampling=ChoiceRandomSampling(),
                  crossover=SinglePointCrossover(prob=0.9),
                  mutation=ChoiceRandomMutation(prob=0.1),
                  eliminate_duplicates=True)
problem = TreeProblem(
        X_meta_train = X_meta_train,
        y_meta_train = y_meta_train,
        X_meta_test = X_meta_test,
        y_meta_test = y_meta_test,
        objectives=objectives 
)

termination = get_termination("n_gen", n_gen)

n_nodes_min, n_nodes_max = None, None

res = minimize(problem,
               algorithm,
               termination,
               seed=42,
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
    for gen_idx, algo in enumerate(res.history):
        problem.resplit_data()


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
        ax.set_xlabel(objectives[0])
        ax.set_ylabel(objectives[1])
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
        ax.set_xlabel(objectives[0])
        ax.set_ylabel(objectives[1])
        ax.grid(True)
    
    fig.suptitle("First 4 Generations (2D Pareto Front)", fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = f"4gen_2obj_{dataset}_{timestamp}.png"
    plt.savefig(f"C:/Users/Aurora/Desktop/DecisionTreesEA/DT/log_test/specific/{filename}", dpi=300, bbox_inches='tight')
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
    filename = f"plot_2obj_{dataset}_{timestamp}.png"
    plt.savefig(f"C:/Users/Aurora/Desktop/DecisionTreesEA/DT/log_test/specific/{filename}", dpi=300, bbox_inches='tight')
    plt.show()

# 3 Objectives
if fronts_per_gen[0].shape[1] == 3:
    step = 10  
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
    

    pareto_plot_filename = f"C:/Users/Aurora/Desktop/DecisionTreesEA/DT/log_test/specific/Pareto_3D_fronts_{dataset}_{timestamp}.png"
    fig.savefig(pareto_plot_filename, dpi=300, bbox_inches='tight')
    
    # Normalise n_nodes
    if n_nodes_min is not None and n_nodes_max is not None and (n_nodes_max != n_nodes_min):
        front[:, 2] = (front[:, 2] - n_nodes_min) / (n_nodes_max - n_nodes_min)
    else:
        print("Attention : bornes de normalisation non définies ou égales.")

    # Plot
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(front[:, 0], front[:, 1], front[:, 2], color='teal', s=20)
    ax.set_xlabel(objectives[0])
    ax.set_ylabel(objectives[1])
    ax.set_zlabel(f"{objectives[2]} (normalized)")
    ax.grid(True)
    fig.suptitle("3D Pareto Fronts Final Result (with normalized n_nodes)", fontsize=18)
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
plt.xscale("log")
# plt.yscale("log")
plt.xlabel('Generation')
plt.ylabel('Hypervolume')
plt.title('Hypervolume Evolution Across Generations')
plt.grid(True)
filename = f"HV_2obj_{dataset}_{timestamp}.png"
plt.savefig(f"C:/Users/Aurora/Desktop/DecisionTreesEA/DT/log_test/specific/{filename}", dpi=300, bbox_inches='tight')
plt.show()


# SAVING RESULTS
base_dir = r"C:\Users\Aurora\Desktop\DecisionTreesEA\DT\log_test\specific"
os.makedirs(base_dir, exist_ok=True)

filename = os.path.join(base_dir, f"results_2obj_{dataset}_{timestamp}.txt")

with open(filename, "w") as f:
    f.write(f"Solutions:\n{res.X}\n")
    f.write(f"Objective functions results:\n{res.F}\n")
    for i in range(res.F.shape[1]):
        mean_val = np.mean(res.F[:, i])
        f.write(f"mean Objective {i+1} : {mean_val}")

from collections import Counter
import datetime

population = res.X  

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

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = r"C:\Users\Aurora\Desktop\DecisionTreesEA\DT\log_test\specific"
os.makedirs(base_dir, exist_ok=True)

filename = os.path.join(base_dir, f"gene_statistics_{dataset}_{timestamp}.txt")

with open(filename, "w") as f:
    for i, title in enumerate(gene_titles):
        f.write(f"{title}:\n")
        values = [indiv[i] for indiv in population]
        counter = Counter(values)
        for key, count in sorted(counter.items()):
            f.write(f"{key}: {count} times\n")
        f.write("\n")
