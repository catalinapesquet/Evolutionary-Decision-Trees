# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 11:28:46 2025

@author: Aurora
"""
# LOAD DATA 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
 
# Reload dataset with proper delimiter
df = pd.read_csv("C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\DT\\Play(Sheet1).csv", sep=";")

le = LabelEncoder()
df["Weather"] = le.fit_transform(df["Weather"])
df["Temp"] = le.fit_transform(df["Temp"])
df["Play"] = df["Play"].str.strip()
df["Play_binary"] = df["Play"].map({"No":0, "Yes":1})

# Split into features and target
X = df[["Weather", "Temp"]].to_numpy()
y = df["Play_binary"].to_numpy().astype(int)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# import ace_tools as tools; tools.display_dataframe_to_user(name="Training Data", dataframe=X_train.join(y_train))
# X_train.shape, X_test.shape

# DEFINE PROBLEM
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
            print("Evaluating x :", x.tolist())
            tree = decode(x.tolist())
 
            metrics = evaluate_tree(tree, self.X_train, self.y_train, self.X_test, self.y_test)
            print(f'metrics = {metrics}')
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

algorithm = NSGA2(pop_size=50,
                  sampling=ChoiceRandomSampling(),
                  crossover=SinglePointCrossover(prob=0.9),
                  mutation=ChoiceRandomMutation(prob=0.3),
                  eliminate_duplicates=True)
problem = TreeProblem(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    objectives=["f1", "n_nodes"]  
)

termination = get_termination("n_gen", 4)

res = minimize(problem,
               algorithm,
               termination,
               seed=40,
               verbose=True,
               save_history=True)

# PLOTS
import matplotlib.pyplot as plt
import numpy as np
from pymoo.indicators.hv import HV

# 1. Extract Pareto fronts from each generation
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
    sc = ax.scatter(front[:, 0], front[:, 1], c=front[:, 1], cmap='viridis', s=20)
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


# 3. Plot only the 4 first generations
gens_to_plot = list(range(0, min(4, len(fronts_per_gen))))

fig, axes = plt.subplots(1, len(gens_to_plot), figsize=(5 * len(gens_to_plot), 4))

for idx, gen_idx in enumerate(gens_to_plot):
    front = fronts_per_gen[gen_idx]
    ax = axes[idx]
    sc = ax.scatter(front[:, 0], front[:, 1], c=front[:, 1], cmap='viridis', s=20)
    ax.set_title(f"Generation {gen_idx+1}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("F1 Score")
    ax.grid(True)

fig.suptitle("First 4 Generations (2D Pareto Front)", fontsize=20)
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

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = axes.flatten()

for idx, gen_idx in enumerate(gens_to_plot):
    front = fronts_per_gen[gen_idx]
    ax = axes[idx]
    sc = ax.scatter(front[:, 0], front[:, 1], c=front[:, 1], cmap='viridis', s=20)
    ax.set_title(f"Generation {gen_idx+1}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("F1 Score")
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

