# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 12:58:12 2025

@author: Catalina
"""

import numpy as np
import matplotlib.pyplot as plt

# Créer 10 points aléatoires dans [0,10] pour f1 et f2
points = [np.random.uniform(0, 10, size=2) for _ in range(30)]

def is_dominated(p, q):
    return all(p_i >= q_i for p_i, q_i in zip(p, q)) and any(p_i > q_i for p_i, q_i in zip(p, q))

def get_non_dominated(points):
    non_dominated = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i != j and is_dominated(p, q):
                dominated = True
                break
        if not dominated:
            non_dominated.append(p)
    return non_dominated

def pareto_layers(points):
    layers = []
    remaining = points[:]
    while remaining:
        front = get_non_dominated(remaining)
        layers.append(front)
        # Supprimer les points du front de remaining
        remaining = [p for p in remaining if tuple(p) not in [tuple(f) for f in front]]
    return layers

import matplotlib.pyplot as plt
layers = pareto_layers(points)
colors = ['red', 'blue', 'green', 'orange', 'purple']

plt.figure(figsize=(8,6))
colors = ['blue', 'green', 'orange', 'purple', 'red', 'brown', 'cyan', 'lime']

for i, front in enumerate(layers):
    color = colors[i % len(colors)]
    front = np.array(front)
    plt.scatter(front[:,0], front[:,1], color=color, label=f'Layer {i+1}')
    
    # Relier les points par ligne (optionnel)
    sorted_front = sorted(front, key=lambda x: x[0])
    x_vals = [pt[0] for pt in sorted_front]
    y_vals = [pt[1] for pt in sorted_front]
    plt.plot(x_vals, y_vals, color=color, linewidth=0.5, alpha=0.7)

plt.xlabel('f1')
plt.ylabel('f2')
plt.title('Pareto Layers')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()

