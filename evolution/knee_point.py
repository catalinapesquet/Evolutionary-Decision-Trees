# -*- coding: utf-8 -*-
"""
Created on Thu May 22 17:16:28 2025

@author: Aurora
"""
import numpy as np

def knee_point(res, objectives, return_all=False, strategy='first'):
    n_obj = len(objectives)
    W = np.full(n_obj, 1.0 / n_obj)
    U = np.dot(res.F, W)

    min_value = np.min(U)
    best_indices = np.where(U == min_value)[0]

    if return_all:
        return [res.X[idx] for idx in best_indices], best_indices
    elif strategy == 'random':
        idx = np.random.choice(best_indices)
    else:  # default to 'first'
        idx = best_indices[0]

    return res.X[idx], idx
