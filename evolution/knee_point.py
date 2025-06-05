# -*- coding: utf-8 -*-
"""
Created on Thu May 22 17:16:28 2025

@author: Aurora
"""
import numpy as np

def knee_point(res, objectives):
    n_obj = len(objectives)

    W = np.full(n_obj, 1.0 / n_obj)

    U = [np.dot(W, obj_vec) for obj_vec in res.F]

    best_idx = np.argmin(U)
    return res.X[best_idx], best_idx
