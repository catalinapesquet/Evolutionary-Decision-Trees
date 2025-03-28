# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 14:14:37 2025

@author: Catalina
"""

class MissingValues:
    def __init__(self, mv_split):
        self.mv_split = mv_split
    
    def handle_split(self, mv_split, X, y, feature):
        if mv_split == "ignore_all":
            return self.ignore_all(X, y, feature)
        elif mv_split == "impute_mv":
            return self.impute_mv(X, y, feature)
        elif mv_split == "weight_split":
            return self.weight_split(X, y, feature)
        elif mv_split == "impute_mv_class":
            return self.impute_mv_class(X, y, feature)
        else:
            raise ValueError(f"Unsupported criterion: {self.mv_split}")
            
    # Gene 0: Ignore All instances with missing values
    def ignore_all(self, X, y, feature):
        mask = X[feature].notna()
        return X[mask], y[mask]
    
    # Gene 1: Impute Missing Values with mode or mean 
    def impute_mv(self, y):
        return None
    
    # Gene 2: Weight Splitting Criterion
    def weight_split(self, y):
        return None
    
    # Gene 3: Impute Missing Values with mode/mean of instances of same class
    def impute_mv_class(self, y):
        return None
    
    
    
        
        