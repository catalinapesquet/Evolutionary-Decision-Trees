# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 12:17:19 2025

@author: Catalina
"""
import numpy as np
class StoppingCriterion:
    def __init__(self, n_tot_samples, criterion, param=2):
        self.criterion = criterion
        self.param = param

    # Gene 0: Reaching homogeneity
    def check_homogeneity(self, y):
        return len(np.unique(y)) == 1  # Only one unique class left
    
    # Gene 1: Reaching Maximum Depth
    def check_max_depth(self, depth):
        mapped_param = (self.param * 0.08) + 2
        return depth >= mapped_param
    
    # Gene 2: Reaching Minimum Number of Instances in non terminal node
    def check_min_samples_split(self,y):
        mapped_param = (self.param * 0.19) + 1
        return len(y) < mapped_param
    
    # Gene 3: Reaching Minimum Percentage of Instances in non terminal node
    def check_min_portion_split(self, y, n_tot_samples):
        # Map from [0, 100] to [1,10]
        mapped_param = (self.param * 0.09) + 1
        return len(y) / n_tot_samples < mapped_param  # Compare to total number of instances
    
    # Gene 4: Reaching a Predictive Accuracy within a Node
    def check_predictive_accuracy(self, y):
        # Map from [0, 100] to {70, 75, 80, 85, 90, 95, 100}
        mapped_param = (self.param % 7) * 5 + 70 
        # Identify majority class
        majority_class = np.argmax(np.bincount(y))
        # Calculate predictive accuracy
        predictive_accuracy = np.sum(y == majority_class) / len(y)
        return predictive_accuracy >= mapped_param
    
    # Check if stopping criterion is observed
    def stop(self, n_tot_samples, y, depth):
        if self.criterion == 'homogeneity':
            if self.check_homogeneity(y):
                return True
        elif self.criterion == 'max_depth': 
            if self.check_max_depth(depth):
                return True
        elif self.criterion == 'min_samples_split':
            if self.check_min_samples_split(y):
                return True
        elif self.criterion == 'min_portion_split':
            if self.check_min_portion_split(y, n_tot_samples):
                return True
        elif self.criterion == 'predictive_accuracy': 
            if self.check_predictive_accuracy(y):
                return True
        return False