# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 14:14:37 2025

@author: Catalina
"""
import pandas as pd
import numpy as np
from Split_Criteria import SplitCriterion

class MissingValues:
    def __init__(self, mv_split='ignore_all', mv_distrib='ignore_all'):
        self.mv_split = mv_split
        self.mv_distrib = mv_distrib
    
    def handle_split(self, X, y, feature):
        if self.mv_split == "ignore_all":
            return self.ignore_all_split(X, y, feature)
        elif self.mv_split == "impute_mv":
            return self.impute_mv_split(X, y, feature)
        elif self.mv_split == "weight_split":
            return self.weight_split(X, y, feature)
        elif self.mv_split == "impute_mv_class":
            return self.impute_mv_class_split(X, y, feature)
        else:
            raise ValueError(f"Unsupported criterion: {self.mv_split}")
    
    # HANDLING MISSING VALUES DURING SPLIT CRITERION EVALUATION
    # Gene 0: Ignore All instances with missing values
    def ignore_all_split(self, X, y, feature):
        # Select instances without missing values
        not_missing_mask = X[feature].notna()
        X_filtered = X[not_missing_mask].copy()
        y_filtered = y[not_missing_mask].copy()
        return X_filtered, y_filtered, X[feature]
    
    # Gene 1: Impute Missing Values with mode or mean
    def impute_mv_split(self, X, y, feature):
        X_copy = X.copy()
        # Check if values are numerical or categorical
        if pd.api.types.is_numeric_dtype(X_copy[feature]):
            # Mean if numerical
            fill_value = X_copy[feature].mean()
        else:
            # Mode if categorical
            fill_value = X_copy[feature].mode().iloc[0]  
    
        X_copy[feature] = X_copy[feature].fillna(fill_value)
        return X_copy, y, X_copy[feature]

    # Gene 2: Weight Splitting Criterion
    def weight_split(self, X, y, feature):
        # Calculate the proportion of each known value
        known_values = X[feature].dropna()
        value_counts = known_values.value_counts(normalize=True)
        # Distribute missing values proportionally
        missing_mask = X[feature].isna()
        X_copy = X.copy()
        for value, proportion in value_counts.items():
            missing_indices = X_copy[missing_mask].index
            num_missing = len(missing_indices)
            num_to_fill = int(num_missing * proportion)
            fill_indices = missing_indices[:num_to_fill]
            X_copy.loc[fill_indices, feature] = value
            missing_mask.loc[fill_indices] = False  # Update mask to reflect filled values

        return X_copy, y, X_copy[feature]
    
    def calculate_weighted_impurity(self, y, criterion='gini', weights=None):
        if weights is None:
            weights = np.ones(len(y)) / len(y)
        else:
            weights = np.array(weights) / np.sum(weights)

        class_counts = pd.Series(y).value_counts()
        impurity = 0
        for cls, count in class_counts.items():
            proportion = weights[y == cls].sum()
            if proportion > 0:
                if criterion == 'gini':
                    impurity += proportion * (1 - proportion)
                elif criterion == 'entropy':
                    impurity -= proportion * np.log2(proportion)
        return impurity

    def calculate_weighted_information_gain(self, X, y, feature, criterion='gini'):
        known_mask = ~X[feature].isna()
        known_indices = X[known_mask].index
        missing_indices = X[~known_mask].index

        if known_indices.empty:
            return 0

        parent_impurity = self.calculate_weighted_impurity(y, criterion=criterion)
        children_impurity = 0
        value_counts = X[feature].value_counts(normalize=True)

        for value, proportion in value_counts.items():
            subset_indices = X[X[feature] == value].index
            subset_y = y.loc[subset_indices]
            children_impurity += proportion * self.calculate_weighted_impurity(subset_y, criterion=criterion)

        gain = parent_impurity - children_impurity
        return gain
    
    # Gene 3: Impute Missing Values with mode/mean of instances of same class
    def impute_mv_class_split(self, X, y, feature):
        X_copy = X.copy()
        
        # Loop through each instance with missing value
        for idx in X_copy[X_copy[feature].isna()].index:
            label = y[idx]
            # Select instances of same class without missing value
            same_class_mask = (y == label) & X_copy[feature].notna()
            same_class_values = X_copy.loc[same_class_mask, feature]
            
            # Handle case where same_class_mask is empty
            if len(same_class_values)==0:
                # Check if values are numerical or categorical
                if pd.api.types.is_numeric_dtype(X_copy[feature]):
                    fill_value = X_copy[feature].mean()
                else:
                    fill_value = X_copy[feature].mode().iloc[0]
            else :
                # Check if values are numerical or categorical
                if pd.api.types.is_numeric_dtype(X_copy[feature]):
                    fill_value = same_class_values.mean()
                else:
                    fill_value = same_class_values.mode().iloc[0]
            # fill missing values
            X_copy.at[idx, feature] = fill_value
        
        return X_copy, y, X_copy[feature]
    
    # HANDLING MISSING VALUES DURING DISTRIBUTION
    # How the distribution is done after we find the best split 
    def apply_split_with_mv(self, X, y, feature, split_value):
        if self.mv_distrib == 'ignore_all':
            return self.ignore_instance(X, y, feature, split_value)
        elif self.mv_distrib == 'most_common':
            return self.most_common_value(X, y, feature, split_value)
        elif self.mv_distrib == 'class_specific_common':
            return self.class_specific_common_value(X, y, feature, split_value)
        elif self.mv_distrib == 'assign_all':
            return self.assign_to_all_partitions(X, y, feature, split_value)
        elif self.mv_distrib == 'largest_partition':
            return self.largest_partition(X, y, feature, split_value)
        elif self.mv_distrib == 'partition_probability':
            return self.partition_probability(X, y, feature, split_value)
        elif self.mv_distrib == 'most_probable_partition':
            return self.most_probable_partition(X, y, feature, split_value)
        else:
            raise ValueError("Invalid distribution strategy specified.")
    
    # Gene 0: Ignoring all
    def ignore_all_dis(self, X, y, feature, split_value):
        # Filter out instances with missing values in the feature
        mask = X[feature].notnull()
        X_filtered = X[mask]
        y_filtered = y[mask]
    
        # Split the filtered instances based on the split value
        left_mask = X_filtered[feature] <= split_value
        right_mask = X_filtered[feature] > split_value
    
        left_split = (X_filtered[left_mask], y_filtered[left_mask])
        right_split = (X_filtered[right_mask], y_filtered[right_mask])
    
        return left_split, right_split
    
    # Gene 1: Impute Missing Values with mode or mean
    def impute_mv_dis(self, X, y, feature, split_value):
        # Determine the most common value for the feature
        # Check if values are numerical or categorical
        if X[feature].dtype == 'object':  
            most_common = X[feature].mode()[0]
        else: 
            most_common = X[feature].mean()
        # Fill missing values with the most common value
        X_filled = X.copy()
        X_filled[feature].fillna(most_common, inplace=True)
        # Split the instances based on the split value
        left_mask = X_filled[feature] <= split_value
        right_mask = X_filled[feature] > split_value

        left_split = (X_filled[left_mask], y[left_mask])
        right_split = (X_filled[right_mask], y[right_mask])
        return left_split, right_split
    
    # Gene 2: Impute Missing Values with mode/mean of instances of same class
    def impute_mv_class_dis(self, X, y, feature, split_value):
        # Determine the most common value for the feature within each class
        if X[feature].dtype == 'object':  # Categorical feature
            most_common_by_class = X.groupby(y)[feature].agg(lambda x: x.mode()[0])
        else:  # Numerical feature
            most_common_by_class = X.groupby(y)[feature].mean()
        # Fill missing values with the class-specific most common value
        X_filled = X.copy()
        for class_label in most_common_by_class.index:
            mask = (y == class_label) & X_filled[feature].isnull()
            X_filled.loc[mask, feature] = most_common_by_class[class_label]
        # Split the instances based on the split value
        left_mask = X_filled[feature] <= split_value
        right_mask = X_filled[feature] > split_value
    
        left_split = (X_filled[left_mask], y[left_mask])
        right_split = (X_filled[right_mask], y[right_mask])
        return left_split, right_split

    def assign_all_dis(self, X, y, feature, split_value):
        """
        Distributes instances with missing values to both left and right partitions.
        """
        # Implement logic to assign to all partitions
        pass

    def largest_part_dis(self, X, y, feature, split_value):
        """
        Assigns instances with missing values to the partition with the most instances.
        """
        # Implement logic to assign to the largest partition
        pass

    def probability_dis(self, X, y, feature, split_value):
        # Implement logic to weight by partition probability
        pass

    def most_probable_partition(self, X, y, feature, split_value):
        # Implement logic to assign to the most probable partition
        pass

    def _split_helper(self, X_col, split_threshold):
        left_indices = np.argwhere(X_col <= split_threshold).flatten()
        right_indices = np.argwhere(X_col > split_threshold).flatten()
        return left_indices, right_indices
    
        
    
        
        