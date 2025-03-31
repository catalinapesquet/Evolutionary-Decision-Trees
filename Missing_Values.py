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
        self.split_criteria = SplitCriterion()
    
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
        """
        Simply ignore instances where the attribute value is missing when evaluating the split criterion.
        """
        # Select instances without missing values
        not_missing_mask = X[feature].notna()
        X_filtered = X[not_missing_mask].copy()
        y_filtered = y[not_missing_mask].copy()
        return X_filtered, y_filtered, X[feature]
    
    # Gene 1: Impute Missing Values with mode or mean
    def impute_mv_split(self, X, y, feature):
        """ 
        Impute missing values with mode or mean.
        """
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
    def weight_split(self, X, y, feature, criterion_type):
        # Calculate the proportion of missing values
        proportion_missing = X[feature].isnull().mean()
        # Calculate the criterion using the SplitCriteria class
        criterion_value = self.split_criteria.calculate(X, y, feature)
        # Adjust the criterion value by the proportion of missing values
        adjusted_value = criterion_value * (1 - proportion_missing)
        return adjusted_value
    
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
            return self.ignore_all_dis(X, y, feature, split_value)
        elif self.mv_distrib == 'most_common':
            return self.impute_mv_dis(X, y, feature, split_value)
        elif self.mv_distrib == 'class_specific_common':
            return self.impute_mv_class_dis(X, y, feature, split_value)
        elif self.mv_distrib == 'assign_all':
            return self.assign_all_dis(X, y, feature, split_value)
        elif self.mv_distrib == 'largest_partition':
            return self.largest_part_dis(X, y, feature, split_value)
        elif self.mv_distrib == 'weight_dis':
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
        # Split the instances based on the split value
        left_mask = X[feature] <= split_value
        right_mask = X[feature] > split_value
        left_split = (X[left_mask], y[left_mask])
        right_split = (X[right_mask], y[right_mask])
        # Assign instances with missing values to both partitions
        missing_mask = X[feature].isnull()
        X_missing = X[missing_mask]
        y_missing = y[missing_mask]
        left_split = (
            pd.concat([left_split[0], X_missing]),
            pd.concat([left_split[1], y_missing])
        )
        right_split = (
            pd.concat([right_split[0], X_missing]),
            pd.concat([right_split[1], y_missing]))
        
        return left_split, right_split

    def largest_part_dis(self, X, y, feature, split_value):
        """
        Assigns instances with missing values to the partition with the most instances.
        """
        # Split the instances based on the split value
        left_mask = X[feature] <= split_value
        right_mask = X[feature] > split_value
        left_split = (X[left_mask], y[left_mask])
        right_split = (X[right_mask], y[right_mask])
        # Determine the largest partition
        if left_split[0].shape[0] >= right_split[0].shape[0]:
            largest_split = left_split
        else:
            largest_split = right_split
        # Assign instances with missing values to the largest partition
        missing_mask = X[feature].isnull()
        X_missing = X[missing_mask]
        y_missing = y[missing_mask]
        largest_split = (
            pd.concat([largest_split[0], X_missing]),
            pd.concat([largest_split[1], y_missing])
        )
        return largest_split, (pd.DataFrame(), pd.Series())  # Return empty for the other partition

    def weight_dis(self, X, y, feature, split_value):
        """
        Distributes instances with missing values based on the probability of each partition.
        """
        # Split the instances based on the split value
        left_mask = X[feature] <= split_value
        right_mask = X[feature] > split_value
        left_split = (X[left_mask], y[left_mask])
        right_split = (X[right_mask], y[right_mask])
        # Calculate the probability of each partition
        total = X.shape[0]
        left_prob = left_split[0].shape[0] / total
        right_prob = right_split[0].shape[0] / total
        # Assign instances with missing values based on partition probability
        missing_mask = X[feature].isnull()
        X_missing = X[missing_mask]
        y_missing = y[missing_mask]
        left_split = (
            pd.concat([left_split[0], X_missing.sample(frac=left_prob)]),
            pd.concat([left_split[1], y_missing.sample(frac=left_prob)])
        )
        right_split = (
            pd.concat([right_split[0], X_missing.sample(frac=right_prob)]),
            pd.concat([right_split[1], y_missing.sample(frac=right_prob)])
        )
        return left_split, right_split

    def most_probable_partition(self, X, y, feature, split_value):
        """
        Assigns instances with missing values to the partition that is most probable considering the class.
        """
        # Split the instances based on the split value
        left_mask = X[feature] <= split_value
        right_mask = X[feature] > split_value
        left_split = (X[left_mask], y[left_mask])
        right_split = (X[right_mask], y[right_mask])
        # Determine the largest partition
        if left_split[0].shape[0] >= right_split[0].shape[0]:
            largest_split = left_split
        else:
            largest_split = right_split
        # Assign instances with missing values to the largest partition
        missing_mask = X[feature].isnull()
        X_missing = X[missing_mask]
        y_missing = y[missing_mask]
        largest_split = (
            pd.concat([largest_split[0], X_missing]),
            pd.concat([largest_split[1], y_missing])
        )
        return largest_split, (pd.DataFrame(), pd.Series())  # Return empty for the other partition

    def _split_helper(self, X_col, split_threshold):
        left_indices = np.argwhere(X_col <= split_threshold).flatten()
        right_indices = np.argwhere(X_col > split_threshold).flatten()
        return left_indices, right_indices
    
        
    
        
        