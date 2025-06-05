# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 14:14:37 2025

@author: Catalina
"""
import pandas as pd
import numpy as np
from collections import defaultdict
import random

random.seed(42)
np.random.seed(42)

class MissingValues:
    def __init__(self, mv_split='ignore_all', mv_distrib='ignore_all', mv_classif='explore_all', split_criterion=None):
        self.mv_split = mv_split
        self.mv_distrib = mv_distrib
        self.mv_classif = mv_classif
        self.split_criteria = split_criterion
    
    # HANDLING MISSING VALUES DURING SPLIT CRITERION EVALUATION
    def handle_split(self, X, y, feature):
        feature = self._standardize_feature_name(feature) ####
        if self.mv_split == "ignore_all":
            return self.ignore_all_split(X, y, feature)
        elif self.mv_split == "impute_mv":
            return self.impute_mv_split(X, y, feature)
        # elif self.mv_split == "weight_split":
        #     return self.weight_split(X, y, feature)
        elif self.mv_split == "impute_mv_class":
            return self.impute_mv_class_split(X, y, feature)
        else:
            raise ValueError(f"Unsupported criterion: {self.mv_split}")

    def _standardize_feature_name(self, feature):
        if isinstance(feature, str) and feature.isdigit():
            return int(feature)
        return feature
    
    # Gene 0: Ignore All instances with missing values
    def ignore_all_split(self, X, y, feature):
        """
        Simply ignore instances where the attribute value is missing when evaluating the split criterion.
        """
        # Select instances without missing values
        feature = self._standardize_feature_name(feature)
        not_missing_mask = X[feature].notna()
        X_filtered = X[not_missing_mask].copy()
        y_filtered = y[not_missing_mask].copy()
        return X_filtered, y_filtered, X_filtered[feature]

    
    # Gene 1: Impute Missing Values with mode or mean
    def impute_mv_split(self, X, y, feature):
        """ 
        Impute missing values with mode or mean regardless of their class.
        """
        feature = self._standardize_feature_name(feature)
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
    def weight_split(self, X, y, feature, threshold):
        feature = self._standardize_feature_name(feature)
        if isinstance(feature, str) and feature.isdigit():
            feature = int(feature)
    
        proportion_missing = X[feature].isnull().mean()
    
        # Filter nan values in y to calculate split criterion
        mask_valid = ~X[feature].isnull()
        X_temp = X[mask_valid]
        y_temp = pd.Series(y)[mask_valid]
    
        # Erase nan values in y
        final_mask = ~y_temp.isnull()
        X_clean = X_temp[final_mask]
        y_clean = y_temp[final_mask]
        # print("👀 y_clean dans weight_split:", y_clean)
        
        X_column = X_clean[feature]

        gain = self.split_criteria.calculate(y_clean.to_numpy(), X_column.to_numpy(), threshold)
        adjusted_gain = gain * (1 - proportion_missing)
    
        return adjusted_gain

    # Gene 3: Impute Missing Values with mode/mean of instances of same class
    def impute_mv_class_split(self, X, y, feature):
        """ 
        Impute missing values with mode or mean of isntances of same class.
        """
        feature = self._standardize_feature_name(feature)
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
            return self.weight_dis(X, y, feature, split_value)
        elif self.mv_distrib == 'most_probable_partition':
            return self.most_probable_partition(X, y, feature, split_value)
        else:
            raise ValueError("Invalid distribution strategy specified.")
    
    # Gene 0: Ignoring all
    def ignore_all_dis(self, X, y, feature, split_value):
        feature = self._standardize_feature_name(feature)
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
        """
        Impute missing values with mode/mean regardless of their class
        """
        # Determine the most common value for the feature
        # Check if values are numerical or categorical
        feature = self._standardize_feature_name(feature)
        if X[feature].dtype == 'object':  
            most_common = X[feature].mode()[0]
        else: 
            most_common = X[feature].mean()
        # Fill missing values with the most common value
        X_filled = X.copy()
        X_filled[feature] = X_filled[feature].fillna(most_common)
        # Split the instances based on the split value
        left_mask = X_filled[feature] <= split_value
        right_mask = X_filled[feature] > split_value

        left_split = (X_filled[left_mask], y[left_mask])
        right_split = (X_filled[right_mask], y[right_mask])
        return left_split, right_split
    
    # Gene 2: Impute Missing Values with mode/mean of instances of same class
    def impute_mv_class_dis(self, X, y, feature, split_value):
        """
        Impute missing values with mode/mean of instances of same class.
        """
        feature = self._standardize_feature_name(feature)
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
    
    # Gene 3: Assign to all nodes
    def assign_all_dis(self, X, y, feature, split_value):
        """
        Distributes instances with missing values to both left and right partitions.
        """
        feature = self._standardize_feature_name(feature)
        # Split the instances based on the split value
        left_mask = X[feature] <= split_value
        right_mask = X[feature] > split_value
    
        # Convert y to Series if needed
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
    
        left_split = (X[left_mask], y_series[left_mask])
        right_split = (X[right_mask], y_series[right_mask])
    
        # Assign instances with missing values to both partitions
        missing_mask = X[feature].isnull()
        X_missing = X[missing_mask]
        y_missing = y_series[missing_mask]
    
        left_split = (
            pd.concat([left_split[0], X_missing], ignore_index=True),
            pd.concat([left_split[1], y_missing], ignore_index=True)
        )
    
        right_split = (
            pd.concat([right_split[0], X_missing], ignore_index=True),
            pd.concat([right_split[1], y_missing], ignore_index=True)
        )
        return left_split, right_split

    # Gene 4: Assign to largest partition
    def largest_part_dis(self, X, y, feature, split_value):
        """
        Assigns instances with missing values to the partition with the most instances.
        """
        feature = self._standardize_feature_name(feature)
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
        # Ensure all data are converted to pandas DataFrame or Series types for concatenation 
        X_part, y_part = largest_split
        if isinstance(X_part, np.ndarray):
            X_part = pd.DataFrame(X_part, index=range(len(X_part)))
        if isinstance(y_part, np.ndarray):
            y_part = pd.Series(y_part, index=range(len(y_part)))
        if isinstance(y_missing, np.ndarray):
            y_missing = pd.Series(y_missing, index=X_missing.index)
    
        largest_split = (
            pd.concat([X_part, X_missing]),
            pd.concat([y_part, y_missing])
        )
        return largest_split, (pd.DataFrame(), pd.Series())
    
    # Gene 5: assign based on probability of each partition
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
        y_missing = pd.Series(y[missing_mask], index=X_missing.index)
        
        # Convert to Series if necessary for compatibility with .concat
        if isinstance(left_split[1], np.ndarray):
            left_split = (
                left_split[0],
                pd.Series(left_split[1], index=left_split[0].index)
            )
        if isinstance(right_split[1], np.ndarray):
            right_split = (
                right_split[0],
                pd.Series(right_split[1], index=right_split[0].index)
            )     
        left_split = (
            pd.concat([left_split[0], X_missing.sample(frac=left_prob, random_state=42)]),
            pd.concat([left_split[1], y_missing.sample(frac=left_prob, random_state=42)])
        )
        
        right_split = (
            pd.concat([right_split[0], X_missing.sample(frac=right_prob, random_state=42)]),
            pd.concat([right_split[1], y_missing.sample(frac=right_prob, random_state=42)])
        )
        return left_split, right_split
    
    # Gene 6: Assign to most probable partition
    def most_probable_partition(self, X, y, feature, split_value):
        """
        Assigns instances with missing values to the partition that is most probable considering the class.
        """
        feature = self._standardize_feature_name(feature)
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
        if isinstance(largest_split[1], np.ndarray):
            largest_split = (
                largest_split[0],
                pd.Series(largest_split[1], index=largest_split[0].index)
            )
        if isinstance(y_missing, np.ndarray):
            y_missing = pd.Series(y_missing, index=X[feature].isnull()[X[feature].isnull()].index)
        largest_split = (
            pd.concat([largest_split[0], X_missing]),
            pd.concat([largest_split[1], y_missing])
        )
        return largest_split, (pd.DataFrame(), pd.Series())  # Return empty for the other partition

    def _split_helper(self, X_col, split_threshold):
        left_indices = np.argwhere(X_col <= split_threshold).flatten()
        right_indices = np.argwhere(X_col > split_threshold).flatten()
        return left_indices, right_indices
    
    # HANDLING MISSING VALUES DURING CLASSIFICATION
    # Gene 0: Explore all
    def predict_explore_all(self, inputs, node, weight=1.0):
        """
        Recursively explores all branches when missing values are encountered.
        Returns a dictionary with class votes weighted by path probabilities.
        """
        # If node is a leaf, return prediction with its weight
        if node.is_leaf:
            return {node.leaf_value: weight}
    
        # If there is a missing value
        if np.isnan(inputs[node.feat_idx]):
            # 🔧 On choisit un poids égal (50/50), ou on pourrait utiliser des ratios appris
            left_ratio = node.left_weight
            right_ratio = node.right_weight
    
            # Recursively explore all branches 
            left_votes = self.predict_explore_all(inputs, node.left, weight * left_ratio)
            right_votes = self.predict_explore_all(inputs, node.right, weight * right_ratio)
    
            # Combine results in a dictionnary
            combined_votes = defaultdict(float)
            for cls, w in left_votes.items():
                combined_votes[cls] += w
            for cls, w in right_votes.items():
                combined_votes[cls] += w
            return combined_votes
        
        # Else we follow normal path
        elif inputs[node.feat_idx] <= node.threshold:
            return self.predict_explore_all(inputs, node.left, weight)
        else:
            return self.predict_explore_all(inputs, node.right, weight)
        
    # Gene 1: Most probable path
    def _predict_most_probable(self, inputs, node):
        """
        Take the rout to the most probable partition (largest subset)
        """
        # While not a leaf
        while node.left:
            val = inputs[node.feat_idx]
            # If there is missing value
            if np.isnan(val):
                # Follow largest branch
                if node.left_weight >= node.right_weight:
                    node = node.left
                else:
                    node = node.right
            elif val <= node.threshold:
                node = node.left
            else:
                node = node.right
        
        return node.leaf_value if node.is_leaf else node.majority_class
    
    # # Gene 2: Assign to majority class of node 
    # def _predict_stop_and_vote(self, inputs, node):
    #     """
    #     Halt the classification process and assign the instance to the majority class of node
    #     """
    #     val = inputs[node.feat_idx]
    #     # Check if val is an array and handle it accordingly
    #     if isinstance(val, np.ndarray):
    #         if np.isnan(val).any():
    #             return node.majority_class # Halt and return the majority class of the current node
    #     elif np.isnan(val):
    #         return node.majority_class  # Halt and return the majority class of the current node
    #     elif val <= node.threshold:
    #         return self._predict_stop_and_vote(inputs, node.left)  # Traverse left
    #     else:
    #         return self._predict_stop_and_vote(inputs, node.right)  # Traverse right

    
