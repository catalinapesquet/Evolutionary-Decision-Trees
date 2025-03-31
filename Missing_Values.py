# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 14:14:37 2025

@author: Catalina
"""
import pandas as pd
import numpy as np

class MissingValues:
    def __init__(self, mv_split):
        self.mv_split = mv_split
    
    def handle_split(self, X, y, feature):
        if self.mv_split == "ignore_all":
            return self.ignore_all(X, y, feature)
        elif self.mv_split == "impute_mv":
            return self.impute_mv(X, y, feature)
        elif self.mv_split == "weight_split":
            return self.weight_split(X, y, feature)
        elif self.mv_split == "impute_mv_class":
            return self.impute_mv_class(X, y, feature)
        else:
            raise ValueError(f"Unsupported criterion: {self.mv_split}")
            
    # Gene 0: Ignore All instances with missing values
    def ignore_all(self, X, y, feature):
        # Select instances without missing values
        not_missing_mask = X[feature].notna()
        X_filtered = X[not_missing_mask].copy()
        y_filtered = y[not_missing_mask].copy()
        return X_filtered, y_filtered, X[feature]
    
    # Gene 1: Impute Missing Values with mode or mean
    def impute_mv(self, X, y, feature):
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
    
    def calculate_weighted_impurity(self, y, criterion='entropy', weights=None):
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

    def calculate_weighted_information_gain(self, X, y, feature, criterion='entropy'):
        # Convert y to pandas Series if it's not already
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        # Indices des valeurs connues et manquantes pour la fonctionnalité
        known_mask = ~X[feature].isna()
        known_indices = X[known_mask].index
        missing_indices = X[~known_mask].index
        if known_indices.empty:
            return 0
        # Calcul de l'impureté de l'ensemble courant
        parent_impurity = self.calculate_weighted_impurity(y, criterion=criterion)
        # Calcul de l'impureté pour chaque valeur connue de la fonctionnalité
        children_impurity = 0
        value_counts = X[feature].value_counts(normalize=True)
    
        for value, proportion in value_counts.items():
            subset_indices = X[X[feature] == value].index
            subset_y = y.loc[subset_indices]
            children_impurity += proportion * self.calculate_weighted_impurity(subset_y, criterion=criterion)
        # Prise en compte des instances avec des valeurs manquantes lors du calcul du gain
        gain = parent_impurity - children_impurity
        return gain

    
    # Gene 3: Impute Missing Values with mode/mean of instances of same class
    def impute_mv_class(self, X, y, feature):
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
    
    def apply_split_with_mv(self, X_column, threshold, mv_split):
        if mv_split == "ignore_all":
            left_idxs_not_missing, right_idxs_not_missing = self._split_helper(X_column[X_column.notna()].to_numpy(), threshold)
            original_indices_not_missing = np.where(X_column.notna())
            left_idxs = original_indices_not_missing[left_idxs_not_missing]
            right_idxs = original_indices_not_missing[right_idxs_not_missing]
            missing_indices = np.where(X_column.isna())
            combined_left_idxs = np.unique(np.concatenate((left_idxs, missing_indices)))
            combined_right_idxs = np.unique(np.concatenate((right_idxs, missing_indices)))
            return combined_left_idxs, combined_right_idxs
        elif mv_split in ["impute_mv", "impute_mv_class", "weight_split"]:
            left_idxs, right_idxs = self._split_helper(X_column.to_numpy(), threshold)
            return left_idxs, right_idxs
        else:
            raise ValueError(f"Unsupported mv_split: {mv_split}")

    def _split_helper(self, X_col, split_threshold):
        left_indices = np.argwhere(X_col <= split_threshold).flatten()
        right_indices = np.argwhere(X_col > split_threshold).flatten()
        return left_indices, right_indices
    
        
    
        
        