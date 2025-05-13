# -*- coding: utf-8 -*-
"""
Created on Thu May  8 11:26:10 2025

@author: Aurora
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

le = LabelEncoder()

def extract_data(dataset):
    
    X = None
    y = None

    if dataset == "outlook":
        path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\Play(Sheet1).csv"
        df = pd.read_csv(path, sep=";")
        df["Weather"] = le.fit_transform(df["Weather"])
        df["Temp"] = le.fit_transform(df["Temp"])
        df["Play"] = df["Play"].str.strip()
        df["Play_binary"] = df["Play"].map({"No":0, "Yes":1})

        # Split into features and target
        X = df[["Weather", "Temp"]].to_numpy()
        y = df["Play_binary"].to_numpy().astype(int)
        
    elif dataset == "abalone":
        path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\abalone.data"
        df = pd.read_csv(path, sep=",",  header=None)
        
        # Name columns
        df.columns = ["Sex", "Length", "Diameter", "Height", "Whole_weight", 
                      "Shucked_weight", "Viscera_weight", "Shell_weight", "Rings"]
        # Encode categorial values
        df["Sex_encoded"] = le.fit_transform(df["Sex"])
        
        # Split into features and target
        X = df[["Sex_encoded", "Length", "Diameter", "Height", 
                "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight"]].to_numpy()
        y = df["Rings"].to_numpy().astype(int)
        
    elif dataset == "anneal":
        path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\anneal.data"
        df = pd.read_csv(path, sep=",")
        
        # Name columns
        df.columns = ["Family", "Product Type", "Steel", "Carbon", "Hardness", #ok
                      "Temper rolling", "Condition", "Formability", "Strenght", #ok
                      "Non-ageing", "Surface-finish","Surface-quality", #ok
                      "Enamelability", "Bc","Bf", "Bt", "Bw/me","Bl","M","Chrom", #ok
                      "Phos", "Cbond", "Marvi","Exptl", "Ferro", "Corr", #ok
                      "Blue/Bright/Varn/Clean", "Lustre", "Jurofm", "S", "P", #ok
                      "Shape","Thick","Width","Len","Oil","Bore","Packing",
                      "Classes"] 
        # Encode categorial values
        categorical_cols = [
            "Family", "Product Type", "Steel", "Temper rolling", "Condition", "Formability",
            "Non-ageing", "Surface-finish", "Surface-quality", "Enamelability", "Bc", "Bf",
            "Bt", "Bw/me", "Bl", "M", "Chrom", "Phos", "Cbond", "Marvi", "Exptl", "Ferro",
            "Corr", "Blue/Bright/Varn/Clean", "Lustre", "Jurofm", "S", "P", "Shape", "Oil",
            "Bore", "Packing", "Classes"
        ]
        for col in categorical_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        
        # Split into features and target
        X = df.drop(columns=["Classes"]).to_numpy()
        y = df["Classes"].to_numpy().astype(int)
        
    elif dataset == "arrhythmia":
        path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\arrhythmia.data"
        df = pd.read_csv(path, sep=",", header=None)
        
        # Name columns
        columns = [
            "Age", "Sex", "Height", "Weight", "QRS duration", "P-R interval", "Q-T interval",
            "T interval", "P interval",
            "QRS angle", "T angle", "P angle", "QRST angle", "J angle",
            "Heart rate",
        ]
        # to generate repetitive names of columns
        def wave_cols(prefix, start):
            labels = ["Q", "R", "S", "R'", "S'", "Nb deflections",
                      "R ragged", "R diphasic", "P ragged", "P diphasic", "T ragged", "T diphasic"]
            return [f"{prefix} {label}" for label in labels]
        
        def amplitude_cols(prefix):
            labels = ["JJ", "Q", "R", "S", "R'", "S'", "P", "T", "QRSA", "QRSTA"]
            return [f"{prefix} {label} amplitude" for label in labels]
        
        channels = ["DI", "DII", "DIII", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        
        for ch in channels:
            columns.extend(wave_cols(ch, start=len(columns)))
        
        for ch in channels:
            columns.extend(amplitude_cols(ch))
        columns.append("Classes")
        df.columns = columns
        
        # Encode categorial values
        non_numeric_cols = ['T angle', 'P angle', 'QRST angle', 'J angle', 'Heart rate']
        
        for col in non_numeric_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('[^\d.-]', '', regex=True), errors='coerce')

        # Split into features and target
        X = df.drop(columns=["Classes"]).to_numpy()
        y = df["Classes"].to_numpy().astype(int)
        counts = pd.Series(y).value_counts()
        valid_classes = counts[counts >= 2].index
        mask = np.isin(y, valid_classes)
        
        X = X[mask]
        y = y[mask]
         
    elif dataset == "audiology":
        path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\audiology.data"
        raw_df = pd.read_csv(path, sep=";", header=None)
        
        # Nettoyage sur raw_df, pas sur df
        raw_df.dropna(axis=1, how='all', inplace=True)  # retire colonnes totalement vides
        raw_df.dropna(axis=0, how='all', inplace=True)  # retire lignes totalement vides
        
        # Ensuite
        # Suppose that the last column is the target
        target_col = raw_df.columns[-1]
        
        # Encode toutes les colonnes
        le = LabelEncoder()
        for col in raw_df.columns:
            raw_df[col] = le.fit_transform(raw_df[col].astype(str))
        
        # Séparer features et target
        X = raw_df.drop(columns=[target_col]).to_numpy()
        y = raw_df[target_col].to_numpy().astype(int)
        
        # Nettoyage supplémentaire : enlever classes rares
        counts = pd.Series(y).value_counts()
        valid_classes = counts[counts >= 2].index
        mask = np.isin(y, valid_classes)
        
        X = X[mask]
        y = y[mask]
        
        print(f"✅ Dataset 'audiology' chargé : {X.shape[0]} exemples après nettoyage.")

    elif dataset == "car":
        path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\car.data"
        df = pd.read_csv(path, sep=";")
        
    elif dataset == "glass":
        path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\glass.data"
        df = pd.read_csv(path, sep=";")
        
    elif dataset == "hepatitis":
        path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\hepatitis.data"
        df = pd.read_csv(path, sep=";")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# extract_data("arrhythmia")
