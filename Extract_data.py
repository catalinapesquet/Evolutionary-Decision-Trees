# -*- coding: utf-8 -*-
"""
Created on Thu May  8 11:26:10 2025

@author: Aurora
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

def extract_data(dataset):
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
        df = pd.read_csv(path, sep=";")
        
    elif dataset == "audiology":
        path = "C:\\Users\\Aurora\\Desktop\\DecisionTreesEA\\dataset\\datasets\\audiology.data"
        df = pd.read_csv(path, sep=";")
        
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
        