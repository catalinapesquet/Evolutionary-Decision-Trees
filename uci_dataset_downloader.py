# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 14:46:52 2025

@author: Catalina
"""

import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split

# Liste des datasets UCI utilisés dans HEAD-DT / MOHEAD-DT
UCI_DATASETS = {
    "iris": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    "abalone": "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data",
    "anneal": "https://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.data",
    "arrhythmia": "https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data",
    "audiology": "https://archive.ics.uci.edu/ml/machine-learning-databases/audiology/audiology.data",
    "car": "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
    "glass": "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
    "hepatitis": "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data",
    "segment": "https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data",
    "semeion": "https://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data",
    "shuttle_landing": "https://archive.ics.uci.edu/ml/machine-learning-databases/shuttle-landing-control/shuttle-landing-control.data",
    "winequality-red": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
    "winequality-white": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv",

    # ➕ Tu peux ajouter ici d'autres datasets de Table 1
}

DATA_DIR = "datasets"
os.makedirs(DATA_DIR, exist_ok=True)

def download_dataset(name, url):
    print(f"Downloading {name}...")
    response = requests.get(url)
    if response.status_code == 200:
        file_ext = url.split("/")[-1].split(".")[-1]
        filename = os.path.join(DATA_DIR, f"{name}.{file_ext}")
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Saved to {filename}\n")
    else:
        print(f"Failed to download {name}. Status code: {response.status_code}")

def parse_csv_dataset(path, header=None):
    df = pd.read_csv(path, header=header)
    return df

def load_and_split_dataset(name, label_col=-1):
    file_path = next((os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.startswith(name)), None)
    if file_path is None:
        raise FileNotFoundError(f"File for dataset '{name}' not found.")

    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path, sep=";")  # winequality uses ;
    else:
        df = pd.read_csv(file_path, header=None)

    X = df.iloc[:, :-1]
    y = df.iloc[:, label_col]
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.3, random_state=42, stratify=y_train_val)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

if __name__ == "__main__":
    for name, url in UCI_DATASETS.items():
        download_dataset(name, url)

    # Exemple : charger et découper le dataset iris
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_and_split_dataset("iris")
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)