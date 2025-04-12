#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Imported all the core libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn Components
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# For feature selection
from sklearn.feature_selection import RFE

# Ignore warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Set a seed for reproducibility
np.random.seed(42)





# ************************************

# 1. Load dataset

# ************************************

def load_data():
    # Load and combine datasets 
    red = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", delimiter=';')
    white = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", delimiter=';')
    
    red['wine_type'], white['wine_type'] = 'red', 'white'
    combined = pd.concat([red, white]).reset_index(drop=True)
    
    print("Red Wine Shape:", red.shape)
    print("White Wine Shape:", white.shape)
    print("Combined Dataset Shape:", combined.shape)
    return red, white, combined


# ************************************

# 2. Exploratory Data Analysis (EDA)

# ************************************


def analyze_data(df, name):
    """
    Exploratory analysis includes:
      - Summary statistics.
      - Histogram of quality scores.
      - Correlation heatmap (excluding non-numeric columns).
    """
    print(f"\n--- EDA for {name} ---")
    print("\nSummary Statistics:\n", df.describe())
    print("\nMissing Values:\n", df.isnull().sum())
    
    plt.figure(figsize=(8, 5))
    sns.histplot(df['quality'], bins=7, kde=True, color='skyblue')
    plt.title(f"Quality Distribution ({name})")
    plt.show()
    
    numeric = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f"Correlation Heatmap ({name})")
    plt.show()



# ************************************

# 3. Data Preprocessing

# ************************************

def prepare_and_scale(df):
    
    """
    Converting wine quality into a binary label: 
      if quality >= 7, then 1 (Good) 
      otherwise 0 (Bad)
    """
    
    df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
    
    """
    Split the dataframe into features and target.
    Then perform a train-test split (70/30) using stratification.
    """
    
    X = df.drop(['quality', 'quality_label', 'wine_type'], axis=1)
    y = df['quality_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    scaler = StandardScaler()
    return (scaler.fit_transform(X_train), scaler.transform(X_test), 
            y_train, y_test, X.columns)

# ************************************

# 4. Model Training and Evaluation

# ************************************


def evaluate_model(model, X_train, X_test, y_train, y_test, cv=5):
    """
    Evaluate each model using Stratified K-Fold cross-validation and test performance.
    Prints the mean CV accuracy, test accuracy, and classification report.
    Returns a dictionary with detailed results.
    """
    cv_scores = cross_val_score(model, X_train, y_train, cv=StratifiedKFold(cv, shuffle=True, random_state=42))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        "CV Mean": np.mean(cv_scores),
        "CV Std": np.std(cv_scores),
        "Test Acc": accuracy_score(y_test, y_pred),
        "Report": classification_report(y_test, y_pred)
    }



# ************************************

# Main Execution

# ************************************

def main():
    # Load and analyze data
    red, white, combined = load_data()
    for df, name in [(red, "Red_Wine"), (white, "White_Wine"), (combined, "Combined_Wine")]:
        analyze_data(df, name)
        df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
    
    # Prepare data for each dataset
    datasets = {
        "Red Wine": prepare_and_scale(red),
        "White Wine": prepare_and_scale(white),
        "Combined": prepare_and_scale(combined)
    }
    
    # Models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "SVM": SVC(random_state=42, kernel='rbf', probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gaussian Naive Bayes": GaussianNB()
    }
    
    # Evaluating each models
    results = {}
    for dataset_name, (X_train, X_test, y_train, y_test, features) in datasets.items():
        print(f"\n=== {dataset_name} Evaluation ===")
        results[dataset_name] = {}
        for name, model in models.items():
            results[dataset_name][name] = evaluate_model(model, X_train, X_test, y_train, y_test)
            print(f"\n--- {name} ---")
            res = results[dataset_name][name]
            print(f"CV Accuracy: {res['CV Mean']:.4f} (±{res['CV Std']:.4f})")
            print(f"Test Accuracy: {res['Test Acc']:.4f}")
            print("Classification Report:\n", res["Report"])
    
    # Selecting feature on combined data
    X_train, X_test, y_train, y_test, features = datasets["Combined"]
    rfe = RFE(LogisticRegression(random_state=42, max_iter=1000), n_features_to_select=5).fit(X_train, y_train)
    X_train_rfe, X_test_rfe = rfe.transform(X_train), rfe.transform(X_test)
    
    print("\n--- Feature Selection ---")
    print("Selected Features:", features[rfe.support_].tolist())
    
    plt.figure(figsize=(10, 6))
    pd.Series(rfe.ranking_, index=features).sort_values().plot(kind='bar')
    plt.title("Feature Ranking by RFE")
    plt.ylabel("Ranking (1 = most important)")
    plt.show()
    
    # Evaluate with selected features
    print("\n=== Evaluation with Selected Features ===")
    selected_results = {}
    for name, model in models.items():
        selected_results[name] = evaluate_model(model, X_train_rfe, X_test_rfe, y_train, y_test)
        print(f"\n--- {name} ---")
        res = selected_results[name]
        print(f"CV Accuracy: {res['CV Mean']:.4f} (±{res['CV Std']:.4f})")
        print(f"Test Accuracy: {res['Test Acc']:.4f}")
    
    # Comparing performance
    print("\n=== Performance Comparison ===")
    for name in models:
        print(f"\n{name}:")
        print(f"All Features - CV: {results['Combined'][name]['CV Mean']:.4f}, Test: {results['Combined'][name]['Test Acc']:.4f}")
        print(f"Selected Features - CV: {selected_results[name]['CV Mean']:.4f}, Test: {selected_results[name]['Test Acc']:.4f}")
        print("-"*50)

if __name__ == "__main__":
    main()
