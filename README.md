# ERP-Ready Wine Quality Dashboard & Classifier

This project analyzes wine quality data using modern machine learning pipelines and ERP-style dashboard concepts. It includes exploratory analysis, classification model training, feature selection, and performance comparison.

## 📊 Features

- Combines red and white wine datasets
- Performs EDA: histograms, correlation heatmaps, summaries
- Binary classification (`quality >= 7` as "Good")
- Scales data and evaluates 6 classifiers:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - SVM
  - K-Nearest Neighbors
  - Gaussian Naive Bayes
- Uses Stratified K-Fold CV and test set evaluation
- Performs Recursive Feature Elimination (RFE)
- Outputs feature rankings and comparative results

## 📁 Dataset

UCI Wine Quality Dataset  
Download links used in code:
- [Red Wine](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv)
- [White Wine](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv)

## 🛠 How to Run

```bash
pip install -r requirements.txt
python wine_dashboard.py
```

## 📈 Output

- Plots for quality distribution and correlation matrix
- Feature importance chart from RFE
- Classification reports and accuracy comparisons

---

