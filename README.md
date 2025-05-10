# 💼 Income Classification Model

A binary classification model built to predict whether an individual's income exceeds $50K/year based on demographic and employment data. Built using Neural Networks and Boosted Decision Trees in **Azure ML Studio**, with data preprocessing in Python and SQL.

---

## 📌 Project Overview

This project aims to support economic segmentation and decision-making by predicting income levels using features such as education, work class, hours worked per week, and age. It combines SQL-based preprocessing with classical ML and deep learning techniques.

---

## 🔧 Tech Stack

- **Languages:** Python, SQL
- **Platforms:** Azure ML Studio, Jupyter Notebook
- **Libraries:** Scikit-learn, matplotlib, seaborn
- **ML Models:** Neural Networks, Boosted Decision Trees
- **Concepts:** EDA, Feature Engineering, Data Balancing

---

## 🧠 Key Features

- Achieved **87.2% accuracy** on the test set
- Identified top predictors: **education, age, and working hours**
- Balanced class distribution using oversampling
- Used **SQL** for filtering and joining large demographic datasets prior to modeling
- Compared classical ML with Neural Network performance

---

## 📊 EDA & Preprocessing

- Handled missing values and encoded categorical variables
- Performed correlation analysis to reduce multicollinearity
- Visualized feature distributions and class imbalance
- Used **oversampling** techniques to balance classes (income >50K vs ≤50K)

---

## 🚀 How to Run

1. Load the dataset (publicly available at [UCI Repository](https://archive.ics.uci.edu/ml/datasets/adult))
2. Run the notebook in Jupyter or Azure ML Studio
3. Modify paths or column names as needed

---

## 📈 Results

- Final accuracy: **87.2%**
- F1 score: 0.84
- Improved interpretability using feature importance and visualizations

---

## 📫 Contact

Built by **[Muzammil Tariq](https://www.linkedin.com/in/muzammiltariq95/)**  
Open to feedback, contributions, and collaboration.
