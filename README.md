# 🍷 Wine Quality Prediction Web App

This repository contains a machine learning project to analyze and predict wine quality using various physicochemical attributes. The project includes **Exploratory Data Analysis (EDA)**, model training, and a **Flask-based web application** for real-time predictions.

---

## 📌 Problem Statement

### 1. Goal of the Data
The main objective of this dataset is to analyze and predict wine quality based on various physicochemical properties such as acidity, sugar levels, sulfur dioxide, pH, and alcohol content.

### 2. Description of Dataset
The datasets contain red and white variants of the Portuguese "Vinho Verde" wine, sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality). Due to privacy concerns, only physicochemical (input) and sensory (output) variables are available (e.g., no grape type, brand, or pricing info).

### 3. About the Data
The dataset includes features like:
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Quality (target)

---

## 📊 Exploratory Data Analysis

The EDA includes:
- Distribution plots for each feature
- Correlation heatmaps
- Boxplots and scatter plots
- Quality class distribution

📷 **Sample Screenshot from EDA:**
![image](https://github.com/user-attachments/assets/3d3dc784-924c-421a-b259-378a0d169f27)


---

## 🌐 Flask Web Application

The project includes a simple web interface built using Flask, where users can input wine parameters and receive a predicted quality rating based on a trained model.

### 🔧 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Flask
- HTML/CSS

---



## 🗂️ Project Structure

![image](https://github.com/user-attachments/assets/33a968bf-8c03-49b1-9394-1b79c44f0f7b)

![image](https://github.com/user-attachments/assets/3d54d966-1fd3-40cd-b531-eeaf43c1a15d)



wine_quality_project/
├── templates/
│ └── index.html # Frontend UI for prediction
├── app.py # Flask application
├── model.pkl # Trained ML model
├── winequality.csv # Dataset file
├── eda.ipynb # Jupyter Notebook for EDA
└── README.md # Project documentation


