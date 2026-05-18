# 🚀 ML Algorithm Comparator Pro

![Streamlit](https://img.shields.io/badge/Streamlit-1.45.1-FF4B4B.svg?style=flat&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg?style=flat&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**ML Algorithm Comparator Pro** is a robust, modular Streamlit application designed for rapid machine learning experimentation. It provides a comprehensive workspace to upload datasets, perform automated preprocessing, train and tune a wide variety of models, and visualize performance through interactive charts and professional reports.

## ✨ Key Features

- **🏗️ Modular Design**: Clean separation of concerns with dedicated modules for `preprocessing`, `models`, `visualizations`, and `utils`.
- **🤖 AutoML Capabilities**: Automatic problem detection (Classification vs. Regression) and hyperparameter tuning using GridSearchCV/RandomizedSearchCV.
- **🔀 Ensemble Learning**: Easily create Voting and Stacking models to boost performance.
- **📊 Advanced Analytics**: ROC/AUC curves, Learning Curves, Feature Importance, and Silhouette Analysis for clustering.
- **🧩 Unsupervised Learning**: Support for both numeric (K-Means) and categorical (K-Modes) clustering.
- **💾 Model Management**: Save trained models with their scalers and metadata; reload them for future predictions.
- **📄 Professional Reporting**: Export comprehensive HTML reports summarizing model performance and best results.

---

## 📂 Project Structure

```text
ML Project/
├── app.py                 # Primary Streamlit web application
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
├── datasets/              # Sample datasets for testing
│   ├── iris.csv           # Classification sample
│   ├── housing.csv        # Regression sample
│   └── mall_customers.csv # Clustering sample
├── saved_models/          # Directory for persisted .pkl models
└── src/                   # Core modular package
    ├── __init__.py        # Package interface
    ├── preprocessing.py   # Data cleaning, encoding & scaling
    ├── models.py          # ML algorithms & training logic
    ├── visualizations.py  # Advanced plotting & metrics
    └── utils.py           # File I/O, caching & reporting
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8 or higher installed.

### 2. Installation
Clone this repository and install the required packages:

```bash
# Navigate to the project folder
cd "F:\ML Project"

# Install dependencies
pip install -r requirements.txt
```

### 3. Running the App
Launch the Streamlit interface:

```bash
streamlit run app.py
```
The application will automatically open in your default browser at `http://localhost:8501`.

---

## 💻 Usage Guide

### 1. Data Ingestion
- **Upload**: Drag and drop your CSV or Excel files.
- **Built-in**: Quickly test features using the provided sample datasets.

### 2. Target Selection & Detection
- Select your target column. The app automatically detects if the task is **Regression** or **Classification**.
- View detailed dataset statistics and missing value reports.

### 3. Model Training
- Choose from a wide range of algorithms (Linear Models, Trees, Ensembles, SVMs, MLPs).
- Enable **Hyperparameter Tuning** for automatic optimization.
- Adjust test-split ratios and cross-validation folds.

### 4. Analysis & Prediction
- **Visualizations**: Compare models using multi-metric bar charts, confusion matrices, and ROC curves.
- **Prediction**: Use your best-performing model to predict on the full dataset or input custom values manually.
- **Export**: Download your predictions or a full HTML performance report.

---

## 🌐 Deployment (Streamlit Cloud)

This app is ready for one-click deployment on [Streamlit Cloud](https://streamlit.io/cloud):

1. **Push** this code to a GitHub repository.
2. **Login** to Streamlit Cloud and click "New app".
3. **Select** your repository and the `app.py` file.
4. **Deploy!** The platform will automatically install dependencies from `requirements.txt`.

---

## 🧪 Modular API Documentation

### Preprocessing (`src.preprocessing`)
```python
from src.preprocessing import preprocess_data

# Automated preprocessing: imputation, encoding, and target detection
X, y, features, encoders, p_type, df_encoded = preprocess_data(df, target_col="target")
```

### Models (`src.models`)
```python
from src.models import train_multiple_models, REGRESSION_MODELS

# Batch train multiple regression models
results = train_multiple_models(REGRESSION_MODELS, X_train, y_train, X_test, y_test)
```

### Visualizations (`src.visualizations`)
```python
from src.visualizations import plot_confusion_matrices

# Generate interactive confusion matrices for all trained models
fig = plot_confusion_matrices(trained_models, X_test, y_test)
```

---

## 💡 Troubleshooting

- **Large Datasets**: If the app slows down, use the "Max rows" slider in the sidebar to sample your data.
- **Module Not Found**: Ensure you are running the app from the root directory (`F:\ML Project`) so the `src` package is discoverable.
- **K-Modes Error**: Ensure `kmodes` is installed via requirements.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

