# Ecommerce Customer Analysis

![Python](https://img.shields.io/badge/Python-3.7%2B-blue?style=flat-square&logo=python)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellowgreen?style=flat-square&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-Academic-lightgrey?style=flat-square)

A comprehensive machine learning and data analysis project focused on ecommerce customer behavior modeling, predictive analytics, and business intelligence extraction.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Author](#author)
- [License](#license)

---

## Overview

This project delivers an end-to-end data science pipeline for analyzing ecommerce customer data. It encompasses exploratory data analysis (EDA), feature engineering, and the development of machine learning models designed to uncover purchasing patterns and support data-driven business decisions.

The analysis aims to answer key business questions such as:
- What behavioral patterns distinguish high-value customers?
- Which features are the strongest predictors of customer spending?
- How can customer segmentation inform targeted marketing strategies?

---

## Project Structure

```
ecommerce-customer-analysis/
│
├── Ecommerce project.ipynb       # Main notebook: EDA, visualizations, and model training
├── ecproject.py                  # Core module for data loading and preprocessing
├── project1ml.py                 # Machine learning model definitions and evaluation utilities
├── Ecommerce Customers/          # Raw customer dataset
├── requirements.txt              # Python dependency manifest
└── README.md                     # Project documentation
```

---

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Git

### Steps

**1. Clone the repository**

```bash
git clone https://github.com/your-username/ecommerce-customer-analysis.git
cd ecommerce-customer-analysis
```

**2. (Recommended) Create and activate a virtual environment**

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Usage

Launch the main Jupyter Notebook to run the full analysis pipeline:

```bash
jupyter notebook "Ecommerce project.ipynb"
```

To use the processing and modeling modules independently:

```python
# Data preprocessing
from ecproject import load_data, preprocess

df = load_data("Ecommerce Customers/")
df_clean = preprocess(df)

# Machine learning models
from project1ml import train_model, evaluate_model

model = train_model(df_clean)
evaluate_model(model, df_clean)
```

---

## Features

| Feature | Description |
|---|---|
| Exploratory Data Analysis | Statistical summaries, correlation analysis, and distribution plots |
| Data Preprocessing | Missing value handling, encoding, normalization, and feature engineering |
| Predictive Modeling | Regression and/or classification models for customer spend prediction |
| Model Evaluation | Performance metrics including RMSE, R², accuracy, and confusion matrices |
| Visualizations | Heatmaps, pair plots, residual plots, and feature importance charts |

---

## Dataset

The dataset is located in the `Ecommerce Customers/` directory and contains anonymized records of customer interactions with an ecommerce platform.

**Key attributes include:**

- Customer demographics (e.g., location, membership duration)
- Session behavior (e.g., time on app, time on website)
- Purchase history and annual spending figures

> **Note:** Ensure the dataset files are present in the `Ecommerce Customers/` directory before running the notebook.

---

## Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.7+ | Core programming language |
| Jupyter Notebook | Latest | Interactive development environment |
| pandas | Latest | Data manipulation and analysis |
| NumPy | Latest | Numerical operations |
| scikit-learn | Latest | Machine learning algorithms |
| Matplotlib | Latest | Base visualizations |
| seaborn | Latest | Statistical visualizations |

---

## Author

**Atia Sanjida**

---

## License

This project is developed for academic and educational purposes. All rights reserved © 2026 Atia Sanjida.

---
