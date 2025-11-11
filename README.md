## Lung Cancer Prediction — Parallelized ML Pipeline

A high-performance, parallelized machine-learning pipeline for lung cancer prediction using demographic and symptom data, built for scalability, maintainability, and speed.

# Table of Contents

Overview

Key Features

Project Structure

Installation

Configuration

Usage

Models & Parameters

Performance

Dataset

Contributing

License

# Overview

This project offers a modular and parallelized machine learning pipeline aimed at predicting lung cancer risk from patient demographic and symptom data. It supports CPU-based parallelism to speed up computation, while preserving clean separation of modules (data loading, preprocessing, model training, evaluation).

The goal is to provide a production-ready baseline that’s easy to extend, monitor, debug, or integrate into larger systems.

# Key Features

Parallel Processing using joblib and Python’s multiprocessing

Modular Design: each pipeline stage is separated (loader, preprocessor, selector, trainer)

Configuration via Pydantic for type-safe, validated environment settings

Automated Data Preprocessing: missing value imputation, encoding, normalization

Feature Selection: e.g. chi-squared ranking

Multiple ML Algorithms: Support Vector Machine, Random Forest, Logistic Regression

Performance Logging: records execution time per stage

Easy Extensibility: plug in new preprocessing steps, feature selectors, or models

# Project Structure

```bashParallelized-ML-for-Lung-Cancer-Prediction/

├── config/
│   ├── __init__.py
│   └── settings.py  
├── data/
│   └── lung_cancer.csv      # Dataset
├── src/
│   ├── data/
│   │   ├── loader.py        # Data loading 
│   │   └── preprocessor.py  # Imputation, encoding, normalization
│   ├── features/
│   │   └── selector.py      # Feature selection 
│   ├── models/
│   │   └── trainer.py       # Model training & evaluation
│   ├── utils/
│   │   ├── decorators.py    # timing, caching
│   │   └── logger.py         # logging setup
│   └── pipeline/
│       └── runner.py        # Orchestrates full pipeline
├── main.py                   # Entry point script
├── requirements.txt
├── pipeline.log              # Logs created by pipeline runs
├── .gitignore
└── README.md
```
# Installation

Prerequisites

Python 3.8+

pip package manager

Steps
```bash
git clone https://github.com/Manarabdelgawad/Parallelized-ML-for-Lung-Cancer-Prediction.git  
cd Parallelized-ML-for-Lung-Cancer-Prediction  
```

```bash
python -m venv venv 
```


 On Linux/macOS:
 ```bash
source venv/bin/activate
```
 On Windows:
```bash
venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```


# Configuration

This project uses Pydantic models and environment variables for configuration validation and flexibility.

Example CONFIG snippet (in code or via .env):

MODELS = [
    {
        "name": "SVC",
        "params": {
            "kernel": "rbf",
            "gamma": 0.5,
            "C": 1.0
        }
    },
    {
        "name": "RandomForest",
        "params": {
            "n_estimators": 15
        }
    },
    {
        "name": "LogisticRegression",
        "params": {
            "max_iter": 200
        }
    },
]


You may also configure logging levels, parallelism settings, or add new pipelines in the configuration.

# Usage

To run the full pipeline:

python main.py


This launches data loading → preprocessing → feature selection → model training → evaluation, logging execution times and metrics.

Models & Parameters
Model	Default Parameters
Support Vector Classifier (SVC)	kernel="rbf", gamma=0.5, C=1.0
Random Forest	n_estimators=15
Logistic Regression	max_iter=200

You may add or override models in the configuration file or via environment settings.

# Performance

Parallel vs Sequential comparison (4-core example):

Sequential mode: ~11.79 sec

Parallel mode: ~3.97 sec

This yields a ~3× speedup thanks to parallelizing across data preprocessing, feature selection, and model training steps.

Parallelism is automatically adapted to the CPU environment via core detection.

# Dataset
Lung Cancer Prediction Dataset

Features include:

Demographics: GENDER, AGE

Symptoms / Risk Factors: SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE,
FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, 
SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN

Target: LUNG_CANCER (YES / NO)

You can replace or augment this dataset, provided your new data fits into the loader + preprocessing pipeline (or you extend those components).

# Contributing

I welcome contributions! Please follow these steps:

Fork the repository

Create a feature branch: git checkout -b feature/my-feature

Make your changes (with tests)

Submit a Pull Request and describe your improvements

# License

This project is licensed under the Apache License 2.0 — see the LICENSE file for details.
