# 🏥 Lung Cancer Prediction - Parallelized ML Pipeline

A high-performance, parallelized machine learning pipeline for lung cancer prediction that supports both CPU and GPU acceleration with modular, production-ready code architecture.

# Table of Contents

Overview

Features

Project Structure

Installation

Usage

Configuration

Models

Performance

Testing

Contributing

# Overview

This project implements a scalable machine learning pipeline for predicting lung cancer risk based on patient demographic and symptom data. The system is designed with parallel processing capabilities and supports both CPU and GPU execution for optimal performance.

Key Highlights
Parallel Processing: Utilizes joblib and multiprocessing for efficient computation

GPU Acceleration: Optional CUDA support via RAPIDS AI libraries

Modular Architecture: Clean separation of concerns with configurable components

Production Ready: Comprehensive logging, error handling, and testing

Flexible Configuration: Environment-based settings with validation

Features
Data Preprocessing: Automated encoding, imputation, and normalization

Feature Selection: Chi-squared based feature importance ranking

Multiple ML Models: Support Vector Machines, Random Forest, Logistic Regression

Cross-Platform: Works on Windows, Linux, and macOS

Performance Monitoring: Execution time tracking for all pipeline stages

Comprehensive Testing: Unit tests for all components

Extensible Design: Easy to add new models and preprocessing techniques

Project Structure
text
Parallelized-ML/

├── config/                 # Configuration management

│   ├── __init__.py

│   └── settings.py        # Pydantic settings with validation

├── data/                  # Dataset directory

│   └── lung_cancer.csv

├── src/                   # Source code

│   ├── data/             # Data handling modules

│   │   ├── loader.py     # Data loading with GPU support

│   │   └── preprocessor.py # Data cleaning and encoding

│   ├── features/         # Feature engineering


│   │   └── selector.py   # Feature selection algorithms

│   ├── models/           # ML models

│   │   └── trainer.py    # Model training and evaluation

│   ├── utils/            # Utilities

│   │   ├── decorators.py # Timing decorators

│   │   └── logger.py     # Logging configuration

│   └── pipeline/         # Pipeline orchestration

│       └── runner.py     # Main pipeline controller
├── tests/                # Test suite

│   ├── test_data.py

│   ├── test_models.py

│   └── test_pipeline.py

├── main.py              # Application entry point

├── requirements.txt     # Python dependencies

 ___ .gitignore
 
└── README.md           # This file

🚀 Installation

Prerequisites
Python 3.8+

pip package manager

Step-by-Step Setup
Clone the repository

bash
git clone <repository-url>
cd Parallelized-ML
Create virtual environment

bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
Install dependencies

bash
pip install -r requirements.txt
Optional GPU Support
For GPU acceleration (NVIDIA GPUs only):

bash
# Install CUDA-enabled versions (Linux only)
pip install cuml-cu11 cudf-cu11 cupy-cuda11x

Configuration
The project uses environment variables and Pydantic for configuration management:

Key Configuration Options


python
MODELS = [
    {
        "name": "SVC",
        "params": {"kernel": "rbf", "gamma": 0.5, "C": 1.0}
    },
    {
        "name": "RandomForest", 
        "params": {"n_estimators": 15}
    },
    {
        "name": "LogisticRegression",
        "params": {"max_iter": 200}
    }
]
Usage
Basic Execution
bash
# Run the complete pipeline
python main.py
# Models
The pipeline includes three machine learning models:

1. Support Vector Classifier (SVC)
Kernel: Radial Basis Function (RBF)

Parameters: gamma=0.5, C=1.0

Best for: Complex non-linear decision boundaries

2. Random Forest Classifier
Ensemble method: 15 decision trees

Best for: Robust performance with feature importance

3. Logistic Regression
Regularization: L2 by default

Max iterations: 200

Best for: Interpretable linear relationships

⚡ Performance
Parallel Processing
The pipeline leverages multiple CPU cores for:

Data preprocessing (column-wise parallelization)

Feature selection (feature-wise parallelization)

Model training (model-wise parallelization)

GPU Acceleration
When configured for GPU:

Data loading with cuDF (10x faster than pandas)

Model training with cuML (5-50x speedup)

Matrix operations with CuPy

Performance Comparison
Operation	CPU (4 cores)	GPU (NVIDIA)
Data Loading	~100ms	~10ms
Preprocessing	~500ms	~50ms
Feature Selection	~200ms	~30ms
Model Training	~2s	~0.5s

# Output:

=== Pipeline Results ===
SVC: Accuracy=0.968, Precision=0.983, Recall=0.983, F1=0.983
RandomForest: Accuracy=0.968, Precision=0.983, Recall=0.983, F1=0.983
LogisticRegression: Accuracy=0.968, Precision=0.983, Recall=0.983, F1=0.983

🧪 Testing
Run the test suite to verify installation:

bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_data.py -v
python -m pytest tests/test_models.py -v
python -m pytest tests/test_pipeline.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
Test Coverage
✅ Data loading and validation

✅ Preprocessing pipelines

✅ Feature selection algorithms

✅ Model training and evaluation

✅ End-to-end pipeline execution

📊 Dataset
Lung Cancer Dataset
The project uses a lung cancer prediction dataset with the following features:

Demographic Features:

GENDER: Patient gender (M/F)

AGE: Patient age

Symptoms and Risk Factors:

SMOKING: Smoking history

YELLOW_FINGERS: Yellow fingers indicator

ANXIETY: Anxiety levels

PEER_PRESSURE: Peer pressure influence

CHRONIC_DISEASE: Chronic disease presence

FATIGUE: Fatigue levels

ALLERGY: Allergy history

WHEEZING: Wheezing symptoms

ALCOHOL_CONSUMING: Alcohol consumption

COUGHING: Coughing frequency

SHORTNESS_OF_BREATH: Breathing difficulties

SWALLOWING_DIFFICULTY: Swallowing problems

CHEST_PAIN: Chest pain presence

Target Variable:

LUNG_CANCER: Cancer diagnosis (YES/NO)

🔧 Customization

python
def normalize_features(self, X):
    """Add feature normalization"""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(X)
# 🐛 Troubleshooting
Common Issues
Import Errors

bash
# Ensure virtual environment is activated
venv\Scripts\activate
# Reinstall requirements
pip install -r requirements.txt
GPU Libraries Not Available

Project automatically falls back to CPU

Check CUDA installation for GPU support

Memory Issues

Reduce N_JOBS in configuration

Use smaller dataset subsets for testing

Logs and Debugging
Check pipeline.log for detailed execution logs

Enable debug mode: LOG_LEVEL=DEBUG in configuration

Use test_run.py to verify component functionality

# 🤝 Contributing
We welcome contributions! Please see our contributing guidelines:

Fork the repository

Create a feature branch: git checkout -b feature/amazing-feature

Commit changes: git commit -m 'Add amazing feature'

Push to branch: git push origin feature/amazing-feature

Open a Pull Request
