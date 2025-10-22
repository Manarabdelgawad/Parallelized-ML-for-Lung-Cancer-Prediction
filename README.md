# 🏥 Lung Cancer Prediction - Parallelized ML Pipeline

A high-performance, parallelized machine learning pipeline for lung cancer prediction supporting CPU acceleration. Designed with a modular, production-ready architecture for scalability, maintainability, and performance.

📑 Table of Contents

Overview

Features

Project Structure

Installation

Configuration

Usage

Models

Performance

Dataset

Customization

Contributing

🔍 Overview

This project implements a scalable and parallelized machine learning pipeline for predicting lung cancer risk using demographic and symptom data. It provides CPU- parallel processing acceleration for optimal speed and flexibility.

Key Highlights

⚙️ Parallel Processing: Efficient multiprocessing using joblib and multiprocessing

🧩 Modular Architecture: Clean separation of components for easy extension

🏗️ Production Ready: Comprehensive logging, validation, and error handling

🔧 Flexible Configuration: Pydantic-based environment configuration

# Features

Automated Data Preprocessing: Encoding, imputation, normalization

Feature Selection: Chi-squared feature ranking

Multiple ML Models: SVM, Random Forest, Logistic Regression

Performance Monitoring: Execution time tracking for each pipeline stage

Extensible Design: Easily integrate new models or preprocessing steps

📂 Project Structure
Parallelized-ML/
├── config/
│   ├── __init__.py
│   └── settings.py          # Pydantic configuration with validation
├── data/
│   └── lung_cancer.csv      # Dataset
├── src/
│   ├── data/
│   │   ├── loader.py        # Data loading with CPU/GPU support
│   │   └── preprocessor.py  # Data cleaning and encoding
│   ├── features/
│   │   └── selector.py      # Feature selection algorithms
│   ├── models/
│   │   └── trainer.py       # Model training and evaluation
│   ├── utils/
│   │   ├── decorators.py    # Timing and performance decorators
│   │   └── logger.py        # Logging configuration
│   └── pipeline/
│       └── runner.py        # Main pipeline controller
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_pipeline.py
├── main.py                  # Application entry point
├── requirements.txt          # Dependencies
├── .gitignore
└── README.md

⚙️ Installation
Prerequisites

Python 3.8+

pip package manager

# Clone the repository
git clone https://github.com/<your-username>/Parallelized-ML.git
cd Parallelized-ML

# Create a virtual environment
python -m venv venv
source venv/bin/activate     # Linux/macOS
venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt

⚙️ Configuration

The project uses environment variables and Pydantic for configuration validation.

Example configuration snippet:

MODELS = [
    {"name": "SVC", "params": {"kernel": "rbf", "gamma": 0.5, "C": 1.0}},
    {"name": "RandomForest", "params": {"n_estimators": 15}},
    {"name": "LogisticRegression", "params": {"max_iter": 200}},
]

▶️ Usage

Run the complete pipeline:

python main.py

🤖 Models

Model	Key Parameters	Best For

Support Vector Classifier (SVC)	kernel=RBF, gamma=0.5, C=1.0	

Random Forest Classifier	n_estimators=15	

Logistic Regression	max_iter=200

⚡ Performance

Parallel Processing

Utilizes multiple CPU cores for:

Data preprocessing (column-wise parallelization)

Feature selection (feature-wise parallelization)

Model training (model-wise parallelization)

Performance Comparison

Execution Mode	Total Time	Speedup

Sequential	~11.79s	1x

Parallel (4 cores)	3.97s	3x

The parallel implementation provides nearly 3x performance improvement over sequential execution, reducing pipeline time from ~11.79 seconds to 3.97 seconds.

Resource Optimization

Automatic CPU core detection

Platform-aware parallelization (optimized for Windows/Linux)

Memory-efficient processing for large datasets

# Sample Output:

=== Pipeline Results ===
SVC: Accuracy=0.968, Precision=0.983, Recall=0.983, F1=0.983
RandomForest: Accuracy=0.968, Precision=0.983, Recall=0.983, F1=0.983
LogisticRegression: Accuracy=0.968, Precision=0.983, Recall=0.983, F1=0.983

📊 Dataset

Lung Cancer Prediction Dataset

Demographic Features

GENDER: Patient gender (M/F)

AGE: Patient age

Symptoms & Risk Factors

SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE,
FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING,
COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN

Target Variable

LUNG_CANCER: Cancer diagnosis (YES/NO)

🔧 Customization

Example: Add feature normalization to the preprocessing stage.

def normalize_features(self, X):
    """Feature normalization using StandardScaler."""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Check logs in pipeline.log for details.

Enable debug mode by setting LOG_LEVEL=DEBUG.

🤝 Contributing

welcome all contributions!
