# 🧬 Breast Cancer Detection: End-to-End MLOps Pipeline

## 📝 Project Overview
This repository contains a professional Machine Learning pipeline developed for the **Programming II Intermediate Challenge**. The project classifies breast cancer tumors as Benign or Malignant using the Breast Cancer Wisconsin dataset. 

Beyond model accuracy, the primary objective of this project is the implementation of **MLOps principles** using **MLflow** for experiment tracking, artifact management, and model versioning.

---

##  Project Development Checklist
This project was executed following a modular and scalable approach:

- [x] **Phase 1: Preprocessing & Data Engineering**: Data loading, cleaning (handling nulls/unnamed columns), and feature/target separation.
- [x] **Phase 2: Modeling & Evaluation**: Random Forest implementation with 5-fold Cross-Validation.
- [x] **Phase 3: MLOps Integration**: Implementation of MLflow tracking (localhost:5000), logging of hyperparameters, and metric tracking.
- [x] **Phase 4: Artifact Management**: Automated generation and logging of Confusion Matrices and ROC Curves.
- [x] **Phase 5: Model Registry**: Logging of the model with inferred signatures for deployment readiness.
- [x] **Phase 6: Clean Version Control**: Optimized repository using `.gitignore` to handle large files and environment clutter.

---

##  Project Structure
```text
Challenge Intermedio/
├── data/                                 # Raw dataset
│   └── breast-cancer-wisconsin.data.csv
├── src/                                  # Modular source code
│   ├── preprocessing.py                  # Data cleaning & engineering
│   ├── model_training.py                 # RF training & CV
│   ├── evaluation.py                     # Metrics & visualization logic
│   └── mlops_pipeline.py                 # Orchestrator with MLflow
├── metrics_plots/                        # Local copies of evaluation plots
├── .gitignore                            # Git rules for a clean repo
├── requirements.txt                      # Project dependencies
└── upload_results.py                     # Git automation script

## Technologies
Python 3.8+
MLflow: Experiment tracking and MLOps.
Scikit-Learn: Machine Learning algorithms and metrics.
Pandas & NumPy: Data manipulation.
Matplotlib: Statistical visualizations.

📊 Key Results
Model: Random Forest Classifier

Accuracy: ~96.4%

Precision: ~97.5%

MLOps Tracking: All runs are registered in the MLflow UI with full traceability.

💻 How to Use
1. Setup Environment
Install the required libraries:

Bash
pip install -r requirements.txt
2. Run MLflow UI
Start the tracking server in a separate terminal:

Bash
mlflow ui
3. Execute Pipeline
Run the main script to train, evaluate, and log the model:

Bash
python src/mlops_pipeline.py --data_path data/breast-cancer-wisconsin.data.csv
4. Explore
#Visit http://localhost:5000 to see the performance metrics, parameters, and saved plots (artifacts).

👤 Author
André Fajardo Master's in Data Science - Programming II (26A) | GitHub Profile