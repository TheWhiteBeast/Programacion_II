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