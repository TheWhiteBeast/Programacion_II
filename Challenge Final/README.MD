# 🚍 Thesis Challenge: Geospatial ML for Public Transit Routing

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![GeoPandas](https://img.shields.io/badge/GeoPandas-Spatial%20Data-green.svg)](https://geopandas.org/)
[![MLflow](https://img.shields.io/badge/MLflow-MLOps-blueviolet.svg)](https://mlflow.org/)

## 📖 Project Overview
This project serves as a foundational Machine Learning pipeline for a master's thesis titled: *"Modelo para el trazado de rutas óptimas de transporte público mediante análisis geoespacial (GIS) y series de tiempo. Caso de estudio municipio de Guadalajara."*

The challenge demonstrates how to extract mathematical features from raw spatial data (Shapefiles) to train a classification model. The goal is to accurately classify a public transport route as either **"Transporte Masivo"** (Massive Transport) or **"Ruta Complementaria"** (Complementary Route) based purely on its geometric properties.

---

## 📂 1. Project Structure

~~~text
Challenge_Final/
├── data/
│   ├── raw/                      # Raw Shapefiles from IMEPLAN (.shp, .dbf, etc.)
│   └── processed/                # Cleaned and normalized datasets (CSV)
├── src/                          # Modular source code
│   ├── 01_preprocessing.py       # Spatial data extraction & engineering
│   ├── 02_model_training.py      # Random Forest training & class balancing
│   ├── 03_evaluation.py          # Metrics & visualization logic (ROC, Confusion Matrix)
│   └── 04_mlops_pipeline.py      # Orchestrator with MLflow tracking
├── metrics_plots/                # Local copies of evaluation plots
├── models/                       # Serialized models (.pkl)
├── .gitignore                    # Git rules for a clean repository
├── requirements.txt              # Project dependencies (geopandas, pyogrio, mlflow, etc.)
└── upload_results.py             # Git automation script
~~~

---

## 📚 2. Documentation

### 🗺️ Dataset Extraction
The raw data consists of official geographic Shapefiles provided by **IMEPLAN** (Instituto de Planeación y Gestión del Desarrollo del Área Metropolitana de Guadalajara).
Using `GeoPandas` and the `Pyogrio` engine, the spatial geometries (LineStrings) are loaded. Since standard Machine Learning models cannot process raw geometries, we apply **Feature Engineering** to extract numerical features:
* `ruta_longitud`: Total length of the route.
* `centroid_x` & `centroid_y`: Geographic center of the route.
* `rango_x` & `rango_y`: The spatial bounding box dimensions.

### 🧠 Model Construction
The extracted spatial dataset presents a strong **Class Imbalance** (e.g., 225 Complementary routes vs. 7 Massive routes). To solve the "Accuracy Paradox," we implemented a **Random Forest Classifier** with the parameter `class_weight='balanced'`. 
This architectural choice forces the algorithm to penalize errors on the minority class more heavily, achieving a significant Recall improvement for detecting "Massive" routes without relying on synthetic data generation (SMOTE).

### ⚙️ MLOps Integration
The entire lifecycle is tracked using **MLflow**. The pipeline logs:
* **Parameters:** Algorithm type, max depth.
* **Metrics:** Accuracy, Precision, Recall, F1-Score.
* **Artifacts:** Evaluation plots (Confusion Matrix, ROC Curve) and the serialized `.pkl` model with inferred signatures.
* **Tracking:** Localhost server (`http://localhost:5000`).

---

##  3. Execution Guide (Usage Instructions)

### Prerequisites
Clone the repository to your local machine:

~~~bash
git clone https://github.com/TheWhiteBeast/Programacion_II.git
cd "Challenge Final"
~~~

### Installation
Ensure you have an active virtual environment (e.g., Conda) and install the dependencies:

~~~bash
pip install -r requirements.txt
~~~

### Running the Pipeline
Execute the modules sequentially to reproduce the experiment:

~~~bash
python src/01_preprocessing.py
python src/02_model_training.py
python src/03_evaluation.py
python src/04_mlops_pipeline.py
~~~

### Visualizing MLOps Dashboard
To view the logged metrics and artifacts, start the MLflow UI:

~~~bash
mlflow ui
~~~

Open your browser and navigate to `http://localhost:5000`. Look for the **Thesis_Routing_Challenge** experiment.

---

## 🤖 4. GitHub Automation
Participants and collaborators can automatically upload their updated results to GitHub using the provided automation script. This script stages all allowed changes, prompts for a custom commit message, and pushes to the `main` branch.

Run the following command:

~~~bash
python upload_results.py
~~~

