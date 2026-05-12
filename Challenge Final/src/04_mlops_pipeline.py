import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def run_mlops():
    print("--- FASE 4: MLOps CON MLFLOW ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    
    test_path = os.path.join(project_dir, 'data', 'processed', 'test_results.csv')
    model_path = os.path.join(project_dir, 'models', 'rf_rutas_model.pkl')
    plots_dir = os.path.join(project_dir, 'metrics_plots')

    df = pd.read_csv(test_path)
    y_test = df['y_test']
    y_pred = df['y_pred']
    X_test = df.drop(columns=['y_test', 'y_pred', 'y_prob'])
    
    model = joblib.load(model_path)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    # Configurar MLflow
    mlruns_dir = os.path.join(project_dir, 'mlruns')
    mlflow.set_tracking_uri(f"file:///{mlruns_dir.replace(chr(92), '/')}")
    mlflow.set_experiment("Thesis_Routing_Challenge")

    with mlflow.start_run(run_name="RF_GIS_Features"):
        # Log parámetros
        mlflow.log_param("model", "Random Forest")
        mlflow.log_param("max_depth", 5)
        
        # Log métricas
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log artefactos
        mlflow.log_artifacts(plots_dir, artifact_path="plots")

        # Signature y Modelo
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=signature)

        print(f"✅ Pipeline MLOps Finalizado con Éxito. Run ID: {mlflow.active_run().info.run_id}")
        print("Ejecuta 'mlflow ui' para ver los resultados.")

if __name__ == "__main__":
    run_mlops()