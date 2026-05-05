import argparse
import logging
import sys
from datetime import datetime
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Importar nuestros módulos locales
from preprocessing import load_data, explore_data, preprocess_data
from model_training import train_model
from evaluation import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main(data_path):
    try:
        logging.info("--- Iniciando MLOps Pipeline ---")
        
        # 1. Configurar MLflow (SOW: Tracking url localhost)
        # Esto indica que MLflow guardará los registros localmente en ./mlruns y estará disponible en el puerto 5000
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("Cancer_Detection_Challenge")

        # 2. Preprocesamiento de Datos (Stage 1 & 2)
        df = load_data(data_path)
        # explore_data(df) # Puedes descomentar esto si quieres ver la info del dataset cada vez que corres el pipeline
        X, y = preprocess_data(df)

        # 3. Iniciar ejecución de MLflow (Stage 3)
        run_name = f"Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name) as run:
            
            # Registrar fecha de fin aproximada (SOW: End Date datetime)
            end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            mlflow.set_tag("End Date", end_date)

            # Registrar parámetros
            mlflow.log_param("data_path", data_path)
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("cv_folds", 5)

            # 4. Entrenamiento del modelo
            model, X_test, y_test = train_model(X, y)

            # 5. Evaluación (Métricas y Gráficas)
            metrics = evaluate_model(model, X_test, y_test)

            # 6. MLOps: Registrar métricas (SOW: Log metrics)
            mlflow.log_metrics(metrics)

            # 7. MLOps: Registrar gráficas como artefactos (SOW: Save the plot and log it as an artifact)
            mlflow.log_artifact("metrics_plots/confusion_matrix.png", "plots")
            mlflow.log_artifact("metrics_plots/roc_curve.png", "plots")

            # 8. MLOps: Firma y guardado del modelo (SOW: Model signatures)
            predictions = model.predict(X_test)
            signature = infer_signature(X_test, predictions)
            mlflow.sklearn.log_model(model, "random_forest_model", signature=signature)

            logging.info(f" Pipeline ejecutado exitosamente.")
            logging.info(f"ID del Run de MLflow: {run.info.run_id}")
            print("\n" + "*"*50)
            print("Para ver tus resultados en MLflow, abre una NUEVA terminal y ejecuta:")
            print("mlflow ui")
            print("Luego abre tu navegador en: http://localhost:5000")
            print("*"*50 + "\n")

    except Exception as e:
        logging.error(f"Error crítico en el pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps Pipeline para Cancer Detection")
    parser.add_argument('--data_path', type=str, required=True, help="Ruta al archivo CSV de datos")
    args = parser.parse_args()
    
    main(args.data_path)