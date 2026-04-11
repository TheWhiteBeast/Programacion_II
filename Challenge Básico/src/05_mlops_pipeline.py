import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def run_mlops_pipeline():
    print("--- INICIANDO PIPELINE DE MLOPS ---")
    
    # 1. Configurar rutas de directorios
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)               
    data_dir = os.path.join(project_dir, 'data') 
    models_dir = os.path.join(project_dir, 'models')
    
    # 2. Rutas de los archivos requeridos
    test_file = os.path.join(data_dir, 'test_data.csv')
    model_file = os.path.join(models_dir, 'naive_bayes_model.pkl')
    vectorizer_file = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
    plot_ngrams = os.path.join(data_dir, 'ngrams_distribution.png')
    plot_confusion = os.path.join(data_dir, 'confusion_matrix.png')
    
    required_files = [test_file, model_file, vectorizer_file, plot_ngrams, plot_confusion]
    for filepath in required_files:
        if not os.path.exists(filepath):
            print(f"Error critico: No se encontro el archivo {filepath}")
            return

    # 3. Cargar datos y modelos
    df_test = pd.read_csv(test_file)
    X_test = df_test['X_test'].fillna("")
    y_test = df_test['y_test']
    y_pred = df_test['y_pred']
    
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)

    # 4. Calcular metricas
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    
    # 5. Configuracion de MLflow
    mlruns_dir = os.path.join(project_dir, 'mlruns')
    tracking_uri = f"file:///{mlruns_dir.replace(chr(92), '/')}"
    mlflow.set_tracking_uri(tracking_uri)
    
    experiment_name = "Glassdoor_NLP_Classification"
    mlflow.set_experiment(experiment_name)
    
    print("Iniciando registro en MLflow...")
    
    with mlflow.start_run(run_name="Naive_Bayes_Run"):
        # Parametros
        mlflow.log_param("model_type", "Multinomial Naive Bayes")
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("max_features", 1000)
        
        # Metricas
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Artefactos
        mlflow.log_artifact(plot_ngrams, artifact_path="plots")
        mlflow.log_artifact(plot_confusion, artifact_path="plots")
        mlflow.log_artifact(vectorizer_file, artifact_path="transformers")
        
        # Firma y Modelo
        X_test_vec = vectorizer.transform(X_test)
        signature = infer_signature(X_test_vec, y_pred)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="naive_bayes_model",
            signature=signature
        )
        
        run_info = mlflow.active_run().info
        print("Registro completado exitosamente.")
        print(f"Run ID: {run_info.run_id}")
        print(f"Tracking URI: {tracking_uri}")
        print("Para visualizar la interfaz, ejecuta 'mlflow ui' en la terminal.")

if __name__ == "__main__":
    run_mlops_pipeline()