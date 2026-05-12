import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model():
    print("--- FASE 2: ENTRENAMIENTO DEL MODELO ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    data_path = os.path.join(project_dir, 'data', 'processed', 'rutas_procesadas_ml.csv')
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Cargar datos
    df = pd.read_csv(data_path)
    X = df.drop(columns=['is_masivo'])
    y = df['is_masivo']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Entrenar Random Forest
    print("Entrenando Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"Accuracy inicial: {accuracy_score(y_test, y_pred):.2f}")

    # Guardar Artefactos
    joblib.dump(model, os.path.join(models_dir, 'rf_rutas_model.pkl'))
    
    # Guardar datos de prueba para evaluación
    test_data = X_test.copy()
    test_data['y_test'] = y_test
    test_data['y_pred'] = y_pred
    test_data['y_prob'] = y_prob
    test_data.to_csv(os.path.join(project_dir, 'data', 'processed', 'test_results.csv'), index=False)
    print("Modelo y datos de prueba guardados.")

if __name__ == "__main__":
    train_model()