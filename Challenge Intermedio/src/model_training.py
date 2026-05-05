from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def train_model(X, y):
    """
    Entrena el modelo usando RandomForest y realiza Cross Validation.
    Retorna el modelo entrenado y los datos de prueba.
    """
    try:
        logging.info("Iniciando separación de datos (Train/Test)...")
        # Separamos 80% para entrenar y 20% para probar
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logging.info("Construyendo modelo (Random Forest)...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        logging.info("Realizando Cross Validation (5 folds)...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        logging.info(f"Cross Validation Scores: {cv_scores}")
        logging.info(f"Cross Validation Mean Accuracy: {cv_scores.mean():.4f}")
        
        logging.info("Entrenando modelo final...")
        model.fit(X_train, y_train)
        
        logging.info("Modelo entrenado exitosamente.")
        return model, X_test, y_test
        
    except Exception as e:
        logging.error(f"Error durante el entrenamiento del modelo: {e}")
        raise