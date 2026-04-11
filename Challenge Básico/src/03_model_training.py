import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def train_classification_model():
    print("--- INICIANDO ENTRENAMIENTO DEL MODELO (MODEL TRAINING) ---")
    
    # 1. Configurar rutas
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)               
    data_dir = os.path.join(project_dir, 'data') 
    models_dir = os.path.join(project_dir, 'models') # Nueva carpeta para guardar modelos
    os.makedirs(models_dir, exist_ok=True)
    
    input_file = os.path.join(data_dir, 'glassdoor_reviews_Sentiment.csv')
    
    if not os.path.exists(input_file):
        print(f"Error: No se encontró el dataset en {input_file}")
        return
        
    # 2. Cargar datos
    df = pd.read_csv(input_file)
    
    # Eliminamos nulos en el texto lematizado y en la etiqueta
    df = df.dropna(subset=['3_Lemmatization', 'VADER_Sentiment'])
    
    X = df['3_Lemmatization'] # Usamos el texto limpio como variable independiente
    y = df['VADER_Sentiment'] # Usamos el sentimiento de VADER como variable objetivo
    
    # 3. Dividir en conjunto de entrenamiento y prueba (80% train, 20% test)
    print("Dividiendo datos en Train y Test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Calculation of grammatical probabilities (TF-IDF) & Extraction of main features
    print("Calculando probabilidades gramaticales (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 5. Construction model (Clasificación Naive Bayes)
    print("Entrenando modelo de Clasificación (Multinomial Naive Bayes)...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    # 6. Predicción rápida para validación
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n¡Modelo entrenado! Exactitud (Accuracy) inicial: {acc:.2f}")
    
    # 7. Guardar artefactos (Vectorizador y Modelo) para MLflow
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.pkl')
    model_path = os.path.join(models_dir, 'naive_bayes_model.pkl')
    
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)
    
    # Guardamos los datos de prueba para el script de evaluación
    test_data_path = os.path.join(data_dir, 'test_data.csv')
    test_df = pd.DataFrame({'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred})
    test_df.to_csv(test_data_path, index=False)
    
    print("\nArtefactos guardados exitosamente:")
    print(f" - Vectorizador: {vectorizer_path}")
    print(f" - Modelo: {model_path}")
    print(f" - Datos de prueba: {test_data_path}")

if __name__ == "__main__":
    train_classification_model()