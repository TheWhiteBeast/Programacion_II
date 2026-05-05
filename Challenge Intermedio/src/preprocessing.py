import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse
import sys
import logging

# Configuración de logs
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_data(file_path):
    try:
        logging.info(f"Cargando datos desde: {file_path}")
        df = pd.read_csv(file_path)
        
        if df.empty:
            raise ValueError("El archivo CSV está vacío.")
        return df
        
    except FileNotFoundError:
        logging.error(f"El archivo {file_path} no fue encontrado.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error inesperado al cargar los datos: {e}")
        sys.exit(1)

def explore_data(df):
    try:
        print("\n" + "-"*40)
        print("--- DATA INFO ---")
        df.info()
        
        print("\n--- DATA DESCRIBE ---")
        print(df.describe())
        
        # Ignoramos la columna vacía Unnamed: 32 para contar nulos reales
        df_temp = df.drop(columns=['Unnamed: 32'], errors='ignore')
        print("\n--- NULLS AND OTHER CHARACTERS ---")
        print("Valores nulos reales (sin contar Unnamed: 32):\n", df_temp.isnull().sum().sum())
        
        if (df_temp == '?').sum().sum() > 0:
            print("Se encontraron caracteres '?' en el dataset.")
        
        # El target en este dataset es 'diagnosis'
        target_col = 'diagnosis' if 'diagnosis' in df.columns else df.columns[-1]
        print(f"\n--- DATA VALUES AND COUNTS (Target: {target_col}) ---")
        print(df[target_col].value_counts())
        print("="*40 + "\n")
        
    except Exception as e:
        logging.error(f"Error durante la exploración de datos: {e}")
        sys.exit(1)

def preprocess_data(df):
    try:
        logging.info("Iniciando preprocesamiento y normalización...")
        
        # 1. Limpieza de columnas basura e IDs
        if 'Unnamed: 32' in df.columns:
            df = df.drop(columns=['Unnamed: 32'])
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
            
        # 2. Reemplazar '?' por nulos de numpy y eliminar filas con nulos reales
        df = df.replace('?', np.nan)
        df = df.dropna()
        
        # 3. Separar características (X) y objetivo (y)
        target_col = 'diagnosis'
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # 4. Transformar Target 'M' (Maligno) y 'B' (Benigno) a 1 y 0
        if y.dtype == 'object':
            y = y.map({'M': 1, 'B': 0})
            
        # Asegurar que X sea numérico
        X = X.apply(pd.to_numeric)
        
        # 5. Normalización usando StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        logging.info("Preprocesamiento y normalización completados exitosamente.")
        return X_scaled, y.values
        
    except Exception as e:
        logging.error(f"Error durante el preprocesamiento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 y 2: Preprocesamiento de Cancer Detection")
    parser.add_argument('--data_path', type=str, required=True, help="Ruta relativa o absoluta al archivo CSV")
    args = parser.parse_args()
    
    df = load_data(args.data_path)
    explore_data(df)
    X, y = preprocess_data(df)
    
    print(f"Dimensiones finales -> X (features): {X.shape}, y (target): {y.shape}")