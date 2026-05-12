import os
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler

def preprocess_gis_data():
    print("--- FASE 1: PREPROCESAMIENTO GEOESPACIAL ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    raw_dir = os.path.join(project_dir, 'data', 'raw')
    processed_dir = os.path.join(project_dir, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # Rutas a tus archivos SHP
    file_comp = os.path.join(raw_dir, 'amg_rutas_complementarias_otros_servicios_mapa_baseLine.shp')
    file_masivo = os.path.join(raw_dir, 'amg_transporte_masivo_pimusLine.shp')

    try:
        # Cargar mapas
        print("Cargando Shapefiles de IMEPLAN...")
        gdf_comp = gpd.read_file(file_comp, engine="pyogrio")
        gdf_masivo = gpd.read_file(file_masivo, engine="pyogrio")

        # Crear la variable objetivo (Target)
        gdf_comp['is_masivo'] = 0
        gdf_masivo['is_masivo'] = 1

        # Unir ambos datasets
        gdf = pd.concat([gdf_comp, gdf_masivo], ignore_index=True)

        # Feature Engineering: Extraer datos matemáticos de la geometría
        print("Extrayendo características geométricas...")
        gdf['ruta_longitud'] = gdf.geometry.length
        gdf['centroid_x'] = gdf.geometry.centroid.x
        gdf['centroid_y'] = gdf.geometry.centroid.y
        
        # Calcular los límites de la ruta (Bounding Box)
        bounds = gdf.geometry.bounds
        gdf['rango_x'] = bounds['maxx'] - bounds['minx']
        gdf['rango_y'] = bounds['maxy'] - bounds['miny']

        # Descartar la geometría pura y variables de texto complejas para el ML básico
        df = pd.DataFrame(gdf.drop(columns=['geometry']))
        
        # Seleccionar solo columnas numéricas para el modelo
        numeric_cols = ['ruta_longitud', 'centroid_x', 'centroid_y', 'rango_x', 'rango_y', 'is_masivo']
        df_ml = df[numeric_cols].copy()

        # Limpiar Nulos
        print("Limpiando nulos e inconsistencias...")
        df_ml = df_ml.fillna(df_ml.mean())

        # Normalización
        scaler = StandardScaler()
        features = ['ruta_longitud', 'centroid_x', 'centroid_y', 'rango_x', 'rango_y']
        df_ml[features] = scaler.fit_transform(df_ml[features])

        # Guardar CSV procesado
        output_path = os.path.join(processed_dir, 'rutas_procesadas_ml.csv')
        df_ml.to_csv(output_path, index=False)
        print(f"¡Preprocesamiento exitoso! Datos listos para ML en: {output_path}")

    except Exception as e:
        print(f"Error al procesar los mapas: {e}")

if __name__ == "__main__":
    preprocess_gis_data()