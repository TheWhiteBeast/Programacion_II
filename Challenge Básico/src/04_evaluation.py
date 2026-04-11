import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

def evaluate_model():
    print("--- INICIANDO EVALUACION DEL MODELO ---")
    
    # 1. Configurar rutas
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)               
    data_dir = os.path.join(project_dir, 'data') 
    
    test_file = os.path.join(data_dir, 'test_data.csv')
    
    if not os.path.exists(test_file):
        print(f"Error: No se encontraron los datos de prueba en {test_file}")
        return
        
    # 2. Cargar datos de prueba
    df_test = pd.read_csv(test_file)
    y_test = df_test['y_test']
    y_pred = df_test['y_pred']
    
    # 3. Reporte de Clasificacion
    print("\n" + "="*50)
    print("REPORTE DE CLASIFICACION (Metricas)")
    print("="*50)
    report = classification_report(y_test, y_pred)
    print(report)
    
    # 4. Generar y guardar la Matriz de Confusion
    # Aseguramos el orden logico de las etiquetas
    labels = ['Positive', 'Neutral', 'Negative']
    
    # Filtramos las etiquetas que realmente existen en el set de prueba
    present_labels = [label for label in labels if label in y_test.unique() or label in y_pred.unique()]
    
    cm = confusion_matrix(y_test, y_pred, labels=present_labels)
    
    # Creacion de paleta de colores unica y profesional (Arena -> Terracota -> Cian Oscuro)
    colors = ["#F4F1DE", "#E07A5F", "#2A9D8F", "#264653"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_terracota_cyan", colors)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=custom_cmap, 
                xticklabels=present_labels,
                yticklabels=present_labels)
    plt.title('Matriz de Confusion - Naive Bayes')
    plt.xlabel('Prediccion del Modelo')
    plt.ylabel('Valor Real (VADER)')
    plt.tight_layout()
    
    cm_path = os.path.join(data_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    
    print("\n" + "="*50)
    print(f"Matriz de confusion guardada con exito en:")
    print(f"   {cm_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    evaluate_model()