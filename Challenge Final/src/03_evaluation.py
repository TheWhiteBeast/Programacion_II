import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def evaluate_model():
    print("--- FASE 3: EVALUACIÓN Y MÉTRICAS ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    test_path = os.path.join(project_dir, 'data', 'processed', 'test_results.csv')
    plots_dir = os.path.join(project_dir, 'metrics_plots')
    os.makedirs(plots_dir, exist_ok=True)

    df = pd.read_csv(test_path)
    y_test = df['y_test']
    y_pred = df['y_pred']
    y_prob = df['y_prob']

    print(classification_report(y_test, y_pred, target_names=['Complementaria', 'Masivo']))

    # 1. Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Comp', 'Masivo'], yticklabels=['Comp', 'Masivo'])
    plt.title('Matriz de Confusión - Tipología de Rutas')
    plt.ylabel('Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
    plt.close()

    # 2. Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'roc_curve.png'))
    plt.close()

    print("Gráficas guardadas en la carpeta 'metrics_plots'.")

if __name__ == "__main__":
    evaluate_model()