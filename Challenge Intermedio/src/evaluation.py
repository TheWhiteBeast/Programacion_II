from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo y calcula Precision, Recall, F1-Score, y Accuracy.
    También genera y guarda las gráficas de Confusion Matrix y Curva ROC.
    """
    try:
        logging.info("Iniciando evaluación del modelo...")
        predictions = model.predict(X_test)
        
        # 1. Calcular Métricas
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions)
        }
        
        logging.info(f"Métricas calculadas: {metrics}")
        
        # Crear carpeta para guardar gráficas temporales si no existe
        os.makedirs("metrics_plots", exist_ok=True)
        
        # 2. Matriz de Confusión
        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benigno", "Maligno"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig("metrics_plots/confusion_matrix.png")
        plt.close()
        logging.info("Matriz de Confusión guardada en metrics_plots/")

        # 3. Curva ROC
        # Necesitamos probabilidades para la curva ROC, no solo predicciones finales
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig("metrics_plots/roc_curve.png")
        plt.close()
        logging.info("Curva ROC guardada en metrics_plots/")
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error durante la evaluación: {e}")
        raise