import os
import pandas as pd
import nltk
from collections import Counter
from nltk.util import ngrams
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Asegurar que se tengan los recursos de nltk
nltk.download('punkt', quiet=True)

def plot_ngrams(tokens_series, n=2, top_k=10, output_path=None):
    """Calcula y grafica la distribución de N-gramas."""
    print(f"\nGenerando distribución de {n}-gramas...")
    all_ngrams = []
    
    for tokens_list in tokens_series:
        # Asegurarnos de que sea una lista (pandas a veces lo lee como string)
        if isinstance(tokens_list, str):
            import ast
            try:
                tokens_list = ast.literal_eval(tokens_list)
            except:
                continue
                
        if isinstance(tokens_list, list) and len(tokens_list) >= n:
            n_grams = list(ngrams(tokens_list, n))
            all_ngrams.extend(n_grams)
            
    # Contar las frecuencias
    ngram_counts = Counter(all_ngrams)
    common_ngrams = ngram_counts.most_common(top_k)
    
    # Preparar datos para la gráfica
    labels = [' '.join(ngram) for ngram, count in common_ngrams]
    counts = [count for ngram, count in common_ngrams]
    
    # Crear la gráfica
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=labels, palette='viridis')
    plt.title(f'Top {top_k} Distribución de {n}-gramas (Bigramas)')
    plt.xlabel('Frecuencia')
    plt.ylabel('N-grama')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"Gráfica guardada en: {output_path}")
    
    plt.close()

def apply_vader_sentiment(df, text_column='0_Original'):
    """Aplica Análisis de Sentimientos usando VADER."""
    print("\nAplicando Análisis de Sentimientos (VADER)...")
    analyzer = SentimentIntensityAnalyzer()
    
    sentiments = []
    for text in df[text_column]:
        if pd.isna(text):
            sentiments.append('Neutral')
            continue
            
        score = analyzer.polarity_scores(str(text))
        compound = score['compound']
        
        # Clasificar basado en el score compuesto
        if compound >= 0.05:
            sentiments.append('Positive')
        elif compound <= -0.05:
            sentiments.append('Negative')
        else:
            sentiments.append('Neutral')
            
    return sentiments

if __name__ == "__main__":
    # Rutas del proyecto
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)               
    data_dir = os.path.join(project_dir, 'data') 
    
    input_file = os.path.join(data_dir, 'glassdoor_reviews_NLP_steps.csv')
    plot_output = os.path.join(data_dir, 'ngrams_distribution.png')
    output_file = os.path.join(data_dir, 'glassdoor_reviews_Sentiment.csv')
    
    print("--- INICIANDO ANÁLISIS DE N-GRAMAS Y SENTIMIENTOS ---")
    
    if not os.path.exists(input_file):
        print(f"Error: No se encontró el dataset en {input_file}")
    else:
        # Cargar datos
        df = pd.read_csv(input_file)
        
        # 1. Distribución de N-gramas (Bigramas en este caso)
        # Usamos los tokens que generamos en el paso 4
        plot_ngrams(df['4_Tokens'], n=2, top_k=10, output_path=plot_output)
        
        # 2. Análisis de Sentimientos con VADER
        df['VADER_Sentiment'] = apply_vader_sentiment(df, text_column='0_Original')
        
        # Guardar dataset con sentimientos
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print("\nResultados de VADER (Distribución de Sentimientos en 'Pros'):")
        print(df['VADER_Sentiment'].value_counts())
        
        print(f"\n¡Éxito! Dataset actualizado guardado en: {output_file}")
        
        # Mostrar unos ejemplos rápidos
        print("\n--- EJEMPLOS DE CLASIFICACIÓN ---")
        print(df[['0_Original', 'VADER_Sentiment']].head(3))