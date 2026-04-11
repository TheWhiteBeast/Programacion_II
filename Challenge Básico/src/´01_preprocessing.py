import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Descargar las herramientas necesarias de NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True) 

def load_reviews_from_repo(file_name="glassdoor_reviews.csv"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)               
    data_dir = os.path.join(project_dir, 'data')             
    file_path = os.path.join(data_dir, file_name)
    
    if not os.path.exists(file_path):
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(file_path, encoding='cp1252')
        return df
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    df_reviews = load_reviews_from_repo()
    
    if not df_reviews.empty:
        # Tomamos solo la columna Pros para hacer las demostraciones y capturas
        df_nlp = df_reviews[['Pros']].copy()
        df_nlp = df_nlp.dropna(subset=['Pros'])
        
        # Guardamos el original para comparar
        df_nlp['0_Original'] = df_nlp['Pros']
        
        # ---------------------------------------------------------
        # PASO 1: Minúsculas y retirar signos de puntuación
        # ---------------------------------------------------------
        def step1_clean(text):
            text = str(text).lower()
            text = re.sub(r'[^\w\s]', '', text) 
            return text
            
        df_nlp['1_Lower_Punct'] = df_nlp['0_Original'].apply(step1_clean)
        
        print("\n" + "="*80)
        print("CAPTURA 1: TEXTO A MINUSCULAS Y SIN PUNTUACION")
        print("="*80)
        print(df_nlp[['1_Lower_Punct']].head(5))


        # ---------------------------------------------------------
        # PASO 2: Retirar stop words
        # ---------------------------------------------------------
        stop_words = set(stopwords.words('english'))
        
        def step2_stopwords(text):
            words = text.split()
            filtered = [word for word in words if word not in stop_words]
            return " ".join(filtered)
            
        df_nlp['2_StopWords'] = df_nlp['1_Lower_Punct'].apply(step2_stopwords)
        
        print("\n" + "="*80)
        print("CAPTURA 2: RETIRAR STOP WORDS")
        print("="*80)
        print(df_nlp[['2_StopWords']].head(5))


        # ---------------------------------------------------------
        # PASO 3: Realizar Lemmatization
        # ---------------------------------------------------------
        lemmatizer = WordNetLemmatizer()
        
        def step3_lemmatize(text):
            words = text.split()
            lemmatized = [lemmatizer.lemmatize(word) for word in words]
            return " ".join(lemmatized)
            
        df_nlp['3_Lemmatization'] = df_nlp['2_StopWords'].apply(step3_lemmatize)
        
        print("\n" + "="*80)
        print("CAPTURA 3: LEMMATIZATION")
        print("="*80)
        print(df_nlp[['3_Lemmatization']].head(5))


        # ---------------------------------------------------------
        # PASO 4: Tokenizar el texto
        # ---------------------------------------------------------
        def step4_tokenize(text):
            return word_tokenize(text)
            
        df_nlp['4_Tokens'] = df_nlp['3_Lemmatization'].apply(step4_tokenize)
        
        print("\n" + "="*80)
        print("CAPTURA 4: TOKENIZACION")
        print("="*80)
        print(df_nlp[['4_Tokens']].head(5))


        # ---------------------------------------------------------
        # PASO 5: Añadir los PoS (Part of Speech)
        # ---------------------------------------------------------
        def step5_pos_tagging(tokens):
            return pos_tag(tokens)
            
        df_nlp['5_PoS_Tags'] = df_nlp['4_Tokens'].apply(step5_pos_tagging)
        
        print("\n" + "="*80)
        print("CAPTURA 5: PART OF SPEECH (PoS)")
        print("="*80)
        print(df_nlp[['5_PoS_Tags']].head(5))
        
        
        # --- Guardar el archivo final con todas las etapas ---
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(current_dir)               
        clean_data_path = os.path.join(project_dir, 'data', 'glassdoor_reviews_NLP_steps.csv')
        
        df_nlp.to_csv(clean_data_path, index=False, encoding='utf-8')
        print("\nArchivo guardado exitosamente para la entrega.")