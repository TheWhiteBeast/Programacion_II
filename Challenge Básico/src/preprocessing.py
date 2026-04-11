import os
import time
import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

def scrape_glassdoor_stealth(url, max_pages=5):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)               
    data_dir = os.path.join(project_dir, 'data')             
    os.makedirs(data_dir, exist_ok=True)
    
    options = uc.ChromeOptions()
    
    print("Iniciando navegador indetectable...")
    driver = uc.Chrome(options=options)
    reviews_data = []
    
    try:
        print(f"Accediendo a la URL: {url}")
        driver.get(url)
        
        print("\n" + "="*50)
        print("MODO MANUAL: TIENES 60 SEGUNDOS")
        print("1. Inicia sesion en Glassdoor (el CAPTCHA deberia dejarte pasar).")
        print("2. Quedate en la pagina de resenas de Continental.")
        print("="*50 + "\n")
        
        for i in range(60, 0, -1):
            if i % 10 == 0 or i <= 5:
                print(f"{i} segundos restantes...")
            time.sleep(1)
            
        print("\nTIEMPO TERMINADO. El bot retoma el control.")
        
        for pagina in range(1, max_pages + 1):
            print(f"\nExtrayendo datos de la pagina {pagina}...")
            time.sleep(3) 
            
            titulos_pros = driver.find_elements(By.XPATH, "//p[contains(text(), 'Ventajas') or contains(text(), 'Pros')]")
            print(f"Se detectaron {len(titulos_pros)} resenas en esta pagina.")

            for p_pro in titulos_pros:
                try:
                    pros_text = p_pro.find_element(By.XPATH, "./following-sibling::*").text
                    contenedor_padre = p_pro.find_element(By.XPATH, "../..")
                    cons_text = contenedor_padre.find_element(By.XPATH, ".//p[contains(text(), 'Desventajas') or contains(text(), 'Contras') or contains(text(), 'Cons')]/following-sibling::*").text
                    
                    if pros_text and cons_text:
                        reviews_data.append({
                            "Pros": pros_text,
                            "Cons": cons_text,
                            "Language": "unknown" 
                        })
                except Exception:
                    continue
            
            if pagina < max_pages:
                try:
                    print("Buscando el boton 'Siguiente'...")
                    btn_siguiente = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Next' or contains(@class, 'nextButton')] | //span[contains(text(), 'Siguiente')]/.."))
                    )
                    driver.execute_script("arguments[0].click();", btn_siguiente)
                except TimeoutException:
                    print("No se encontro el boton 'Siguiente'. Terminando extraccion.")
                    break
            
    except Exception as e:
        print(f"Ocurrio un error general: {e}")
        
    finally:
        print(f"\nCerrando el navegador. Total extraidas: {len(reviews_data)}")
        driver.quit()
    
    df = pd.DataFrame(reviews_data)
    return df, data_dir

if __name__ == "__main__":
    url_target = "https://www.glassdoor.com.mx/Evaluaciones/Continental-Evaluaciones-E3768.htm"
    df_results, folder_path = scrape_glassdoor_stealth(url_target, max_pages=5) 
    
    if not df_results.empty:
        csv_path = os.path.join(folder_path, "glassdoor_reviews.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"\nEXITO: Archivo CSV generado en: {csv_path}")
        print(df_results.head())
    else:
        print("\nEl DataFrame esta vacio.")