import os

def upload_to_github():
    print("--- INICIANDO SINCRONIZACION CON GITHUB ---")
    
    print("Ejecutando: git add .")
    os.system("git add .")
    
    print("Ejecutando: git commit")
    os.system('git commit -m "Resultados del challenge NLP Glassdoor y pipelines MLOps"')
    
    print("Ejecutando: git push origin main")
    os.system("git push origin main")
    
    print("--- PROCESO FINALIZADO ---")

if __name__ == "__main__":
    upload_to_github()