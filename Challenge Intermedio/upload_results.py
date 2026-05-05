import os
import sys

def upload_to_github():
    try:
        print("--- Iniciando proceso de carga a GitHub ---")
        # Agregamos todos los archivos
        os.system("git add .")
        
        # Pedimos un mensaje de commit o usamos uno por defecto
        message = "Resultados del challenge - Cancer Detection"
        os.system(f'git commit -m "{message}"')
        
        # Subimos a la rama principal (main o master)
        print("Subiendo archivos...")
        os.system("git push origin main") 
        
        print("\n ¡Éxito! Tus resultados han sido subidos a:")
        print("https://github.com/TheWhiteBeast/Programacion_II")
        
    except Exception as e:
        print(f"X Error al subir a GitHub: {e}")

if __name__ == "__main__":
    upload_to_github()