import os

def upload_to_github():
    print("--- Iniciando proceso de carga a GitHub ---")
    
    # Pedimos un mensaje dinámico en la terminal
    mensaje_usuario = input("Escribe el mensaje del commit (o da Enter para usar el default): ")
    message = mensaje_usuario if mensaje_usuario else "Resultados Actualizados Challenge Final"
    
    # Ejecutamos los comandos de Git
    os.system("git add .")
    os.system(f'git commit -m "{message}"')
    
    print(f"Subiendo archivos con el mensaje: '{message}'...")
    os.system("git push origin main") 
    
    print("\n ¡Éxito! Tus resultados han sido subidos a:")
    print("https://github.com/TheWhiteBeast/Programacion_II")

if __name__ == "__main__":
    upload_to_github()