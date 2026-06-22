#!/usr/bin/env python3
import os
import sys
import subprocess
import tempfile
import time

def main():
    # 1. Crear un path temporal para la captura de pantalla
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        temp_image_path = tmpfile.name

    try:
        print("Selecciona un área de la pantalla para hacer OCR (el cursor cambiará a una cruz)...")
        # 2. Capturar pantalla interactivamente mediante screencapture -i
        # Esto permite al usuario seleccionar el área.
        result = subprocess.run(["screencapture", "-i", temp_image_path], check=False)
        
        if result.returncode != 0:
            print("Captura cancelada o fallida.")
            return

        # Verificar si el archivo realmente se creó y tiene tamaño
        if not os.path.exists(temp_image_path) or os.path.getsize(temp_image_path) == 0:
            print("No se capturó ninguna imagen.")
            return

        print("Procesando imagen con el OCR nativo de macOS (Apple Vision Framework)...")
        # 3. Llamar al binario compilado en Swift
        swift_bin_path = "/Users/laalquimia/ocr_opencv/ocr_bin"
        ocr_result = subprocess.run([swift_bin_path, temp_image_path], capture_output=True, text=True, check=False)
        
        if ocr_result.returncode != 0:
            print("Error ejecutando el OCR.")
            print(ocr_result.stderr)
            return

        output_text = ocr_result.stdout.strip()
        
        if output_text:
            print("\n--- TEXTO DETECTADO ---")
            print(output_text)
            print("-----------------------")
            
            # 4. Copiar automáticamente el texto al portapapeles
            try:
                subprocess.run(["pbcopy"], input=output_text, text=True, check=True)
                print("¡Texto copiado al portapapeles de tu MacBook!")
            except Exception as e:
                print(f"No se pudo copiar al portapapeles: {e}")
        else:
            print("No se reconoció ningún texto en el área seleccionada.")

    finally:
        # Limpiar el archivo temporal
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

if __name__ == "__main__":
    main()
