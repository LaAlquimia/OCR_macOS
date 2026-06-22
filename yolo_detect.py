#!/usr/bin/env python3
"""
Detecta elementos UI en screenshots de Windows, macOS o web.
Usa modelo YOLO pre-entrenado (OmniParser/Salesforce).
Ejecución en Mac con aceleración Metal.
"""

import subprocess, sys, os, tempfile, time
from pathlib import Path

deps = ["ultralytics", "opencv-python-headless", "mss", "numpy", "huggingface_hub"]
try:
    import cv2, numpy as np
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Instalando dependencias (1 vez)...")
    subprocess.run([sys.executable, "-m", "pip", "install", *deps], check=True)
    import cv2, numpy as np
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download

import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
if device == "mps":
    print("Usando aceleración Metal (Apple Silicon)")

# --- Elegí el modelo según tu caso ---
#
# Opción A: Windows UI (botones, inputs, íconos Windows)
#   https://huggingface.co/IndextDataLab/windows-ui-locator
#   Clases: button, textbox, checkbox, dropdown, icon, tab, menu_item
#   YOLO11s → rápido (~50ms en M2)
#
# Opción B: Multiplataforma (cualquier OS, finetuned de OmniParser)
#   https://huggingface.co/Salesforce/GPA-GUI-Detector
#   Detecta elementos interactivos en general (íconos, botones, etc.)
#   Sin clases fijas públicas — usa el detector de OmniParser
#
# Opción C: macOS nativo (solo para screenshots de Mac)
#   https://huggingface.co/macpaw-research/yolov11l-ui-elements-detection
#   Clases: AXButton, AXImage, AXLink, AXTextArea, AXDisclosureTriangle
#   YOLOv11l → más pesado

MODELO = "macos"  # cambia a "windows" o "multiplatform" si querés

CACHE_DIR = Path.home() / ".cache" / "yolo_ui"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SCREENS_DIR = Path(__file__).parent / "screens"
SCREENS_DIR.mkdir(parents=True, exist_ok=True)

if MODELO == "windows":
    # Windows UI Locator — YOLO11s, entrenado en UI de Windows
    repo = "IndextDataLab/windows-ui-locator"
    archivo = "best.pt"
    model_path = CACHE_DIR / "windows-ui-locator.pt"
    if not model_path.exists():
        print("Descargando modelo Windows UI Locator (YOLO11s, rápido)...")
        hf_hub_download(repo_id=repo, filename=archivo, local_dir=CACHE_DIR,
                        local_dir_use_symlinks=False)
        (CACHE_DIR / archivo).rename(model_path)
    conf = 0.3

elif MODELO == "multiplatform":
    # GPA-GUI-Detector — de Salesforce/OmniParser, funciona en cualquier OS
    repo = "Salesforce/GPA-GUI-Detector"
    archivo = "model.pt"
    model_path = CACHE_DIR / "gpa-gui-detector.pt"
    if not model_path.exists():
        print("Descargando GPA-GUI-Detector (OmniParser)...")
        hf_hub_download(repo_id=repo, filename=archivo, local_dir=CACHE_DIR,
                        local_dir_use_symlinks=False)
        (CACHE_DIR / archivo).rename(model_path)
    conf = 0.05  # OmniParser recomienda threshold bajo

else:  # macos
    repo = "macpaw-research/yolov11l-ui-elements-detection"
    model_path = CACHE_DIR / "ui-elements-detection.pt"
    if not model_path.exists():
        print("Descargando modelo macOS UI (YOLOv11l)...")
        hf_hub_download(repo_id=repo, filename="ui-elements-detection.pt",
                        local_dir=CACHE_DIR, local_dir_use_symlinks=False)
    conf = 0.3

try:
    model = YOLO(str(model_path))
    model.to(device)
    print(f"Modelo cargado: {model_path.name}")
except Exception as e:
    print(f"ERROR al cargar el modelo: {e}")
    sys.exit(1)

# --- Captura de pantalla ---
tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
tmp.close()
try:
    print("Seleccioná un área de la pantalla (cursor en cruz)...")
    subprocess.run(["screencapture", "-i", tmp.name], check=True)
    img = cv2.imread(tmp.name)
finally:
    if os.path.exists(tmp.name):
        os.remove(tmp.name)

if img is None or img.size == 0:
    print("No se capturó nada.")
    sys.exit(1)

# --- Inferencia ---
t0 = time.time()
try:
    results = model(img, conf=conf, iou=0.5, verbose=False, imgsz=1280)
except Exception as e:
    print(f"ERROR durante inferencia: {e}")
    sys.exit(1)
elapsed = time.time() - t0

annotated = results[0].plot()
boxes = results[0].boxes

print(f"\n⏱  {elapsed*1000:.0f}ms  |  {img.shape[1]}x{img.shape[0]}px")

if len(boxes) == 0:
    print("No se detectaron elementos.")
else:
    # Intentar obtener nombres de clases
    names = model.names if hasattr(model, 'names') and model.names else None
    print(f"\n--- {len(boxes)} elemento(s) detectado(s) ---")
    lines = []
    for i, b in enumerate(boxes):
        cls_id = int(b.cls[0])
        label = names[cls_id] if names and cls_id in names else f"class_{cls_id}"
        score = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        print(f"  [{i}] {label:<20} {score:.2f}  ({x1},{y1},{x2},{y2})")
        lines.append(f"{label} ({score:.2f})")
    subprocess.run(["pbcopy"], input="\n".join(lines), text=True)
    print("(copiado al portapapeles)")

timestamp = time.strftime("%Y%m%d_%H%M%S")
out = str(SCREENS_DIR / f"detect_{timestamp}.png")
cv2.imwrite(out, annotated)
print(f"\nResultado: {out}")
subprocess.run(["open", out])
