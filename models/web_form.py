from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import torch
from .base import run_model

CACHE_DIR = Path.home() / ".cache" / "yolo_ui"
REPO = "foduucom/web-form-ui-field-detection"
MODEL_PATH = CACHE_DIR / "web-form-ui-field-detection.pt"

def ensure_model():
    if not MODEL_PATH.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print("  Descargando Web Form UI Detection...")
        hf_hub_download(repo_id=REPO, filename="best.pt",
                        local_dir=CACHE_DIR, local_dir_use_symlinks=False)
        (CACHE_DIR / "best.pt").rename(MODEL_PATH)

def run(image_path, output_dir=None):
    ensure_model()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(MODEL_PATH)).to(device)
    import cv2
    img = cv2.imread(str(image_path))
    return run_model(model, img, "web_form", 0.25, image_path, output_dir)
