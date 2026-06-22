from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import torch
from .base import run_model

CACHE_DIR = Path.home() / ".cache" / "yolo_ui"
MODEL_PATH = CACHE_DIR / "ui-groups-detection.pt"

def ensure_model():
    if not MODEL_PATH.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print("  Descargando macOS UI Groups...")
        hf_hub_download(repo_id="macpaw-research/yolov11l-ui-groups-detection",
                        filename="ui-groups-detection.pt",
                        local_dir=CACHE_DIR, local_dir_use_symlinks=False)

def run(image_path, output_dir=None):
    ensure_model()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(MODEL_PATH)).to(device)
    import cv2
    img = cv2.imread(str(image_path))
    return run_model(model, img, "groups", 0.3, image_path, output_dir)
