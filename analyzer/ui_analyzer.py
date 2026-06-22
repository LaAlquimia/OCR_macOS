#!/usr/bin/env python3
"""
UI Analyzer v2 — Pipeline completo de detección multi-modelo + OCR.
Corre TODOS los modelos sobre la imagen completa y fusiona resultados.
"""

import os, sys, json, time, tempfile, subprocess
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = Path.home() / ".cache" / "yolo_ui"
OCR_BIN = PROJECT_ROOT / "ocr_bin"
OUTPUT_DIR = PROJECT_ROOT / "analyzer" / "results"

# --- Model configs ---
MODELS = [
    {
        "name": "GPA-GUI-Detector",
        "key": "gpa",
        "repo": "Salesforce/GPA-GUI-Detector",
        "file": "model.pt",
        "save_as": "gpa-gui-detector.pt",
        "conf": 0.05,
        "weight": 1,
    },
    {
        "name": "Windows UI Locator",
        "key": "windows",
        "repo": "IndextDataLab/windows-ui-locator",
        "file": "best.pt",
        "save_as": "windows-ui-locator.pt",
        "conf": 0.30,
        "weight": 5,
    },
    {
        "name": "macOS UI Elements",
        "key": "macos",
        "repo": "macpaw-research/yolov11l-ui-elements-detection",
        "file": "ui-elements-detection.pt",
        "save_as": None,
        "conf": 0.30,
        "weight": 4,
    },
    {
        "name": "macOS UI Groups",
        "key": "groups",
        "repo": "macpaw-research/yolov11l-ui-groups-detection",
        "file": "ui-groups-detection.pt",
        "save_as": None,
        "conf": 0.30,
        "weight": 3,
    },
    {
        "name": "Web Form Detection",
        "key": "webform",
        "repo": "foduucom/web-form-ui-field-detection",
        "file": "best.pt",
        "save_as": "web-form-ui-field-detection.pt",
        "conf": 0.25,
        "weight": 2,
    },
]

IOU_THRESH = 0.5
IMGSZ = 1280
OCR_PADDING = 4
IOU_MERGE = 0.3

# --- Colors por clase ---
CLASS_COLORS = {
    "AXButton": (0, 180, 255),
    "AXImage": (200, 0, 200),
    "AXLink": (100, 200, 255),
    "AXTextArea": (255, 200, 0),
    "AXDisclosureTriangle": (0, 200, 100),
    "AXCell": (0, 200, 200),
    "button": (0, 180, 255),
    "textbox": (255, 200, 0),
    "checkbox": (0, 255, 100),
    "dropdown": (255, 100, 0),
    "icon": (200, 0, 200),
    "tab": (100, 200, 255),
    "menu_item": (0, 200, 200),
    "unknown": (128, 128, 128),
}
DEFAULT_COLOR = (128, 128, 128)


# =============================================================
#  MODEL LOADING
# =============================================================

def ensure_model(cfg):
    if cfg["save_as"]:
        path = CACHE_DIR / cfg["save_as"]
    else:
        path = CACHE_DIR / cfg["file"]
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"  Descargando {cfg['name']}...")
        hf_hub_download(repo_id=cfg["repo"], filename=cfg["file"],
                        local_dir=CACHE_DIR, local_dir_use_symlinks=False)
        if cfg["save_as"] and (CACHE_DIR / cfg["file"]).exists():
            (CACHE_DIR / cfg["file"]).rename(path)
    return path


def load_models(model_configs):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Dispositivo: {device}\n")

    loaded = []
    for cfg in model_configs:
        path = ensure_model(cfg)
        print(f"  Cargando {cfg['name']}...")
        m = YOLO(str(path)).to(device)
        loaded.append({**cfg, "model": m})
    return loaded


# =============================================================
#  DETECTION — run all models on full image
# =============================================================

def run_detections(model_cfgs, img):
    all_dets = []
    for cfg in model_cfgs:
        model = cfg["model"]
        results = model(img, conf=cfg["conf"], iou=IOU_THRESH,
                        verbose=False, imgsz=IMGSZ)
        boxes = results[0].boxes
        names = model.names if hasattr(model, "names") and model.names else {}
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            w, h = x2 - x1, y2 - y1
            if w < 5 or h < 5:
                continue
            cls_id = int(b.cls[0])
            label = names.get(cls_id, f"class_{cls_id}")
            all_dets.append({
                "source": cfg["key"],
                "source_name": cfg["name"],
                "class": label,
                "confidence": float(b.conf[0]),
                "bbox": [x1, y1, x2, y2],
                "width": w,
                "height": h,
                "weight": cfg["weight"],
            })
    return all_dets


# =============================================================
#  MERGE — spatial fusion of all detections
# =============================================================

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xi1, yi1 = max(ax1, bx1), max(ay1, by1)
    xi2, yi2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0


def merge_detections(all_dets):
    merged = []
    used = set()

    sorted_dets = sorted(all_dets, key=lambda d: -d["confidence"])

    for i, det in enumerate(sorted_dets):
        if i in used:
            continue
        used.add(i)

        group = [det]
        for j, other in enumerate(sorted_dets):
            if j in used or j == i:
                continue
            if iou(det["bbox"], other["bbox"]) >= IOU_MERGE:
                used.add(j)
                group.append(other)

        group.sort(key=lambda d: -d["weight"])
        best = group[0]

        cls_counts = {}
        for g in group:
            cls_counts[g["class"]] = cls_counts.get(g["class"], 0) + g["weight"]
        best_class = max(cls_counts, key=cls_counts.get)

        combined_bbox = group[0]["bbox"][:]
        for g in group[1:]:
            combined_bbox[0] = min(combined_bbox[0], g["bbox"][0])
            combined_bbox[1] = min(combined_bbox[1], g["bbox"][1])
            combined_bbox[2] = max(combined_bbox[2], g["bbox"][2])
            combined_bbox[3] = max(combined_bbox[3], g["bbox"][3])

        merged.append({
            "class": best_class,
            "confidence": round(best["confidence"], 3),
            "bbox": combined_bbox,
            "width": combined_bbox[2] - combined_bbox[0],
            "height": combined_bbox[3] - combined_bbox[1],
            "sources": [g["source_name"] for g in group],
            "detections": len(group),
        })

    return merged


# =============================================================
#  OCR
# =============================================================

def ocr_crop(img, bbox):
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - OCR_PADDING)
    y1 = max(0, y1 - OCR_PADDING)
    x2 = min(img.shape[1], x2 + OCR_PADDING)
    y2 = min(img.shape[0], y2 + OCR_PADDING)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, crop)
        result = subprocess.run(
            [str(OCR_BIN), tmp_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return ""
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# =============================================================
#  OUTPUT GENERATION
# =============================================================

def draw_annotations(img, elements):
    for el in elements:
        x1, y1, x2, y2 = el["bbox"]
        label = el["class"]
        text = el.get("text", "")
        conf = el["confidence"]
        color = CLASS_COLORS.get(label, DEFAULT_COLOR)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label_str = f"{label} {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x1, y1 - lh - 6), (x1 + lw + 6, y1), color, -1)
        cv2.putText(img, label_str, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if text:
            display_text = text[:50].replace("\n", " ")
            cv2.putText(img, display_text, (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 255, 220), 1)
    return img


def write_txt_report(elements, path, image_path, elapsed_ms, model_summary):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sorted_el = sorted(elements, key=lambda e: (e["bbox"][1], e["bbox"][0]))

    lines = [
        "=" * 80,
        "  UI ANALYZER REPORT v2 — Multi-Model Fusion",
        "=" * 80,
        f"  Image:       {image_path}",
        f"  Date:        {timestamp}",
        f"  Time:        {elapsed_ms:.0f} ms",
        f"  Total:       {len(elements)} elements",
        "",
        "  Models used:",
    ]
    for ms in model_summary:
        lines.append(f"    • {ms['name']:<30} {ms['dets']:>3} detections  (conf≥{ms['conf']})")
    lines += [
        "",
        "=" * 80,
        "  DETECTED ELEMENTS (sorted top-to-bottom, left-to-right)",
        "=" * 80,
        "",
    ]

    for i, el in enumerate(sorted_el, 1):
        b = el["bbox"]
        w, h = b[2] - b[0], b[3] - b[1]
        text = el.get("text", "").replace("\n", " ")[:50]
        sources = ", ".join(el.get("sources", ["?"]))
        lines.append(
            f"  #{i:04d}  {el['class']:<20} | {el['confidence']:.2f}  "
            f'| "{text:<48}" | ({b[0]:>4},{b[1]:>4},{b[2]:>4},{b[3]:>4})  '
            f"| {w:>3}x{h:<3} | {sources}"
        )

    lines += [
        "",
        "=" * 80,
        "  CLASS SUMMARY",
        "=" * 80,
    ]
    counts = {}
    for el in elements:
        cls = el["class"]
        counts[cls] = counts.get(cls, 0) + 1
    for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = count / len(elements) * 100
        lines.append(f"    {cls:<25} {count:>3}  ({pct:>5.1f}%)")
    lines.append("")
    lines.append("=" * 80)
    lines.append("  END OF REPORT")
    lines.append("=" * 80)

    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_json_report(elements, path, image_path, elapsed_ms, model_summary):
    timestamp = datetime.now().isoformat()
    counts = {}
    for el in elements:
        cls = el["class"]
        counts[cls] = counts.get(cls, 0) + 1

    report = {
        "metadata": {
            "image": str(image_path),
            "timestamp": timestamp,
            "total_elements": len(elements),
            "elapsed_ms": round(elapsed_ms, 1),
            "models": model_summary,
        },
        "elements": elements,
        "class_summary": counts,
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


# =============================================================
#  SCREENSHOT
# =============================================================

def capture_screenshot():
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        ret = subprocess.run(["screencapture", "-i", tmp_path], check=False)
        if ret.returncode != 0:
            print("Captura cancelada.")
            return None
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            print("No se capturó nada.")
            return None
        return cv2.imread(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# =============================================================
#  MAIN
# =============================================================

def analyze(image_path=None, output_dir=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if image_path:
        img = cv2.imread(str(image_path))
        src_path = image_path
        if img is None:
            print(f"ERROR: no se pudo leer {image_path}")
            sys.exit(1)
    else:
        print("Seleccioná un área de la pantalla (cursor en cruz)...")
        img = capture_screenshot()
        if img is None:
            sys.exit(1)
        src_path = "screenshot"

    h, w = img.shape[:2]
    print(f"\nImagen: {w}x{h}px\n")

    model_cfgs = load_models(MODELS)

    t0 = time.time()

    print(f"\nCorriendo {len(model_cfgs)} modelos sobre la imagen completa...")
    all_dets = run_detections(model_cfgs, img)
    elapsed_det = (time.time() - t0) * 1000

    model_summary = []
    for cfg in model_cfgs:
        count = sum(1 for d in all_dets if d["source"] == cfg["key"])
        model_summary.append({"name": cfg["name"], "key": cfg["key"],
                              "dets": count, "conf": cfg["conf"]})

    print(f"\nDetecciones por modelo:")
    for ms in model_summary:
        print(f"  {ms['name']:<30} {ms['dets']:>3}")
    print(f"  {'─'*40}")
    print(f"  {'Total (raw)':<30} {len(all_dets):>3}")

    print(f"\nFusionando detecciones por solapamiento espacial (IoU≥{IOU_MERGE})...")
    merged = merge_detections(all_dets)
    print(f"  → {len(merged)} elementos únicos después de fusión\n")

    print(f"Aplicando OCR a cada elemento...")
    for i, el in enumerate(merged):
        text = ocr_crop(img, el["bbox"])
        merged[i]["text"] = text
        if (i + 1) % 10 == 0 or (i + 1) == len(merged):
            print(f"    OCR: {i+1}/{len(merged)}")

    elapsed_ms = (time.time() - t0) * 1000

    stem = f"ui_analysis_{time.strftime('%Y%m%d_%H%M%S')}"

    annotated = draw_annotations(img.copy(), merged)
    img_path = output_dir / f"{stem}.png"
    cv2.imwrite(str(img_path), annotated)

    txt_path = output_dir / f"{stem}.txt"
    write_txt_report(merged, txt_path, src_path, elapsed_ms, model_summary)

    json_path = output_dir / f"{stem}.json"
    write_json_report(merged, json_path, src_path, elapsed_ms, model_summary)

    counts = {}
    for el in merged:
        cls = el["class"]
        counts[cls] = counts.get(cls, 0) + 1

    print(f"\n{'='*60}")
    print(f"  ANALISIS COMPLETO — {len(merged)} elementos en {elapsed_ms:.0f}ms")
    print(f"  (detección: {elapsed_det:.0f}ms, OCR: {elapsed_ms - elapsed_det:.0f}ms)")
    print(f"{'='*60}")
    for cls, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = count / len(merged) * 100
        print(f"    {cls:<25} {count:>3}  ({pct:>5.1f}%)")
    print(f"{'='*60}")
    print(f"\n  Outputs:")
    print(f"    Imagen:  {img_path}")
    print(f"    Report:  {txt_path}")
    print(f"    JSON:    {json_path}")

    subprocess.run(["open", str(img_path)])
    subprocess.run(["open", str(txt_path)])

    return merged


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="UI Analyzer v2 — multi-model detection + OCR")
    parser.add_argument("--image", "-i", help="Path to screenshot")
    parser.add_argument("--output", "-o", help="Output directory")
    args = parser.parse_args()
    analyze(image_path=args.image, output_dir=args.output)


if __name__ == "__main__":
    main()
