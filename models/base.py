from pathlib import Path
import time, json
import cv2
import numpy as np

RESULT_DIR = Path(__file__).parent.parent / "benchmark" / "results"

def run_model(model, img, model_name, conf, image_path, output_dir=None):
    if output_dir is None:
        output_dir = RESULT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = model(img, conf=conf, iou=0.5, verbose=False, imgsz=1280)
    elapsed = time.time() - t0

    annotated = results[0].plot()
    boxes = results[0].boxes
    names = model.names if hasattr(model, 'names') and model.names else None

    stamp = Path(image_path).stem
    out_path = output_dir / f"{model_name}_{stamp}.png"
    cv2.imwrite(str(out_path), annotated)

    dets = []
    for b in boxes:
        cls_id = int(b.cls[0])
        label = names[cls_id] if names and cls_id in names else f"class_{cls_id}"
        score = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        dets.append({"label": label, "confidence": round(score, 3),
                     "bbox": [x1, y1, x2, y2]})

    info = {
        "model": model_name,
        "image": str(image_path),
        "time_ms": round(elapsed * 1000, 1),
        "detections": len(dets),
        "classes_found": list(set(d["label"] for d in dets)),
        "avg_confidence": round(np.mean([d["confidence"] for d in dets]), 3) if dets else 0,
        "output": str(out_path)
    }

    json_path = output_dir / f"{model_name}_{stamp}.json"
    with open(json_path, "w") as f:
        json.dump(info, f, indent=2)

    return info
