#!/usr/bin/env python3
"""
Benchmark: corre todos los modelos UI sobre una imagen Test.png
y muestra tabla comparativa.
"""
import sys, time, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "benchmark" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from models.base import run_model

def main():
    # Buscar imagen Test
    test_img = None
    for p in [ROOT / "Test.png", ROOT / "test.png", ROOT / "screens" / "Test.png"]:
        if p.exists():
            test_img = p
            break

    if test_img is None:
        print("ERROR: No se encuentra Test.png en el proyecto.")
        print("Colocá una imagen llamada 'Test.png' en la raíz del proyecto.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  BENCHMARK UI DETECTION")
    print(f"  Imagen: {test_img}")
    print(f"{'='*60}\n")

    # Modelos a probar
    modelos = [
        ("Windows UI Locator",    "windows"),
        ("macOS UI Elements",     "macos"),
        ("GPA-GUI-Detector",      "multiplatform"),
        ("macOS UI Groups",       "groups"),
        ("Web Form Detection",    "web_form"),
    ]

    resultados = []
    for nombre, mod_name in modelos:
        print(f"▶  {nombre}...")
        try:
            mod = __import__(f"models.{mod_name}", fromlist=["run"])
            t0 = time.time()
            info = mod.run(str(test_img), RESULTS_DIR)
            t_total = time.time() - t0
            info["load_time_ms"] = round(t_total * 1000 - info["time_ms"], 1)
            resultados.append(info)
            print(f"   ✓ {info['detections']} detecciones en {info['time_ms']}ms\n")
        except Exception as e:
            print(f"   ✗ ERROR: {e}\n")
            resultados.append({
                "model": nombre,
                "error": str(e),
                "detections": -1,
                "time_ms": 0,
                "classes_found": [],
                "avg_confidence": 0
            })

    # --- Tabla comparativa ---
    print(f"\n{'='*60}")
    print(f"  RESULTADOS")
    print(f"{'='*60}")
    print(f"{'Modelo':<25} {'Dets':>6} {'Tiempo':>8} {'Conf':>7}  {'Clases'}")
    print(f"{'-'*25} {'-'*6} {'-'*8} {'-'*7}  {'-'*30}")

    resultados.sort(key=lambda r: r.get("detections", -1), reverse=True)

    ganador = resultados[0] if resultados else None

    for r in resultados:
        dets = r.get("detections", -1)
        if dets < 0:
            print(f"{r['model']:<25} {'ERROR':>6} {'':>8} {'':>7}  {r.get('error', '')[:40]}")
        else:
            t = r.get("time_ms", 0)
            conf = r.get("avg_confidence", 0)
            cls = ", ".join(r.get("classes_found", []))[:40]
            print(f"{r['model']:<25} {dets:>6} {t:>7.0f}ms {conf:>6.2f}  {cls}")

    print(f"\n{'='*60}")
    if ganador and ganador.get("detections", -1) >= 0:
        print(f"  🏆 MEJOR: {ganador['model']} ({ganador['detections']} detecciones)")
    print(f"{'='*60}\n")

    # Mostrar paths de outputs
    print("Outputs guardados en:")
    for r in resultados:
        if "output" in r:
            print(f"  • {r['output']}")

    # Guardar resultados completos como JSON
    res_file = RESULTS_DIR / "benchmark_results.json"
    with open(res_file, "w") as f:
        json.dump(resultados, f, indent=2)
    print(f"\nResultados JSON: {res_file}")

if __name__ == "__main__":
    main()
