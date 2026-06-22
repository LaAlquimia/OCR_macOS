# OCR & UI Detection Toolkit

Suite de herramientas para detección de elementos UI y OCR en macOS, combinando modelos YOLO (ultralytics) con el framework Vision de Apple.

## Arquitectura

```
                    ┌─────────────────────┐
                    │   ocr_mac.py        │  OCR rápido (screenshot → texto)
                    └──────────┬──────────┘
                               │
                               v
                    ┌─────────────────────┐
                    │   ocr_bin (Swift)   │  Vision Framework OCR
                    └─────────────────────┘

                    ┌─────────────────────┐
                    │   yolo_detect.py     │  Detección con 1 modelo
                    │  (screenshot→YOLO)   │
                    └─────────────────────┘

                    ┌──────────────────────────────────┐
                    │ analyzer/ui_analyzer.py           │  Pipeline completo
                    │  5 modelos → fusión IoU → OCR     │
                    └──┬────┬────┬────┬────┬───────────┘
                       │    │    │    │    │
                 ┌─────┴──┐ │ ┌──┴──┐ │ ┌──┴──────────┐
                 │windows │ │ │macos│ │ │  web_form    │
                 ├────────┤ │ ├─────┤ │ ├─────────────┤
                 │multipl.│ │ │groups│ │ │  base.py     │
                 └────────┘ │ └─────┘ │ └─────────────┘
                           └─────────┘

                    ┌─────────────────────┐
                    │ benchmark/benchmark  │  Compara los 5 modelos
                    └─────────────────────┘
```

---

## Scripts

### `yolo_detect.py`
Detección UI con un solo modelo. Toma screenshot, corre inferencia YOLO, guarda imagen anotada en `screens/`.

```bash
.env/bin/python yolo_detect.py
```

Configurar modelo en línea 45:
```python
MODELO = "windows"   # Windows UI Locator (7 clases UI)
MODELO = "macos"     # macOS UI Elements (5 clases AX)
MODELO = "multiplatform"  # GPA-GUI-Detector (detección genérica)
```

### `ocr_mac.py`
OCR nativo de macOS. Toma screenshot, extrae texto con Vision Framework y lo copia al portapapeles.

```bash
.env/bin/python ocr_mac.py
```

### `ocr.swift`
Código fuente del binario `ocr_bin`. Usa `VNRecognizeTextRequest` con nivel `.accurate`.

Compilar:
```bash
swiftc -o ocr_bin ocr.swift -framework Vision -framework AppKit -framework Foundation
```

### `analyzer/ui_analyzer.py`
Pipeline completo: corre los 5 modelos, fusiona detecciones por IoU, aplica OCR a cada elemento y genera reportes.

```bash
# Con imagen existente
.env/bin/python analyzer/ui_analyzer.py --image Test.png

# Tomando screenshot
.env/bin/python analyzer/ui_analyzer.py
```

**Salida** en `analyzer/results/`:
- `ui_analysis_*.png` — imagen anotada con bounding boxes + texto
- `ui_analysis_*.txt` — reporte legible
- `ui_analysis_*.json` — datos estructurados

**Fusión multi-modelo**: las detecciones se fusionan por IoU ≥ 0.3. La clase se elige por voto ponderado (pesos: Windows=5, macOS=4, Groups=3, WebForm=2, GPA=1).

### `benchmark/benchmark.py`
Corre los 5 modelos sobre `Test.png` y muestra tabla comparativa.

```bash
.env/bin/python benchmark/benchmark.py
```

---

## Modelos

| Modelo | HF Repo | Clases | Confianza | Peso fusión |
|---|---|---|---|---|
| **GPA-GUI-Detector** | Salesforcedotcom/GPA-GUI-Detector | icon (genérico) | 0.05 | 1 |
| **Windows UI Locator** | IndexdDataLab/windows-ui-locator | button, textbox, checkbox, dropdown, icon, tab, menu_item | 0.30 | 5 |
| **macOS UI Elements** | macpaw-research/yolov11l-ui-elements-detection | AXButton, AXImage, AXLink, AXTextArea, AXDisclosureTriangle | 0.30 | 4 |
| **macOS UI Groups** | macpaw-research/yolov11l-ui-groups-detection | AXCell, AXGroup, AXList, etc. | 0.30 | 3 |
| **Web Form Detection** | foduucom/web-form-ui-field-detection | checkbox, radio button, dropdown, textbox, button, etc. | 0.25 | 2 |

---

## Outputs

### Reporte `.txt`
```
  #0001  button                | 0.87  | "Send"              | (120, 340, 200, 370)  |  80x30  | Windows UI Locator
  #0002  AXButton              | 0.76  | "Cancel"            | (220, 340, 310, 370)  |  90x30  | macOS UI Elements
  #0003  icon                  | 0.65  | ""                  | (600,  50, 630,  80)  |  30x30  | GPA-GUI-Detector
  ...
```

### JSON estructurado
```json
{
  "metadata": { "total_elements": 116, "elapsed_ms": 18272 },
  "elements": [
    {
      "class": "button",
      "text": "Send",
      "confidence": 0.87,
      "bbox": [120, 340, 200, 370],
      "sources": ["Windows UI Locator", "macOS UI Elements"]
    }
  ],
  "class_summary": { "button": 28, "icon": 45 }
}
```

### Imagen anotada
Bounding boxes con color por clase, etiqueta `<clase> <confianza>` arriba y texto OCR debajo.

---

## Benchmark (Test.png — Outlook macOS)

| Modelo | Detecciones | Tiempo | Confianza media |
|---|---|---|---|
| GPA-GUI-Detector | **95** | 361ms | 0.40 |
| Windows UI Locator | 60 | 519ms | **0.73** |
| macOS UI Groups | 35 | 396ms | 0.65 |
| macOS UI Elements | 32 | **335ms** | 0.49 |
| Web Form Detection | 4 | **157ms** | 0.46 |

**Fusión v2**: 116 elementos, 9 clases distintas en 18s (1.7s detección + 16.5s OCR).

---

## Entorno

```bash
# Crear entorno virtual dentro de la carpeta del proyecto
python3 -m venv .env

# Activar
source .env/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Aceleración Metal (Apple Silicon) — automático
```

Los modelos se cachean en `~/.cache/yolo_ui/` y se descargan una sola vez.

---

## Estructura de archivos

```
ocr_opencv/
├── yolo_detect.py          # Detección con 1 modelo
├── ocr_mac.py              # OCR nativo macOS
├── ocr.swift               # Código fuente Swift OCR
├── ocr_bin                 # Binario compilado Swift
├── Test.png                # Imagen de prueba
│
├── models/
│   ├── base.py             # Runner compartido
│   ├── windows.py          # Windows UI Locator
│   ├── macos.py            # macOS UI Elements
│   ├── multiplatform.py    # GPA-GUI-Detector
│   ├── groups.py           # macOS UI Groups
│   └── web_form.py         # Web Form Detection
│
├── analyzer/
│   ├── ui_analyzer.py      # Pipeline multi-modelo + OCR
│   └── results/            # Outputs del análisis
│
├── benchmark/
│   ├── benchmark.py        # Comparativa de modelos
│   └── results/            # Outputs del benchmark
│
├── screens/                # Capturas anotadas (yolo_detect.py)
│
├── .env/                   # Entorno virtual Python
└── yolo_data/              # (reservado)
```
