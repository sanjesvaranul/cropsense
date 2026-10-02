# 🌾 CropSense: Mobile Crop, Stage & Disease Intelligence
### **Hackathon Task 4: Mobile Crop, Stage & Disease Intelligence**

CropSense is an edge-deployable, multi-task deep learning system designed for resource-constrained mobile agricultural intelligence. Built upon a lightweight **EfficientNet-B0** backbone with a **three-head classification architecture**, CropSense simultaneously predicts:

1. **Crop Identification** (12 classes)
2. **Growth Stage Recognition** (5 stages)
3. **Foliar Disease Diagnosis** (Healthy + 4 critical rice foliar diseases)

With an integrated **selective abstention engine (70% confidence floor)**, CropSense delivers actionable agronomic advisories for confident predictions and flags borderline images for expert agricultural review, preventing dangerous misdiagnoses in the field.

---

## 🚀 Key Results & Performance (Held-Out Test Set: 351 Images)

Evaluated on the held-out test split (`data/labeled/test_multitask.csv`):

| Intelligence Task | Metric / Accuracy | Macro F1 | Weighted F1 | Primary Classes Supported |
| :--- | :---: | :---: | :---: | :--- |
| **Crop Identification** | **93.16%** | 0.43 | 0.93 | Rice (Paddy), Coconut, Sugarcane, Maize, Banana, Groundnut, etc. |
| **Growth Stage Recognition** | **80.91%** | 0.56 | 0.81 | Sown, Vegetation, Flowering, Full Growth, Harvesting |
| **Foliar Disease Diagnosis** | **99.15%** | **0.93** | **0.99** | Healthy (1.00), Bacterial Leaf Blight (0.96), Brown Spot (0.89), Leaf Blast (1.00), Leaf Smut (0.80) |

- **High-Confidence Automated Decisions:** `70.66%` of all field photos pass the strict 70% threshold across all three heads simultaneously.
- **Selective Abstention & Expert Review:** `29.34%` of images trigger human-in-the-loop routing with candidate flags to ensure zero-risk farming recommendations.

---

## 🏛️ System Architecture

```text
                               ┌─────────────────────────────────────────┐
                               │             Input Crop Photo            │
                               │               (224 x 224)               │
                               └────────────────────┬────────────────────┘
                                                    │
                               ┌────────────────────▼────────────────────┐
                               │         EfficientNet-B0 Backbone        │
                               │      (Edge-optimized, 1280 features)    │
                               └────────────────────┬────────────────────┘
                                                    │
                     ┌──────────────────────────────┼──────────────────────────────┐
                     │                              │                              │
       ┌─────────────▼─────────────┐  ┌─────────────▼─────────────┐  ┌─────────────▼─────────────┐
       │      Crop Head (Linear)   │  │     Stage Head (Linear)   │  │   Disease Head (Linear)   │
       │         (12 classes)      │  │         (5 stages)        │  │       (5 conditions)      │
       └─────────────┬─────────────┘  └─────────────┬─────────────┘  └─────────────┬─────────────┘
                     │                              │                              │
                     └──────────────────────────────┼──────────────────────────────┘
                                                    │
                               ┌────────────────────▼────────────────────┐
                               │  Confidence Estimator & Abstention Gate │
                               │            (Floor >= 0.70)              │
                               └────────────────────┬────────────────────┘
                                                    │
                           ┌────────────────────────┴────────────────────────┐
                           ▼                                                 ▼
             [Status: confident]                               [Status: needs_review]
             - Automated Diagnostics                           - Abstention Flagged
             - Confident Crop & Stage                          - Suspected Candidates
             - Agronomic Advisory & Treatment                  - Routed for KVK / Expert Inspection
```

---

## 📦 Project Layout

```text
cropsense/
├── data/
│   ├── external/rice_diseases/     # Downloaded & deduplicated rice disease dataset (342 images)
│   │   ├── Bacterial Leaf Blight/  # 117 images
│   │   ├── Brown Spot/             # 136 images
│   │   ├── Leaf Blast/             # 49 images
│   │   └── Leaf Smut/              # 40 images
│   ├── labeled/                    # Clean stratified multi-task datasets
│   │   ├── train_multitask.csv     # 2,807 labeled samples
│   │   ├── val_multitask.csv       # 349 validation samples
│   │   └── test_multitask.csv      # 351 held-out test samples
│   └── raw/images/                 # Original multi-crop images (~3,165 images)
├── demo/
│   └── app.py                      # Interactive Streamlit UI with disease advisory & test presets
├── inference/
│   ├── confidence.py               # Selective abstention filter (0.70 floor)
│   └── router.py                   # FastAPI REST API serving 3-head multi-task model
├── models/
│   ├── class_mapping.json          # Index-to-label maps (crops, stages, conditions)
│   ├── cropsense_best.pth          # Best multi-task trained model weights
│   └── cropsense_crop_stage_only.pth # Original 2-head baseline backup
├── pipeline/
│   ├── integrate_external_disease.py # Automated external dataset downloader & deduplicator
│   └── prepare_multitask_dataset.py  # Stratified merger for 3-task labeling
└── training/
    ├── evaluate_multitask.py       # Full held-out test evaluation script
    ├── train_multitask.py          # Multi-task training script with class-weighted loss
    └── evaluation/                 # Metrics & confusion matrices (.csv & .txt)
        ├── multitask_evaluation_summary.txt
        ├── crop_confusion_matrix.csv
        ├── stage_confusion_matrix.csv
        └── condition_confusion_matrix.csv
```

---

## ⚡ Quick Start: Running the Prototype

### Prerequisites
Activate your Python environment:
```bash
# Windows
.\cropsense-env\Scripts\activate
# Linux/macOS
source cropsense-env/bin/activate
```

### 1. Launch the FastAPI Backend
```bash
python -m uvicorn inference.router:app --host 127.0.0.1 --port 8000
```
- Interactive API Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

### 2. Launch the Streamlit Interactive Interface
In a second terminal:
```bash
python -m streamlit run demo/app.py
```
- Open your browser at [http://localhost:8501](http://localhost:8501)
- You can either **upload a photo** or **select preloaded test samples from the sidebar** (Healthy, Bacterial Leaf Blight, Brown Spot, Leaf Blast, Leaf Smut).

### 3. Run Model Evaluation
To reproduce the full held-out test set benchmark:
```bash
python training/evaluate_multitask.py
```

---

## 📡 API Specification

### `POST /predict`
Send a multipart form request containing `file`:

**Confident Prediction Response Example:**
```json
{
  "source": "student",
  "crop": "Rice (Paddy)",
  "stage": "vegetation",
  "condition": "Bacterial Leaf Blight",
  "crop_confidence": 0.9973,
  "stage_confidence": 0.9808,
  "disease_confidence": 0.9045,
  "status": "confident"
}
```

**Selective Abstention Response Example (`< 70%` Confidence):**
```json
{
  "source": "student",
  "crop": "Rice (Paddy)",
  "stage": "vegetation",
  "condition": "Leaf Blast",
  "crop_confidence": 0.9944,
  "stage_confidence": 0.9722,
  "disease_confidence": 0.5872,
  "status": "needs_review",
  "abstention_reason": "One or more prediction heads fell below the 70% confidence threshold"
}
```

---

## 🌿 Agronomic Knowledge & Disease Action Engine

When a disease is detected, CropSense provides instant agronomic treatment advice:

| Condition | Pathogen | Actionable Treatment Recommendation |
| :--- | :--- | :--- |
| **Healthy** | None | Maintain standard N-P-K nutrient management, field scouting, and optimal irrigation. |
| **Bacterial Leaf Blight** | *Xanthomonas oryzae* | Reduce excessive nitrogen fertilizer. Ensure prompt field drainage. Apply copper hydroxide or approved bactericides in early infection. |
| **Brown Spot** | *Bipolaris oryzae* | Soil nutrient deficiency indicator (Zinc/Potassium). Apply balanced NPK + Micronutrients. Treat with Mancozeb or Tricyclazole if spreading. |
| **Leaf Blast** | *Magnaporthe oryzae* | High-urgency intervention. Apply systemic fungicide (Tricyclazole 75 WP or Isoprothiolane). Avoid high nitrogen top-dressing and cool standing water. |
| **Leaf Smut** | *Entyloma oryzae* | Avoid canopy stagnation and high humidity. If extensive during late grain fill, apply propiconazole or copper oxychloride foliar spray. |
