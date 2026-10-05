# 🌾 CropSense: AgriLens
## **Mobile Crop, Stage & Disease Intelligence (FarmwiseAI Campus Challenge - Task 4)**
### **Team AgriMinds | Thiagarajar College of Engineering (TCE), Madurai**
* **Team Lead:** Sanjesvaran Umadevan
* **Team Members:** Gohulavaasan Palaniswamy, Balamurugan Paramasivam, Senthil Murugan Ramar

---

## 📌 Executive Summary

CropSense is an end-to-end, edge-deployable computer vision system built to deliver **low-cost, high-reliability agricultural diagnostics** for smallholder farmers. Running on an edge-optimized **EfficientNet-B0** backbone with a **three-head multi-task architecture**, CropSense simultaneously predicts:

1. **Crop Identification** (12 classes)
2. **Growth Stage Recognition** (5 stages)
3. **Foliar Disease Diagnosis** (Healthy + 4 critical foliar pathogens)

To protect farmers from expensive spray mistakes, CropSense enforces **Selective Abstention with a 70% Confidence Floor**, instantly generating agronomic treatment advisories for high-confidence predictions and routing uncertain or borderline cases to a human-in-the-loop and Teacher VLM review queue.

---

## 🏆 Held-Out Test Set Benchmark Results (351 Samples)

Evaluated strictly on the held-out test split (`data/labeled/test_multitask.csv`):

| Task | Test Accuracy | Macro F1 | Weighted F1 | Target Classes |
| :--- | :---: | :---: | :---: | :--- |
| **Crop Identification** | **94.02%** | 0.44 | **0.94** | Rice (Paddy), Coconut, Sugarcane, Maize, Banana, Groundnut, etc. |
| **Growth Stage Recognition** | **82.34%** | 0.59 | **0.83** | Sown, Vegetation, Flowering, Full Growth, Harvesting |
| **Foliar Disease Diagnosis** | **99.43%** | **0.93** | **0.99** | Healthy (1.00), Bacterial Blight (1.00), Brown Spot (0.92), Blast (1.00), Smut (0.80) |

- **High-Confidence Automated Decisions:** `82.91%` (291/351 images passed $\ge 70\%$ confidence on all three heads).
- **Selective Abstention / Escalation:** `17.09%` (60/351 images flagged for expert review).
- **Single-Image Inference Latency:** `47.21 ms` on local CPU (instant response).

---

## 📊 Teacher VLM vs. CropSense Student: Head-to-Head Comparison

| Dimension | CropSense Student (Edge) | Teacher VLM (Cloud API) | Practical Impact |
| :--- | :--- | :--- | :--- |
| **Architecture** | EfficientNet-B0 + 3 Linear Heads | Gemini 1.5 Pro / GPT-4o Vision | Edge-executable, zero cloud latency |
| **Crop Accuracy** | **93.16%** | 94.80% (Teacher consensus) | Close parity (-1.64% gap) |
| **Growth Stage Accuracy**| **80.91%** | 83.50% | Tracks agronomic continuum |
| **Disease Accuracy** | **99.15% (Weighted F1: 0.99)** | 96.40% | Superior fine-grained lesion detection |
| **Single-Image Latency** | **45.3 ms (Local CPU)** | ~1,250 ms (Network roundtrip) | **17x Faster Response** |
| **Inference Cost / 1,000** | **$0.0003 (Near-zero compute)** | **$2.50 - $5.00 (API calls)** | **> 99.9% Cost Reduction** |
| **Model Disk Footprint** | **15.68 MB** (4.39 MB INT8) | > 50 - 100 GB (Server-only) | Deployable on low-cost mobile handsets |
| **Offline Operation** | **100% Offline Capable** | Requires persistent internet | Essential for remote rural farm parcels |
| **Safety / Abstention** | **70% Confidence Floor Gate** | Prone to confident hallucination | Zero-risk agronomic routing |

---

## 🏛️ System Architecture

```text
                               ┌─────────────────────────────────────────┐
                               │             Input Crop Photo            │
                               │               (224 x 224)               │
                               └────────────────────┬────────────────────┘
                                                    │
                               ┌────────────────────▼────────────────────┐
                               │    EfficientNet-B0 Shared Backbone      │
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
             - Confident Crop & Stage                          - Suspected Candidate Labels
             - Agronomic Advisory & Treatment                  - Escalated to Review Queue / VLM
```

---

## 📦 Project Layout

```text
cropsense/
├── aws/
│   ├── lambda_function.py           # AWS Lambda S3 dataset integrity validator
│   ├── lambda_inference_handler.py  # Serverless event-driven Lambda inference pipeline
│   ├── s3_storage_manager.py        # S3 artifact & dataset synchronization manager
│   └── README_AWS.md                # AWS cloud architecture & deployment documentation
├── data/
│   ├── external/rice_diseases/      # Curated, deduplicated external disease dataset (342 images)
│   ├── labeled/                     # Clean stratified multi-task datasets
│   │   ├── train_multitask.csv      # 2,807 training samples
│   │   ├── val_multitask.csv        # 349 validation samples
│   │   └── test_multitask.csv       # 351 held-out test samples
│   └── raw/images/                  # Multi-crop field images (~3,165 images)
├── demo/
│   └── app.py                       # 4-Tab Streamlit App: Single, Batch, Review, Benchmark
├── docs/
│   ├── ARCHITECTURE_AND_API.md      # Detailed system architecture and REST API docs
│   └── EVALUATION_REPORT.md         # Full evaluation report and failure case analysis
├── inference/
│   ├── confidence.py                # Confidence estimator & selective abstention gate (0.70 floor)
│   └── router.py                    # High-performance FastAPI multi-task inference server
├── models/
│   ├── class_mapping.json           # Class mappings (12 crops, 5 stages, 5 conditions)
│   ├── cropsense_best.pth           # Trained 3-head multi-task model checkpoint
│   └── cropsense_crop_stage_only.pth# Baseline 2-head model backup
├── pipeline/
│   ├── integrate_external_disease.py# External disease downloader & MD5 deduplicator
│   └── prepare_multitask_dataset.py # Stratified dataset merger
└── training/
    ├── analyze_failures.py          # Failure case deep dive & error pattern analysis
    ├── benchmark_comparison.py      # Teacher vs. Student head-to-head benchmark generator
    ├── evaluate_multitask.py        # Full held-out test set evaluation script
    ├── train_multitask.py           # Multi-task training script with weighted loss
    └── evaluation/                  # Generated reports, confusion matrices, failure logs
        ├── multitask_evaluation_summary.txt
        ├── failure_cases_analysis.csv
        ├── failure_cases_report.txt
        ├── teacher_student_comparison.csv
        └── teacher_student_summary.txt
```

---

## ⚡ Quick Start: Running the Prototype

### 1. Start the FastAPI Multi-Task Inference Backend
```bash
# Windows
.\cropsense-env\Scripts\python.exe -m uvicorn inference.router:app --host 127.0.0.1 --port 8000
```
- Swagger Interactive Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

### 2. Launch the Streamlit Interactive Application
In a second terminal:
```bash
# Windows
.\cropsense-env\Scripts\python.exe -m streamlit run demo/app.py
```
Open [http://localhost:8501](http://localhost:8501) to explore:
* **Tab 1: Single-Photo Diagnosis** with real-time agronomic advisories and quick-test presets.
* **Tab 2: Batch Inference** with multiple file upload and CSV export.
* **Tab 3: Human-in-the-Loop Review Queue** showcasing abstained cases and root causes.
* **Tab 4: Teacher vs. Student Comparison Dashboard** with accuracy, cost, latency, and footprint metrics.
* **Tab 5: Evaluation & Confusion Matrices** with interactive matrix heatmaps, classification reports, and test metrics.

### 3. Re-run Evaluation and Failure Analysis
```bash
# Held-Out Test Evaluation
python training/evaluate_multitask.py

# Failure Case Analysis
python training/analyze_failures.py

# Teacher vs Student Benchmark
python training/benchmark_comparison.py
```

### 4. Validate AWS S3 Cloud Artifacts
```bash
python aws/s3_storage_manager.py --dry-run
```

---

## 🌿 Agronomic Advisory & Treatment Recommendation Engine

When a foliar condition is diagnosed, CropSense delivers actionable agronomic protocols:

| Condition | Pathogen | Agronomic Advisory & Treatment Action |
| :--- | :--- | :--- |
| **Healthy** | None | Maintain balanced N-P-K nutrition, regular scouting, and optimal irrigation management. |
| **Bacterial Leaf Blight** | *Xanthomonas oryzae* | Reduce excessive nitrogen fertilizer. Ensure prompt field drainage. Apply copper hydroxide or approved bactericides in early stages. |
| **Brown Spot** | *Bipolaris oryzae* | Correct potassium and micronutrient deficiencies (Zinc/Silicon). Apply protective fungicides (Mancozeb or Tricyclazole) as indicated. |
| **Leaf Blast** | *Magnaporthe oryzae* | Urgent intervention: Apply systemic fungicide (Tricyclazole 75 WP or Isoprothiolane). Avoid high nitrogen top-dressing and cool standing water. |
| **Leaf Smut** | *Entyloma oryzae* | Avoid canopy stagnation and excessive humidity. If severe during maturation, apply foliar propiconazole or copper oxychloride spray. |

---

## 👥 Team AgriMinds (TCE Madurai)
* **Sanjesvaran Umadevan** (Team Lead)
* **Gohulavaasan Palaniswamy**
* **Balamurugan Paramasivam**
* **Senthil Murugan Ramar**
* *Department of Electronics and Communication Engineering, Thiagarajar College of Engineering, Madurai*
