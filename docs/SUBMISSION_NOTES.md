# 🏆 FarmwiseAI Campus Challenge — Task 4 Final Submission Document
## **CropSense: AgriLens — Mobile Crop, Stage & Disease Intelligence**
### **Team AgriMinds | Department of Electronics and Communication Engineering**
### **Thiagarajar College of Engineering (TCE), Madurai**

* **Team Lead:** Sanjesvaran Umadevan
* **Team Members:** Gohulavaasan Palaniswamy, Balamurugan Paramasivam, Senthil Murugan Ramar
* **Institution:** Thiagarajar College of Engineering (TCE), Madurai, Tamil Nadu, India

---

## 📌 Executive Summary

Smallholder farmers face severe crop yield losses due to misidentified crop health conditions and delayed agronomic interventions. Existing commercial Vision-Language Models (VLMs) like GPT-4V and Gemini 1.5 Pro offer high general reasoning but suffer from:
1. **Excessive Cost:** \$2.50 to \$5.00 per 1,000 queries—prohibitive for rural farmers.
2. **High Latency:** 1.2 to 2.5 seconds per API call.
3. **Connectivity Dependency:** Cloud-only models fail in remote field parcels without 4G/5G connectivity.
4. **Hallucination Risk:** Cloud VLMs confidently predict incorrect diagnoses on out-of-distribution symptoms.

**CropSense** solves this challenge through a student-teacher distillation paradigm:
- **Shared Edge Backbone:** EfficientNet-B0 (15.68 MB footprint, 4.39 MB INT8) capable of running entirely offline on modest mobile hardware or cheap micro-servers.
- **Three-Head Multi-Task Output:** Simultaneously predicts **Crop Type (12 classes)**, **Growth Stage (5 stages)**, and **Foliar Condition (Healthy + 4 fungal/bacterial diseases)** from a single RGB field photograph.
- **Selective Abstention Gate (70% Confidence Floor):** Automatically validates predictions against a strict confidence threshold. Ambiguous cases are routed to human agronomists / teacher review queues, preventing costly spray errors.
- **Instant Agronomic Treatment Protocols:** Provides immediate actionable treatment advice (chemical and cultural management) for confirmed foliar diseases.

---

## 📋 Task 4 Requirements Compliance Matrix

| Requirement | Proposed Solution | Implementation in CropSense | Status |
| :--- | :--- | :--- | :---: |
| **Multi-Task Vision Model** | Predict Crop, Stage, and Foliar Condition simultaneously | Shared EfficientNet-B0 backbone with 3 distinct linear heads | ✅ **Complete** |
| **Model Optimization** | Edge-friendly, mobile deployable architecture | EfficientNet-B0 (15.68 MB unquantized, 4.39 MB INT8) | ✅ **Complete** |
| **Handling Class Imbalance** | Address rare crops and stage skewness | Class-weighted CrossEntropyLoss + WeightedRandomSampler + targeted minority augmentation | ✅ **Complete** |
| **Safety & Selective Abstention** | Do not make dangerous spray recommendations when uncertain | 70% Confidence Floor Gate routing ambiguous cases to review queue | ✅ **Complete** |
| **Teacher vs. Student Benchmark** | Compare distilled student against commercial cloud VLM | 99.94% cost reduction, 19x lower latency, superior specialized disease accuracy | ✅ **Complete** |
| **Interactive User Interface** | Web/mobile app demonstration with batch and review features | 5-Tab Streamlit application (`demo/app.py`) + FastAPI REST server (`inference/router.py`) | ✅ **Complete** |
| **AWS Cloud Integration** | S3 data lake & serverless verification | S3 sync manager (`s3_storage_manager.py`) + Lambda functions (`lambda_function.py`, `lambda_inference_handler.py`) | ✅ **Complete** |

---

## 📊 Held-Out Test Set Benchmark Results (351 Images)

Evaluated strictly on the held-out test split (`data/labeled/test_multitask.csv`):

| Classification Task | Classes | Test Accuracy | Macro F1 | Weighted F1 | Target Classes Included |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Crop Identification** | 12 | **94.02%** | 0.44 | **0.94** | Rice (Paddy), Coconut, Sugarcane, Maize, Banana, Groundnut, etc. |
| **Growth Stage Recognition** | 5 | **82.34%** | 0.59 | **0.83** | Sown, Vegetation, Flowering, Full Growth, Harvesting |
| **Foliar Disease Diagnosis** | 5 | **99.43%** | **0.93** | **0.99** | Healthy, Bacterial Leaf Blight, Brown Spot, Leaf Blast, Leaf Smut |

### Operational Gating & Latency:
- **High-Confidence Automated Decisions:** `291 / 351` (**82.91%**) passed $\ge 70\%$ confidence on all three heads.
- **Selective Abstention Rate:** `60 / 351` (**17.09%**) flagged for review.
- **Single-Image Inference Latency:** **46.7 ms** on standard CPU (~15.1 images/sec throughput).

---

## ⚖️ Class Imbalance Mitigation Strategy

To address the extreme frequency disparity between majority crops (Rice: 2,000+ samples) and rare minority crops (Groundnut, Cassava, Banana: < 20 samples):
1. **Square-Root Smoothed Inverse Frequency Loss Weights:**
   $$\text{Weight}_c = \text{clip}\left(\sqrt{\frac{N}{C \cdot N_c}}, 0.3, 5.0\right)$$
   Penalizes errors on underrepresented classes without causing gradient destabilization.
2. **Multi-Task WeightedRandomSampler:**
   Calculates a composite geometric sampling weight across all three task heads per sample:
   $$W_i = \left(w_{\text{crop}, i} \cdot w_{\text{stage}, i} \cdot w_{\text{condition}, i}\right)^{1/3}$$
   Ensures minority crops and rare stages appear regularly in every mini-batch.
3. **Targeted Minority Data Augmentation:**
   Applies dynamic random affine transforms, vertical flips, color jitter, and random perspective crops exclusively to minority class instances during training.

---

## 📈 Teacher VLM vs. CropSense Student: Head-to-Head Benchmark

| Benchmark Dimension | CropSense Student (Edge) | Teacher VLM (Gemini / GPT-4V Cloud) | Practical Agronomic Advantage |
| :--- | :--- | :--- | :--- |
| **Architecture** | EfficientNet-B0 + 3 Heads | Gemini 1.5 Pro / GPT-4o Vision | Edge-executable, no cloud roundtrip |
| **Model Size** | **15.68 MB** (4.39 MB INT8) | > 50 - 100 GB | Fits on budget mobile phones |
| **Inference Cost / 1,000** | **\$0.0003** (Local CPU) | **\$2.50 - \$5.00** (API calls) | **> 99.9% Cost Reduction** |
| **Single-Image Latency** | **~47 ms** (CPU) | **~1,250 ms** (Network + decode) | **19x Faster Response** |
| **Batch Throughput** | **~15.1 images / sec** | 5 - 15 RPM (Rate limited) | Unlimited offline processing |
| **Disease Accuracy** | **99.43% (Weighted F1: 0.99)** | 96.40% (Teacher consensus) | Superior specialized foliar precision |
| **Offline Operation** | **100% Offline Capable** | Requires active Internet | Functions in remote rural parcels |
| **Safety Assurance** | **70% Confidence Floor Gate** | Hallucination prone | Zero-risk agronomist fallback |

---

## 🖥️ Interactive Demo Suite Overview (`demo/app.py`)

The prototype provides a 5-tab dashboard built with Streamlit and backed by FastAPI:

1. **Tab 1: Single-Photo Diagnosis**
   - Quick-test preset library with pre-loaded field and disease samples.
   - User photo upload (JPG, PNG, WEBP).
   - Real-time multi-task prediction with confidence scoring and badge indicators.
   - Actionable Agronomic Advisory & Treatment Recommendation protocols.
2. **Tab 2: Batch Inference & Export**
   - Multi-image drag-and-drop batch processing.
   - Live progress indicator with table view.
   - Downloadable CSV report with filename, crop, stage, condition, confidences, and route status.
3. **Tab 3: Human-in-the-Loop Review Queue**
   - Displays all flagged test images where confidence fell below 70%.
   - In-depth failure analysis and agricultural root cause explanations (stage continuum ambiguity).
4. **Tab 4: Teacher vs. Student Benchmark**
   - Side-by-side comparative table against commercial VLMs.
   - Highlights cost reduction (99.94%), latency speedup (19x), and offline capability.
5. **Tab 5: Evaluation & Confusion Matrices**
   - Interactive dropdown selector to view confusion matrices for Diseases, Stages, and Crops.
   - Matplotlib heatmaps with annotated numerical counts and normalized color scaling.
   - Full scikit-learn classification reports with downloadable TXT summary.

---

## ⚡ Quick-Start Instructions

### Step 1: Start FastAPI Backend
```bash
python -m uvicorn inference.router:app --host 127.0.0.1 --port 8000
```
- API Docs: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/`

### Step 2: Start Streamlit Application
```bash
python -m streamlit run demo/app.py
```
- Access the web interface at `http://localhost:8501`

### Step 3: Run Evaluation Scripts
```bash
# Held-Out Test Evaluation
python training/evaluate_multitask.py

# Failure Case Analysis
python training/analyze_failures.py

# Teacher vs Student Benchmark
python training/benchmark_comparison.py
```

---

## 📁 Repository & Submission Structure

```text
cropsense/
├── aws/                              # AWS serverless integration & S3 management
│   ├── lambda_function.py            # Lambda dataset validator
│   ├── lambda_inference_handler.py   # Lambda event-driven inference handler
│   ├── s3_storage_manager.py         # S3 dataset synchronization tool
│   └── README_AWS.md                 # Complete AWS architecture documentation
├── data/
│   ├── external/rice_diseases/       # Curated external disease dataset (342 images)
│   ├── labeled/                      # Multi-task train, val, and test splits (CSV)
│   └── raw/images/                   # Multi-crop raw field photos
├── demo/
│   └── app.py                        # 5-Tab Streamlit Interactive Application
├── docs/
│   ├── ARCHITECTURE_AND_API.md       # Complete system architecture and REST API schema
│   ├── EVALUATION_REPORT.md          # Quantitative evaluation and failure breakdown
│   └── SUBMISSION_NOTES.md           # This document
├── inference/
│   ├── confidence.py                 # Selective abstention gate (0.70 confidence floor)
│   └── router.py                     # FastAPI REST API multi-task inference server
├── models/
│   ├── class_mapping.json            # Class dictionary mappings
│   └── cropsense_best.pth            # Trained 3-head multi-task model weights
├── pipeline/                         # Dataset preparation, filtering, and validation
├── training/
│   ├── analyze_failures.py           # Failure case analysis generator
│   ├── benchmark_comparison.py       # Teacher vs student benchmark generator
│   ├── evaluate_multitask.py         # Held-out test evaluation script
│   ├── train_multitask.py            # Class-weighted multi-task training script
│   └── evaluation/                   # Saved evaluation metrics & confusion matrices
├── requirements.txt                  # Python dependencies
└── README.md                         # Project overview and instructions
```

---

*Submitted by Team AgriMinds, Thiagarajar College of Engineering (TCE), Madurai.*
