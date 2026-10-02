# ☁️ CropSense: AWS Cloud Architecture & Integration Guide
### **FarmwiseAI Campus Challenge - Task 4 | Team AgriMinds**

This document outlines the AWS cloud infrastructure developed for the CropSense prototype, covering dataset versioning, model artifact storage, dataset integrity validation, and serverless event-driven inference.

---

## 🏛️ Architecture Overview

```text
[Field Photos / Data Pipeline]
           │
           ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ AWS S3 Bucket: `fai-tce-team46-cropsense-images`               │
 ├─────────────────────────────────────────────────────────────────┤
 │ 📁 raw/images/             -> Raw mobile-phone field photos     │
 │ 📁 labeled/                -> Verified multi-task split CSVs    │
 │ 📁 models/                 -> Production EfficientNet-B0 weights│
 │ 📁 incoming_queue/         -> Trigger bucket for new uploads    │
 │ 📁 predictions/            -> High-confidence automated results │
 │ 📁 human_review_queue/     -> Selective abstention escalations  │
 └─────────────────┬───────────────────────────────┬───────────────┘
                   │                               │
        (S3 Event Notification)          (CSV vs Object Check)
                   ▼                               ▼
 ┌───────────────────────────────────┐   ┌─────────────────────────┐
 │ AWS Lambda: Serverless Inference  │   │ AWS Lambda: S3 Verifier │
 │ `aws/lambda_inference_handler.py` │   │ `aws/lambda_function.py`│
 ├───────────────────────────────────┤   ├─────────────────────────┤
 │ - Multi-task Crop/Stage/Disease   │   │ - Cross-checks S3 keys  │
 │ - Confidence Floor Gate (>= 0.70) │   │   against CSV manifest  │
 │ - Saves JSON / Routes Escalations │   │ - Detects missing files │
 └───────────────────────────────────┘   └─────────────────────────┘
```

---

## 📁 AWS S3 Layout (`fai-tce-team46-cropsense-images`)

| Prefix / Path | Purpose | Content Type |
| :--- | :--- | :--- |
| `raw/images/` | Raw crop photos from field visits | Image binaries (`.jpg`, `.png`) |
| `labeled/` | Labeled datasets (`train_multitask.csv`, `val`, `test`) | CSV manifests |
| `models/` | Trained model weights (`cropsense_best.pth`, `class_mapping.json`) | PyTorch weights & JSON |
| `evaluation/` | Evaluation reports, confusion matrices, failure analysis | TXT, CSV, JSON |
| `incoming_queue/` | Ingestion folder for automated mobile field uploads | JPG / PNG uploads |
| `predictions/` | Confident inferences (`status: confident`) | Structured JSON results |
| `human_review_queue/`| Borderline images (`status: needs_review`, `< 0.70`) | Escalation JSON dossiers |

---

## 🛠️ AWS Components & Scripts

### 1. S3 Artifact Manager (`aws/s3_storage_manager.py`)
Synchronizes local models, manifests, and benchmark evaluation reports to AWS S3.
```bash
# Validate local artifacts in dry-run mode:
python aws/s3_storage_manager.py --dry-run

# Perform live sync (requires AWS credentials configured):
python aws/s3_storage_manager.py
```

### 2. S3 Dataset Integrity Lambda (`aws/lambda_function.py`)
Validates that every single image referenced in the dataset CSV manifest exists in S3 without missing references or silent corruptions.
- Reads `s3://fai-tce-team46-cropsense-images/labeled/clean_dataset_fixed.csv`
- Paginates through `s3://.../raw/images/`
- Returns audit status (`ALL FOUND` vs `MISSING IMAGES`).

### 3. Serverless Inference Handler (`aws/lambda_inference_handler.py`)
Provides an event-driven serverless architecture:
- Triggers automatically upon image upload to `incoming_queue/`.
- Executes confidence evaluation.
- Saves result directly to `predictions/` or routes to `human_review_queue/`.

---

## 🔐 Configuration & Credentials

Set standard AWS credentials via environment variables:
```bash
export AWS_ACCESS_KEY_ID="your-access-key-id"
export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
export AWS_DEFAULT_REGION="ap-south-1"
export CROPSENSE_S3_BUCKET="fai-tce-team46-cropsense-images"
```
Or configure locally using the AWS CLI:
```bash
aws configure
```
