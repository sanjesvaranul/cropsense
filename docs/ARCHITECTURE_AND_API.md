# 🏛️ CropSense: System Architecture & API Reference
### **Task 4: Mobile Crop, Stage and Disease Model | Team AgriMinds (TCE Madurai)**

---

## 1. System Architecture Overview

CropSense is built on a **Teacher-Student paradigm**:
- **Teacher Models:** High-parameter Vision-Language Models (Gemini 1.5, GPT-4V) provide consensus pseudo-labeling on uncurated field images and act as high-tier safety fallback verifiers.
- **Student Model:** A compact, multi-task **EfficientNet-B0** vision model with three shared-backbone classification heads (`crop_head`, `stage_head`, `condition_head`).
- **Selective Abstention Gate:** Every prediction passes through a 70% confidence floor. Confident predictions receive immediate treatment advisories; uncertain cases are escalated to the human review queue and teacher model.
- **Versioned Continuous Learning:** Escalated images with confirmed ground truth feed back into the retraining dataset to produce challenger models evaluated against the frozen held-out test set.

```text
                           [ Mobile Field Photo ]
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │  CropSense Student Model    │
                      │      (EfficientNet-B0)      │
                      └──────────────┬──────────────┘
                                     │
            ┌────────────────────────┼────────────────────────┐
            ▼                        ▼                        ▼
     [Crop Head (12)]         [Stage Head (5)]        [Disease Head (5)]
            │                        │                        │
            └────────────────────────┼────────────────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │  Confidence Floor (>= 0.70) │
                      └──────────────┬──────────────┘
                                     │
                   ┌─────────────────┴─────────────────┐
                   ▼                                   ▼
        [Confident: >= 0.70]                 [Uncertain: < 0.70]
        - Instant Diagnosis                  - Selective Abstention Flagged
        - Agronomic Advisory                 - Escalated to Review Queue / VLM
        - Edge / Offline execution           - Appended to Feedback Dataset
```

---

## 2. API Specification

### `GET /`
**Health Check & Capability Descriptor**
```json
{
  "status": "CropSense API is running",
  "model": "EfficientNet-B0 (Multi-Task: Crop, Stage, Disease)",
  "device": "cpu",
  "classes": {
    "num_crops": 12,
    "num_stages": 5,
    "num_conditions": 5
  }
}
```

### `POST /predict`
**Multi-Task Image Prediction Endpoint**
- **Content-Type:** `multipart/form-data`
- **Body Parameter:** `file` (Binary image: JPG, PNG, WEBP)

#### Example Response (High Confidence):
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

#### Example Response (Selective Abstention / Review Required):
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

## 3. Supported Categories

### Crop Categories (12)
`Banana`, `Cassava`, `Coconut`, `Diploid Cotton (Paruththi)`, `Fodder Sorghum`, `Groundnut`, `Maize`, `Mango`, `Marigold`, `Rice (Paddy)`, `Sorghum`, `Sugarcane`

### Growth Stages (5)
`sown`, `vegetation`, `flowering`, `full_growth`, `harvesting`

### Foliar Conditions (5)
`Healthy`, `Bacterial Leaf Blight`, `Brown Spot`, `Leaf Blast`, `Leaf Smut`
