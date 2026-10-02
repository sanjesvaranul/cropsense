import io
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms
from fastapi import FastAPI, HTTPException, UploadFile

from .confidence import predict_with_abstention


# ---------------- CONFIG ----------------

MODEL_PATH = Path("models/cropsense_best.pth")
LABELS_PATH = Path("models/class_mapping.json")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

app = FastAPI(title="CropSense API")


# ---------------- MODEL ----------------

class CropSenseModel(nn.Module):
    def __init__(self, num_crops, num_stages, num_conditions=5):
        super().__init__()

        backbone = models.efficientnet_b0(weights=None)

        self.features = backbone.features
        self.pool = backbone.avgpool

        feature_size = backbone.classifier[1].in_features

        self.crop_head = nn.Linear(feature_size, num_crops)
        self.stage_head = nn.Linear(feature_size, num_stages)
        self.condition_head = nn.Linear(feature_size, num_conditions)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        crop_logits = self.crop_head(x)
        stage_logits = self.stage_head(x)
        condition_logits = self.condition_head(x)

        return crop_logits, stage_logits, condition_logits


# ---------------- LOAD MODEL ----------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not LABELS_PATH.exists():
    raise FileNotFoundError(f"Label mapping not found: {LABELS_PATH}")

with open(LABELS_PATH, "r") as f:
    mappings = json.load(f)

crop_map = mappings["crop_map"]
stage_map = mappings["stage_map"]
condition_map = mappings.get("condition_map", {
    "Healthy": 0,
    "Bacterial Leaf Blight": 1,
    "Brown Spot": 2,
    "Leaf Blast": 3,
    "Leaf Smut": 4
})

# Convert index -> class name
crop_names = {
    int(index): name for name, index in crop_map.items()
}

stage_names = {
    int(index): name for name, index in stage_map.items()
}

condition_names = {
    int(index): name for name, index in condition_map.items()
}

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

num_crops = checkpoint.get("num_crops", len(crop_names))
num_stages = checkpoint.get("num_stages", len(stage_names))
num_conditions = checkpoint.get("num_conditions", len(condition_names))

model = CropSenseModel(
    num_crops,
    num_stages,
    num_conditions
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

IMAGE_SIZE = checkpoint.get("image_size", 224)
MEAN = checkpoint.get("mean", [0.485, 0.456, 0.406])
STD = checkpoint.get("std", [0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ---------------- PREDICTION ----------------

def run_student_model(image_bytes):
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            image = img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image."
        )

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        crop_logits, stage_logits, condition_logits = model(tensor)

    crop_result = predict_with_abstention(crop_logits)
    stage_result = predict_with_abstention(stage_logits)
    condition_result = predict_with_abstention(condition_logits)

    return crop_result, stage_result, condition_result


def log_feedback_case(filename, crop_result, stage_result, condition_result=None):
    # Placeholder for future feedback-dataset integration
    print(
        "Needs review:",
        filename,
        "Crop:", crop_result,
        "Stage:", stage_result,
        "Condition:", condition_result
    )


# ---------------- API ROUTES ----------------

@app.get("/")
def health_check():
    return {
        "status": "CropSense API is running",
        "model": "EfficientNet-B0 (Multi-Task: Crop, Stage, Disease)",
        "device": str(DEVICE),
        "classes": {
            "num_crops": len(crop_names),
            "num_stages": len(stage_names),
            "num_conditions": len(condition_names)
        }
    }


@app.post("/predict")
async def predict(file: UploadFile):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file uploaded."
        )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, PNG, or WEBP image."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    crop_result, stage_result, condition_result = run_student_model(image_bytes)

    crop_id = crop_result["prediction"]
    stage_id = stage_result["prediction"]
    condition_id = condition_result["prediction"]

    crop_confidence = crop_result["confidence"]
    stage_confidence = stage_result["confidence"]
    disease_confidence = condition_result["confidence"]

    crop_name = crop_names.get(crop_id) if crop_id is not None else None
    stage_name = stage_names.get(stage_id) if stage_id is not None else None
    condition_name = condition_names.get(condition_id) if condition_id is not None else None

    # Candidate names in case of abstention
    cand_crop = crop_names.get(crop_result.get("candidate"))
    cand_stage = stage_names.get(stage_result.get("candidate"))
    cand_cond = condition_names.get(condition_result.get("candidate"))

    # Require all outputs to pass the confidence threshold
    is_confident = (
        crop_result["status"] == "confident"
        and stage_result["status"] == "confident"
        and condition_result["status"] == "confident"
    )

    if not is_confident:
        log_feedback_case(file.filename, crop_result, stage_result, condition_result)

        return {
            "source": "student",
            "crop": crop_name if crop_name is not None else cand_crop,
            "stage": stage_name if stage_name is not None else cand_stage,
            "condition": condition_name if condition_name is not None else cand_cond,
            "crop_confidence": crop_confidence,
            "stage_confidence": stage_confidence,
            "disease_confidence": disease_confidence,
            "status": "needs_review",
            "abstention_reason": "One or more prediction heads fell below the 70% confidence threshold"
        }

    return {
        "source": "student",
        "crop": crop_name,
        "stage": stage_name,
        "condition": condition_name,
        "crop_confidence": crop_confidence,
        "stage_confidence": stage_confidence,
        "disease_confidence": disease_confidence,
        "status": "confident"
    }
