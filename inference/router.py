from types import SimpleNamespace

import torch
from fastapi import FastAPI, HTTPException, UploadFile

from .confidence import predict_with_abstention


app = FastAPI(title="CropSense API")


def run_student_model(file):
    """
    Temporary mock student model.

    This will be replaced with the real trained model
    after the dataset and training stage are completed.
    """

    # Mock logits for 3 example classes
    logits = torch.tensor([[3.0, 1.0, 0.5]])

    return SimpleNamespace(logits=logits)


def log_feedback_case(file, verdict, teacher_result=None):
    """
    Temporary feedback logger.

    This will later save low-confidence cases to the
    versioned feedback dataset.
    """
    print("Feedback case logged:", verdict)


@app.get("/")
def health_check():
    return {"status": "CropSense API is running"}


@app.post("/predict")
async def predict(file: UploadFile):

    # Check that an image was uploaded
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file uploaded"
        )

    # Check file type
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, PNG, or WEBP image"
        )

    result = run_student_model(file)

    verdict = predict_with_abstention(result.logits)

    # Convert mock class index into a readable result
    class_info = {
        0: {
            "crop": "Tomato",
            "stage": "Vegetative",
            "condition": "Healthy"
        },
        1: {
            "crop": "Tomato",
            "stage": "Flowering",
            "condition": "Leaf Disease"
        },
        2: {
            "crop": "Tomato",
            "stage": "Fruiting",
            "condition": "Leaf Spot"
        }
    }

    prediction_id = verdict["prediction"]

    if verdict["status"] == "confident":

        info = class_info.get(
            prediction_id,
            {
                "crop": "Unknown",
                "stage": "Unknown",
                "condition": "Unknown"
            }
        )

        return {
            "source": "student",
            "crop": info["crop"],
            "stage": info["stage"],
            "condition": info["condition"],
            "prediction": prediction_id,
            "confidence": verdict["confidence"],
            "status": "confident"
        }

    # Low-confidence prediction
    log_feedback_case(file, verdict)

    return {
        "source": "needs review",
        "crop": None,
        "stage": None,
        "condition": None,
        "prediction": None,
        "confidence": verdict["confidence"],
        "status": "needs_review"
    }