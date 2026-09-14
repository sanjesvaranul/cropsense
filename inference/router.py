from types import SimpleNamespace

import torch
from fastapi import FastAPI, UploadFile

from .confidence import predict_with_abstention

app = FastAPI()


def run_student_model(file):
    """
    Temporary mock student model.

    This will be replaced by the real fine-tuned CropSense
    model once the dataset and training stage are available.
    """
    # Mock logits for 3 example classes
    logits = torch.tensor([[3.0, 1.0, 0.5]])

    return SimpleNamespace(logits=logits)


def log_feedback_case(file, verdict, teacher_result=None):
    """
    Temporary feedback logger.

    Will be connected to the versioned feedback dataset later.
    """
    print("Feedback case logged:", verdict)


@app.get("/")
def health_check():
    return {"status": "CropSense API is running"}


@app.post("/predict")
async def predict(file: UploadFile):
    result = run_student_model(file)

    verdict = predict_with_abstention(result.logits)

    if verdict["status"] == "confident":
        return {
            "source": "student",
            **verdict
        }

    return {
        "source": "needs review",
        **verdict
    }