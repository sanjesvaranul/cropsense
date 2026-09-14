import os, json, requests
from dotenv import load_dotenv

load_dotenv()
TEACHER_API_KEY = os.getenv("TEACHER_API_KEY")

LABEL_SCHEMA_PROMPT = """
Analyze this crop photograph and return ONLY valid JSON:
{
  "crop": "<crop name>",
  "stage": "<growth stage>",
  "condition": "healthy | <disease name>",
  "confidence": <0.0-1.0>,
  "evidence": "<short description of what you observed>"
}
"""

def label_image(image_path: str) -> dict:
    # Encode image, call the approved teacher model's vision endpoint
    # with LABEL_SCHEMA_PROMPT. Swap in actual SDK call as needed.
    # response = call_teacher_model(image_path, LABEL_SCHEMA_PROMPT)
    # return json.loads(response)
    pass

def label_batch(image_dir: str, out_path: str):
    results = []
    for fname in os.listdir(image_dir):
        label = label_image(os.path.join(image_dir, fname))
        label["filename"] = fname
        results.append(label)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)