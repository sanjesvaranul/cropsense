import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report, accuracy_score


# ---------------- CONFIG ----------------

TEST_CSV = "data/labeled/test.csv"
MODEL_PATH = Path("models/cropsense_best.pth")
LABELS_PATH = Path("models/class_mapping.json")

IMAGE_SIZE = 224
BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ---------------- DATASET ----------------

class TestDataset(Dataset):
    def __init__(self, csv_path, crop_map, stage_map):
        self.df = pd.read_csv(csv_path)
        self.crop_map = crop_map
        self.stage_map = stage_map

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = Path(
            str(row["image_path"]).replace("\\", "/")
        )

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        image = self.transform(image)

        crop_label = self.crop_map[row["crop"]]
        stage_label = self.stage_map[row["growth_stage"]]

        return (
            image,
            torch.tensor(crop_label, dtype=torch.long),
            torch.tensor(stage_label, dtype=torch.long),
        )


# ---------------- MODEL ----------------

class CropSenseModel(nn.Module):
    def __init__(self, num_crops, num_stages):
        super().__init__()

        # Architecture must match the trained model
        backbone = models.efficientnet_b0(weights=None)

        self.features = backbone.features
        self.pool = backbone.avgpool

        feature_size = backbone.classifier[1].in_features

        self.crop_head = nn.Linear(feature_size, num_crops)
        self.stage_head = nn.Linear(feature_size, num_stages)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)

        crop_logits = self.crop_head(x)
        stage_logits = self.stage_head(x)

        return crop_logits, stage_logits


# ---------------- EVALUATION ----------------

def main():
    print("=" * 60)
    print("CropSense - Held-Out Test Evaluation")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Mapping not found: {LABELS_PATH}")

    # Load saved label mappings
    with open(LABELS_PATH, "r") as f:
        mappings = json.load(f)

    crop_map = mappings["crop_map"]
    stage_map = mappings["stage_map"]

    # Ensure class names follow the original training indices
    crop_classes = [
        name for name, idx in sorted(
            crop_map.items(), key=lambda item: item[1]
        )
    ]

    stage_classes = [
        name for name, idx in sorted(
            stage_map.items(), key=lambda item: item[1]
        )
    ]

    # Load checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model = CropSenseModel(
        num_crops=checkpoint["num_crops"],
        num_stages=checkpoint["num_stages"]
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test data
    test_dataset = TestDataset(
        TEST_CSV, crop_map, stage_map
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    true_crops, pred_crops = [], []
    true_stages, pred_stages = [], []

    with torch.no_grad():
        for images, crops, stages in test_loader:
            images = images.to(DEVICE)

            crop_logits, stage_logits = model(images)

            pred_crops.extend(
                crop_logits.argmax(1).cpu().tolist()
            )
            pred_stages.extend(
                stage_logits.argmax(1).cpu().tolist()
            )

            true_crops.extend(crops.tolist())
            true_stages.extend(stages.tolist())

    # Overall accuracy
    crop_acc = accuracy_score(true_crops, pred_crops)
    stage_acc = accuracy_score(true_stages, pred_stages)

    print("\nTEST RESULTS")
    print("-" * 60)
    print("Test images:", len(test_dataset))
    print(f"Crop accuracy:  {crop_acc * 100:.2f}%")
    print(f"Stage accuracy: {stage_acc * 100:.2f}%")

    # Detailed per-class metrics
    print("\nCROP CLASSIFICATION REPORT")
    print(classification_report(
        true_crops,
        pred_crops,
        labels=list(range(len(crop_classes))),
        target_names=crop_classes,
        zero_division=0
    ))

    print("\nGROWTH-STAGE CLASSIFICATION REPORT")
    print(classification_report(
        true_stages,
        pred_stages,
        labels=list(range(len(stage_classes))),
        target_names=stage_classes,
        zero_division=0
    ))

    print("=" * 60)
    print("Test evaluation completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
