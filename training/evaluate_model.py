import os
import json
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

VAL_CSV = "data/labeled/val.csv"
IMAGE_DIR = "data/raw/images"
MODEL_DIR = "training/models"
OUTPUT_DIR = "training/evaluation"

IMAGE_SIZE = 224
BATCH_SIZE = 16


class CropDataset(Dataset):

    def __init__(self, csv_path, crop_to_id, stage_to_id):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)

        self.crop_to_id = crop_to_id
        self.stage_to_id = stage_to_id

        # No random augmentation during evaluation
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = os.path.join(
            IMAGE_DIR,
            row["image_filename"]
        )

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        crop_id = self.crop_to_id[row["crop"]]
        stage_id = self.stage_to_id[row["growth_stage"]]

        return image, crop_id, stage_id


class CropSenseModel(nn.Module):

    def __init__(self, num_crops, num_stages):
        super().__init__()

        # Same architecture used during training
        self.backbone = models.mobilenet_v3_small(
            weights=None
        )

        feature_size = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()

        self.crop_head = nn.Linear(feature_size, num_crops)
        self.stage_head = nn.Linear(feature_size, num_stages)

    def forward(self, x):
        features = self.backbone(x)

        crop_logits = self.crop_head(features)
        stage_logits = self.stage_head(features)

        return crop_logits, stage_logits


def main():

    print("=" * 60)
    print("CropSense - Model Evaluation")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load label mappings created during training
    with open(os.path.join(MODEL_DIR, "label_maps.json")) as f:
        label_maps = json.load(f)

    crop_to_id = label_maps["crop_to_id"]
    stage_to_id = label_maps["stage_to_id"]

    crop_names = [
        name for name, idx in sorted(
            crop_to_id.items(), key=lambda x: x[1]
        )
    ]

    stage_names = [
        name for name, idx in sorted(
            stage_to_id.items(), key=lambda x: x[1]
        )
    ]

    # Load validation data
    dataset = CropDataset(
        VAL_CSV, crop_to_id, stage_to_id
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")
    print(f"Validation images: {len(dataset)}")

    # Load trained model
    model = CropSenseModel(
        num_crops=len(crop_names),
        num_stages=len(stage_names)
    )

    model_path = os.path.join(
        MODEL_DIR, "cropsense_crop_stage.pth"
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    model.to(device)
    model.eval()

    true_crops, pred_crops = [], []
    true_stages, pred_stages = [], []

    # Run validation predictions
    with torch.no_grad():

        for images, crop_labels, stage_labels in loader:

            images = images.to(device)

            crop_logits, stage_logits = model(images)

            crop_preds = crop_logits.argmax(dim=1).cpu().tolist()
            stage_preds = stage_logits.argmax(dim=1).cpu().tolist()

            true_crops.extend(crop_labels.tolist())
            pred_crops.extend(crop_preds)

            true_stages.extend(stage_labels.tolist())
            pred_stages.extend(stage_preds)

    # Calculate accuracy
    crop_acc = accuracy_score(true_crops, pred_crops)
    stage_acc = accuracy_score(true_stages, pred_stages)

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print(f"Crop accuracy:        {crop_acc:.4f} ({crop_acc * 100:.2f}%)")
    print(f"Growth-stage accuracy: {stage_acc:.4f} ({stage_acc * 100:.2f}%)")

    # Detailed classification reports
    crop_report = classification_report(
        true_crops,
        pred_crops,
        labels=list(range(len(crop_names))),
        target_names=crop_names,
        zero_division=0
    )

    stage_report = classification_report(
        true_stages,
        pred_stages,
        labels=list(range(len(stage_names))),
        target_names=stage_names,
        zero_division=0
    )

    print("\nCROP CLASSIFICATION REPORT")
    print(crop_report)

    print("\nGROWTH-STAGE CLASSIFICATION REPORT")
    print(stage_report)

    # Save confusion matrices
    crop_cm = confusion_matrix(
        true_crops,
        pred_crops,
        labels=list(range(len(crop_names)))
    )

    stage_cm = confusion_matrix(
        true_stages,
        pred_stages,
        labels=list(range(len(stage_names)))
    )

    pd.DataFrame(
        crop_cm,
        index=crop_names,
        columns=crop_names
    ).to_csv(os.path.join(OUTPUT_DIR, "crop_confusion_matrix.csv"))

    pd.DataFrame(
        stage_cm,
        index=stage_names,
        columns=stage_names
    ).to_csv(os.path.join(OUTPUT_DIR, "stage_confusion_matrix.csv"))

    # Save summary
    with open(os.path.join(OUTPUT_DIR, "evaluation_summary.txt"), "w") as f:
        f.write(f"Validation images: {len(dataset)}\n")
        f.write(f"Crop accuracy: {crop_acc:.4f}\n")
        f.write(f"Growth-stage accuracy: {stage_acc:.4f}\n\n")
        f.write("CROP CLASSIFICATION REPORT\n")
        f.write(crop_report)
        f.write("\nGROWTH-STAGE CLASSIFICATION REPORT\n")
        f.write(stage_report)

    print("\nEvaluation files saved in:", OUTPUT_DIR)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
