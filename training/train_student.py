import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights


# ---------------- CONFIG ----------------

TRAIN_CSV = "data/labeled/train.csv"
VAL_CSV = "data/labeled/val.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "cropsense_best.pth"
LABELS_PATH = MODEL_DIR / "class_mapping.json"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ---------------- DATASET ----------------

class CropDataset(Dataset):
    def __init__(self, csv_path, crop_map, stage_map, training=False):
        self.df = pd.read_csv(csv_path)
        self.crop_map = crop_map
        self.stage_map = stage_map

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip() if training
            else transforms.Lambda(lambda x: x),
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

        weights = EfficientNet_B0_Weights.DEFAULT
        backbone = models.efficientnet_b0(weights=weights)

        self.features = backbone.features
        self.pool = backbone.avgpool

        # Freeze pretrained feature extractor initially
        for param in self.features.parameters():
            param.requires_grad = False

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


# ---------------- TRAINING ----------------

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    crop_correct = 0
    stage_correct = 0
    total = 0

    with torch.no_grad():
        for images, crops, stages in loader:
            images = images.to(device)
            crops = crops.to(device)
            stages = stages.to(device)

            crop_logits, stage_logits = model(images)

            loss = (
                criterion(crop_logits, crops)
                + criterion(stage_logits, stages)
            ) / 2

            total_loss += loss.item() * len(images)

            crop_correct += (
                crop_logits.argmax(1) == crops
            ).sum().item()

            stage_correct += (
                stage_logits.argmax(1) == stages
            ).sum().item()

            total += len(images)

    return (
        total_loss / total,
        crop_correct / total,
        stage_correct / total,
    )


def main():
    print("=" * 55)
    print("CropSense - Model Training")
    print("=" * 55)
    print("Device:", DEVICE)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    crop_classes = sorted(train_df["crop"].unique())
    stage_classes = sorted(train_df["growth_stage"].unique())

    crop_map = {
        name: idx for idx, name in enumerate(crop_classes)
    }

    stage_map = {
        name: idx for idx, name in enumerate(stage_classes)
    }

    # Stop if validation contains a label unseen in training
    unknown_crops = set(val_df["crop"]) - set(crop_classes)
    unknown_stages = set(val_df["growth_stage"]) - set(stage_classes)

    if unknown_crops or unknown_stages:
        raise ValueError(
            f"Unknown validation labels. Crops: {unknown_crops}, "
            f"Stages: {unknown_stages}"
        )

    print("Crop classes:", len(crop_classes))
    print("Growth-stage classes:", len(stage_classes))
    print("Training rows:", len(train_df))
    print("Validation rows:", len(val_df))

    with open(LABELS_PATH, "w") as f:
        json.dump(
            {
                "crop_map": crop_map,
                "stage_map": stage_map,
            },
            f,
            indent=2,
        )

    train_dataset = CropDataset(
        TRAIN_CSV, crop_map, stage_map, training=True
    )

    val_dataset = CropDataset(
        VAL_CSV, crop_map, stage_map, training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = CropSenseModel(
        len(crop_classes), len(stage_classes)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0

        for images, crops, stages in train_loader:
            images = images.to(DEVICE)
            crops = crops.to(DEVICE)
            stages = stages.to(DEVICE)

            optimizer.zero_grad()

            crop_logits, stage_logits = model(images)

            crop_loss = criterion(crop_logits, crops)
            stage_loss = criterion(stage_logits, stages)

            loss = (crop_loss + stage_loss) / 2

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(images)

        train_loss = total_loss / len(train_dataset)

        val_loss, crop_acc, stage_acc = evaluate(
            model, val_loader, criterion, DEVICE
        )

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train loss: {train_loss:.4f}")
        print(f"Val loss:   {val_loss:.4f}")
        print(f"Crop accuracy:  {crop_acc * 100:.2f}%")
        print(f"Stage accuracy: {stage_acc * 100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_crops": len(crop_classes),
                    "num_stages": len(stage_classes),
                    "image_size": IMAGE_SIZE,
                    "mean": MEAN,
                    "std": STD,
                },
                MODEL_PATH,
            )

            print("Saved best model:", MODEL_PATH)

    print("\nTraining completed.")
    print("Model:", MODEL_PATH)
    print("Labels:", LABELS_PATH)


if __name__ == "__main__":
    main()
