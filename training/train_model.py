import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

TRAIN_CSV = "data/labeled/train.csv"
VAL_CSV = "data/labeled/val.csv"
IMAGE_DIR = "data/raw/images"

MODEL_DIR = "training/models"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 0.0001


class CropDataset(Dataset):

    def __init__(self, csv_path, crop_to_id, stage_to_id):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)

        self.crop_to_id = crop_to_id
        self.stage_to_id = stage_to_id

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
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

        weights = models.MobileNet_V3_Small_Weights.DEFAULT

        self.backbone = models.mobilenet_v3_small(
            weights=weights
        )

        feature_size = self.backbone.classifier[0].in_features

        self.backbone.classifier = nn.Identity()

        self.crop_head = nn.Linear(
            feature_size,
            num_crops
        )

        self.stage_head = nn.Linear(
            feature_size,
            num_stages
        )

    def forward(self, x):

        features = self.backbone(x)

        crop_logits = self.crop_head(features)
        stage_logits = self.stage_head(features)

        return crop_logits, stage_logits


def main():

    print("=" * 60)
    print("CropSense - Model Training")
    print("=" * 60)

    os.makedirs(MODEL_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    # Create label mappings from the complete dataset
    # so validation-only classes do not cause KeyError.
    crops = sorted(
    	set(train_df["crop"].unique()) |
    	set(val_df["crop"].unique())
    )

    stages = sorted(
    	set(train_df["growth_stage"].unique()) |
    	set(val_df["growth_stage"].unique())
    )

    crop_to_id = {
        name: i
        for i, name in enumerate(crops)
    }

    stage_to_id = {
        name: i
        for i, name in enumerate(stages)
    }

    print("\nCrop classes:")
    for name, idx in crop_to_id.items():
        print(f"  {idx}: {name}")

    print("\nGrowth-stage classes:")
    for name, idx in stage_to_id.items():
        print(f"  {idx}: {name}")

    train_dataset = CropDataset(
        TRAIN_CSV,
        crop_to_id,
        stage_to_id
    )

    val_dataset = CropDataset(
        VAL_CSV,
        crop_to_id,
        stage_to_id
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nDevice: {device}")

    model = CropSenseModel(
        num_crops=len(crops),
        num_stages=len(stages)
    )

    model = model.to(device)

    crop_loss_fn = nn.CrossEntropyLoss()
    stage_loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0
        correct_crop = 0
        correct_stage = 0
        total = 0

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        for batch_idx, (images, crop_labels, stage_labels) in enumerate(train_loader):

            images = images.to(device)
            crop_labels = crop_labels.to(device)
            stage_labels = stage_labels.to(device)

            optimizer.zero_grad()

            crop_logits, stage_logits = model(images)

            crop_loss = crop_loss_fn(
                crop_logits,
                crop_labels
            )

            stage_loss = stage_loss_fn(
                stage_logits,
                stage_labels
            )

            loss = crop_loss + stage_loss

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            crop_pred = crop_logits.argmax(dim=1)
            stage_pred = stage_logits.argmax(dim=1)

            correct_crop += (
                crop_pred == crop_labels
            ).sum().item()

            correct_stage += (
                stage_pred == stage_labels
            ).sum().item()

            total += images.size(0)

            if (batch_idx + 1) % 20 == 0:

                print(
                    f"Batch {batch_idx + 1}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        train_crop_acc = correct_crop / total
        train_stage_acc = correct_stage / total

        print(
            f"\nTraining loss: {total_loss / len(train_loader):.4f}"
        )

        print(
            f"Training crop accuracy: "
            f"{train_crop_acc:.4f}"
        )

        print(
            f"Training stage accuracy: "
            f"{train_stage_acc:.4f}"
        )

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        model.eval()

        val_correct_crop = 0
        val_correct_stage = 0
        val_total = 0

        with torch.no_grad():

            for images, crop_labels, stage_labels in val_loader:

                images = images.to(device)
                crop_labels = crop_labels.to(device)
                stage_labels = stage_labels.to(device)

                crop_logits, stage_logits = model(images)

                crop_pred = crop_logits.argmax(dim=1)
                stage_pred = stage_logits.argmax(dim=1)

                val_correct_crop += (
                    crop_pred == crop_labels
                ).sum().item()

                val_correct_stage += (
                    stage_pred == stage_labels
                ).sum().item()

                val_total += images.size(0)

        val_crop_acc = val_correct_crop / val_total
        val_stage_acc = val_correct_stage / val_total

        print(
            f"Validation crop accuracy: "
            f"{val_crop_acc:.4f}"
        )

        print(
            f"Validation stage accuracy: "
            f"{val_stage_acc:.4f}"
        )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "cropsense_crop_stage.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    with open(
        os.path.join(MODEL_DIR, "label_maps.json"),
        "w"
    ) as f:

        json.dump(
            {
                "crop_to_id": crop_to_id,
                "stage_to_id": stage_to_id
            },
            f,
            indent=2
        )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(f"\nModel saved to:")
    print(model_path)


if __name__ == "__main__":
    main()
