import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights

# ---------------- CONFIG ----------------

TRAIN_CSV = "data/labeled/train_multitask.csv"
VAL_CSV = "data/labeled/val_multitask.csv"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "cropsense_best.pth"
BACKUP_BASE_PATH = MODEL_DIR / "cropsense_crop_stage_only.pth"
LABELS_PATH = MODEL_DIR / "class_mapping.json"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ---------------- DATASET ----------------

class MultiTaskCropDataset(Dataset):
    def __init__(self, csv_path, crop_map, stage_map, condition_map, training=False):
        self.df = pd.read_csv(csv_path)
        self.crop_map = crop_map
        self.stage_map = stage_map
        self.condition_map = condition_map

        if training:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = Path(str(row["image_path"]).replace("\\", "/"))

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            # Fallback for any corrupt image: black placeholder
            print(f"Warning: Error opening {image_path}: {e}")
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))

        image = self.transform(image)

        crop_label = self.crop_map[row["crop"]]
        stage_label = self.stage_map[row["growth_stage"]]
        condition_label = self.condition_map[row["condition"]]

        return (
            image,
            torch.tensor(crop_label, dtype=torch.long),
            torch.tensor(stage_label, dtype=torch.long),
            torch.tensor(condition_label, dtype=torch.long),
        )


# ---------------- MODEL ----------------

class CropSenseModel(nn.Module):
    def __init__(self, num_crops, num_stages, num_conditions):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT
        backbone = models.efficientnet_b0(weights=weights)

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


# ---------------- EVALUATE ----------------

def evaluate(model, loader, crop_crit, stage_crit, cond_crit, device):
    model.eval()

    total_loss = 0.0
    crop_correct = 0
    stage_correct = 0
    cond_correct = 0
    total = 0

    with torch.no_grad():
        for images, crops, stages, conditions in loader:
            images = images.to(device)
            crops = crops.to(device)
            stages = stages.to(device)
            conditions = conditions.to(device)

            crop_logits, stage_logits, cond_logits = model(images)

            loss = (
                crop_crit(crop_logits, crops)
                + stage_crit(stage_logits, stages)
                + cond_crit(cond_logits, conditions)
            ) / 3.0

            total_loss += loss.item() * len(images)

            crop_correct += (crop_logits.argmax(1) == crops).sum().item()
            stage_correct += (stage_logits.argmax(1) == stages).sum().item()
            cond_correct += (cond_logits.argmax(1) == conditions).sum().item()
            total += len(images)

    return (
        total_loss / total,
        crop_correct / total,
        stage_correct / total,
        cond_correct / total,
    )


# ---------------- MAIN ----------------

def main():
    print("=" * 60)
    print("CropSense - Unified Multi-Task Model Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    with open(LABELS_PATH, "r") as f:
        mappings = json.load(f)

    crop_map = mappings["crop_map"]
    stage_map = mappings["stage_map"]
    condition_map = mappings["condition_map"]

    num_crops = len(crop_map)
    num_stages = len(stage_map)
    num_conditions = len(condition_map)

    print(f"Crops: {num_crops}, Stages: {num_stages}, Conditions: {num_conditions}")
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Compute balanced weights for condition loss
    cond_counts = np.zeros(num_conditions, dtype=np.float32)
    for c_idx in train_df["condition"].map(condition_map):
        cond_counts[c_idx] += 1

    cond_weights_arr = len(train_df) / (num_conditions * np.maximum(cond_counts, 1.0))
    # Normalize and clip weights to avoid excessive gradient explosion
    cond_weights_arr = np.clip(cond_weights_arr / cond_weights_arr.mean(), 0.3, 4.0)
    condition_weights = torch.tensor(cond_weights_arr, dtype=torch.float).to(DEVICE)
    print("Condition loss weights:", {name: round(cond_weights_arr[idx], 2) for name, idx in condition_map.items()})

    train_dataset = MultiTaskCropDataset(
        TRAIN_CSV, crop_map, stage_map, condition_map, training=True
    )
    val_dataset = MultiTaskCropDataset(
        VAL_CSV, crop_map, stage_map, condition_map, training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = CropSenseModel(num_crops, num_stages, num_conditions).to(DEVICE)

    # Initialize from existing trained model checkpoint if available
    if BACKUP_BASE_PATH.exists():
        print(f"Loading weights from existing checkpoint: {BACKUP_BASE_PATH}")
        ckpt = torch.load(BACKUP_BASE_PATH, map_location=DEVICE, weights_only=True)
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v for k, v in ckpt["model_state_dict"].items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Successfully transferred {len(pretrained_dict)} layers!")

    crop_criterion = nn.CrossEntropyLoss()
    stage_criterion = nn.CrossEntropyLoss()
    condition_criterion = nn.CrossEntropyLoss(weight=condition_weights)

    # Differential learning rate: condition head learns faster, pretrained layers fine-tune gently
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
        {"params": model.crop_head.parameters(), "lr": 2e-4, "weight_decay": 1e-4},
        {"params": model.stage_head.parameters(), "lr": 2e-4, "weight_decay": 1e-4},
        {"params": model.condition_head.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
    ])

    best_val_loss = float("inf")
    best_cond_acc = 0.0

    print("\nStarting training loop...")
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()

        running_loss = 0.0
        train_crop_corr = 0
        train_stage_corr = 0
        train_cond_corr = 0
        train_total = 0

        for batch_i, (images, crops, stages, conditions) in enumerate(train_loader):
            images = images.to(DEVICE)
            crops = crops.to(DEVICE)
            stages = stages.to(DEVICE)
            conditions = conditions.to(DEVICE)

            optimizer.zero_grad()

            crop_logits, stage_logits, cond_logits = model(images)

            c_loss = crop_criterion(crop_logits, crops)
            s_loss = stage_criterion(stage_logits, stages)
            cond_loss = condition_criterion(cond_logits, conditions)

            total_loss = c_loss + s_loss + 1.2 * cond_loss

            total_loss.backward()
            optimizer.step()

            bs = len(images)
            running_loss += total_loss.item() * bs
            train_crop_corr += (crop_logits.argmax(1) == crops).sum().item()
            train_stage_corr += (stage_logits.argmax(1) == stages).sum().item()
            train_cond_corr += (cond_logits.argmax(1) == conditions).sum().item()
            train_total += bs

        train_loss = running_loss / train_total
        val_loss, val_crop_acc, val_stage_acc, val_cond_acc = evaluate(
            model, val_loader, crop_criterion, stage_criterion, condition_criterion, DEVICE
        )

        elapsed = time.time() - start_time
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ({elapsed:.1f}s) ---")
        print(f"Train Loss: {train_loss:.4f} | Crop Acc: {train_crop_corr/train_total*100:.2f}% | Stage Acc: {train_stage_corr/train_total*100:.2f}% | Cond Acc: {train_cond_corr/train_total*100:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Crop Acc: {val_crop_acc*100:.2f}% | Stage Acc: {val_stage_acc*100:.2f}% | Cond Acc: {val_cond_acc*100:.2f}%")

        # Save checkpoint if better val loss or improved disease accuracy
        if val_loss < best_val_loss or val_cond_acc > best_cond_acc:
            best_val_loss = min(val_loss, best_val_loss)
            best_cond_acc = max(val_cond_acc, best_cond_acc)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_crops": num_crops,
                    "num_stages": num_stages,
                    "num_conditions": num_conditions,
                    "image_size": IMAGE_SIZE,
                    "mean": MEAN,
                    "std": STD,
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_crop_acc": val_crop_acc,
                    "val_stage_acc": val_stage_acc,
                    "val_cond_acc": val_cond_acc,
                },
                MODEL_PATH,
            )
            print(f" Saved new best model checkpoint -> {MODEL_PATH}")

    print("\n" + "=" * 60)
    print("Multi-task model training completed successfully!")
    print(f"Checkpoint saved: {MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
