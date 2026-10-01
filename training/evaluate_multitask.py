import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ---------------- CONFIG ----------------

TEST_CSV = "data/labeled/test_multitask.csv"
MODEL_PATH = Path("models/cropsense_best.pth")
LABELS_PATH = Path("models/class_mapping.json")
EVAL_DIR = Path("training/evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 16
CONF_FLOOR = 0.70

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ---------------- DATASET ----------------

class MultiTaskTestDataset(Dataset):
    def __init__(self, csv_path, crop_map, stage_map, condition_map):
        self.df = pd.read_csv(csv_path)
        self.crop_map = crop_map
        self.stage_map = stage_map
        self.condition_map = condition_map

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
        except Exception:
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


# ---------------- MAIN ----------------

def main():
    print("=" * 65)
    print("CropSense - Held-Out Multi-Task Test Evaluation")
    print("=" * 65)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels mapping not found: {LABELS_PATH}")

    with open(LABELS_PATH, "r") as f:
        mappings = json.load(f)

    crop_map = mappings["crop_map"]
    stage_map = mappings["stage_map"]
    condition_map = mappings["condition_map"]

    crop_classes = [name for name, _ in sorted(crop_map.items(), key=lambda x: x[1])]
    stage_classes = [name for name, _ in sorted(stage_map.items(), key=lambda x: x[1])]
    condition_classes = [name for name, _ in sorted(condition_map.items(), key=lambda x: x[1])]

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)

    num_crops = checkpoint.get("num_crops", len(crop_classes))
    num_stages = checkpoint.get("num_stages", len(stage_classes))
    num_conditions = checkpoint.get("num_conditions", len(condition_classes))

    model = CropSenseModel(num_crops, num_stages, num_conditions).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = MultiTaskTestDataset(TEST_CSV, crop_map, stage_map, condition_map)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    true_crops, pred_crops, conf_crops = [], [], []
    true_stages, pred_stages, conf_stages = [], [], []
    true_conds, pred_conds, conf_conds = [], [], []

    with torch.no_grad():
        for images, crops, stages, conditions in test_loader:
            images = images.to(DEVICE)

            crop_logits, stage_logits, cond_logits = model(images)

            c_probs = torch.softmax(crop_logits, dim=-1)
            s_probs = torch.softmax(stage_logits, dim=-1)
            cond_probs = torch.softmax(cond_logits, dim=-1)

            c_max_conf, c_pred = c_probs.max(dim=-1)
            s_max_conf, s_pred = s_probs.max(dim=-1)
            cond_max_conf, cond_pred = cond_probs.max(dim=-1)

            pred_crops.extend(c_pred.cpu().tolist())
            pred_stages.extend(s_pred.cpu().tolist())
            pred_conds.extend(cond_pred.cpu().tolist())

            conf_crops.extend(c_max_conf.cpu().tolist())
            conf_stages.extend(s_max_conf.cpu().tolist())
            conf_conds.extend(cond_max_conf.cpu().tolist())

            true_crops.extend(crops.tolist())
            true_stages.extend(stages.tolist())
            true_conds.extend(conditions.tolist())

    # Accuracies
    crop_acc = accuracy_score(true_crops, pred_crops)
    stage_acc = accuracy_score(true_stages, pred_stages)
    cond_acc = accuracy_score(true_conds, pred_conds)

    # Abstention / Confidence verification
    confident_count = sum(
        1 for c, s, d in zip(conf_crops, conf_stages, conf_conds)
        if c >= CONF_FLOOR and s >= CONF_FLOOR and d >= CONF_FLOOR
    )
    abstention_rate = 1.0 - (confident_count / len(test_dataset))

    summary_lines = []
    summary_lines.append("=" * 65)
    summary_lines.append("CropSense - Comprehensive Multi-Task Held-Out Test Evaluation")
    summary_lines.append("=" * 65)
    summary_lines.append(f"Total Test Samples   : {len(test_dataset)}")
    summary_lines.append(f"Crop Accuracy        : {crop_acc * 100:.2f}%")
    summary_lines.append(f"Stage Accuracy       : {stage_acc * 100:.2f}%")
    summary_lines.append(f"Disease Accuracy     : {cond_acc * 100:.2f}%")
    summary_lines.append(f"High-Confidence Pass : {confident_count}/{len(test_dataset)} ({confident_count/len(test_dataset)*100:.2f}%)")
    summary_lines.append(f"Abstention / Review  : {len(test_dataset)-confident_count}/{len(test_dataset)} ({abstention_rate*100:.2f}%)")
    summary_lines.append("-" * 65)

    summary_lines.append("\n" + "=" * 30 + " CROP REPORT " + "=" * 22)
    crop_rep = classification_report(
        true_crops, pred_crops, labels=list(range(len(crop_classes))), target_names=crop_classes, zero_division=0
    )
    summary_lines.append(crop_rep)

    summary_lines.append("\n" + "=" * 30 + " STAGE REPORT " + "=" * 21)
    stage_rep = classification_report(
        true_stages, pred_stages, labels=list(range(len(stage_classes))), target_names=stage_classes, zero_division=0
    )
    summary_lines.append(stage_rep)

    summary_lines.append("\n" + "=" * 28 + " DISEASE/CONDITION REPORT " + "=" * 11)
    cond_rep = classification_report(
        true_conds, pred_conds, labels=list(range(len(condition_classes))), target_names=condition_classes, zero_division=0
    )
    summary_lines.append(cond_rep)

    full_summary = "\n".join(summary_lines)
    print(full_summary)

    # Save summary
    with open(EVAL_DIR / "multitask_evaluation_summary.txt", "w") as f:
        f.write(full_summary)

    # Save Confusion Matrices
    crop_cm = pd.DataFrame(
        confusion_matrix(true_crops, pred_crops, labels=list(range(len(crop_classes)))),
        index=crop_classes, columns=crop_classes
    )
    crop_cm.to_csv(EVAL_DIR / "crop_confusion_matrix.csv")

    stage_cm = pd.DataFrame(
        confusion_matrix(true_stages, pred_stages, labels=list(range(len(stage_classes)))),
        index=stage_classes, columns=stage_classes
    )
    stage_cm.to_csv(EVAL_DIR / "stage_confusion_matrix.csv")

    cond_cm = pd.DataFrame(
        confusion_matrix(true_conds, pred_conds, labels=list(range(len(condition_classes)))),
        index=condition_classes, columns=condition_classes
    )
    cond_cm.to_csv(EVAL_DIR / "condition_confusion_matrix.csv")

    print(f"\nAll reports and confusion matrices saved to: {EVAL_DIR.resolve()}")


if __name__ == "__main__":
    main()
