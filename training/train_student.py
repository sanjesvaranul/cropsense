import torch
import torch.nn as nn
from transformers import AutoModel

BACKBONE = "google/siglip-base-patch16-224"  # or CLIP/EfficientNet equivalent

class CropClassifier(nn.Module):
    def __init__(self, backbone_name: str, n_crop: int, n_stage: int, n_condition: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.crop_head = nn.Linear(hidden, n_crop)
        self.stage_head = nn.Linear(hidden, n_stage)
        self.condition_head = nn.Linear(hidden, n_condition)

    def forward(self, pixel_values):
        feats = self.backbone(pixel_values=pixel_values).pooler_output
        return self.crop_head(feats), self.stage_head(feats), self.condition_head(feats)