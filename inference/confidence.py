import torch
import torch.nn.functional as F

CONF_FLOOR = 0.70

def predict_with_abstention(logits: torch.Tensor) -> dict:
    probs = F.softmax(logits, dim=-1)
    conf, pred = probs.max(dim=-1)
    if conf.item() < CONF_FLOOR:
        return {"prediction": None, "confidence": conf.item(), "status": "needs_review"}
    return {"prediction": pred.item(), "confidence": conf.item(), "status": "confident"}