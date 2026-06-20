import torch


def dice_score_from_logits(logits, targets, threshold: float = 0.5, eps: float = 1e-7):
    predictions = (torch.sigmoid(logits) > threshold).float()
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    total = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (total + eps)).mean()


def iou_score_from_logits(logits, targets, threshold: float = 0.5, eps: float = 1e-7):
    predictions = (torch.sigmoid(logits) > threshold).float()
    intersection = (predictions * targets).sum(dim=(1, 2, 3))
    union = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    return ((intersection + eps) / (union + eps)).mean()

