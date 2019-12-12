# Importation of librairies
import torch
import numpy as np

import pdb

# Evaluation functions
def compute_performance_metrics(labels_cat, labels_seg, prediction, weights, parts_tensor, zero_tensor, one_tensor):
    batch_size, nb_points, nb_parts = prediction.size()

    # Filtering out the parts not related to the class object
    spread = torch.max(prediction) - torch.min(prediction)
    prediction = prediction + parts_tensor.unsqueeze(1) * spread
    _, labels_pred = torch.max(prediction, dim=2)

    # Weighted Accuracy
    mask_equality = (labels_pred == labels_seg).float()
    if weights is not None:
        mask_equality = weights * mask_equality
        normalization = torch.sum(weights).item()
    else:
        normalization = batch_size * nb_points
    batch_accuracy = torch.sum(mask_equality).item() / normalization

    # Intersection & Union
    labels_pred_1hot = zero_tensor.clone().view(1, 1, 1).repeat(batch_size, nb_points, nb_parts)
    labels_pred_1hot = labels_pred_1hot.scatter_(2, labels_pred.unsqueeze(2), 1)
    labels_seg_1hot = zero_tensor.clone().view(1, 1, 1).repeat(batch_size, nb_points, nb_parts)
    labels_seg_1hot = labels_seg_1hot.scatter_(2, labels_seg.unsqueeze(2), 1)
    intersection = labels_pred_1hot * labels_seg_1hot
    if weights is not None:
        intersection = weights.unsqueeze(2) * intersection
    intersection = torch.sum(intersection, dim=1)
    union = labels_pred_1hot + labels_seg_1hot
    if weights is not None:
        union = weights.unsqueeze(2) * union
    union = torch.sum(union, dim=1) - intersection
    batch_intersection = torch.sum(intersection, dim=0)
    batch_union = torch.sum(union, dim=0)
    
    # Intersection Over Union
    batch_iou = intersection / union
    mask = parts_tensor.byte()
    batch_iou = torch.where(mask, batch_iou, zero_tensor.clone())
    mask = parts_tensor.byte() & (union == 0).byte()
    batch_iou = torch.where(mask, one_tensor.clone(), batch_iou)
    batch_iou = torch.sum(batch_iou, dim=1) / torch.sum(parts_tensor, dim=1)
    batch_iou = torch.mean(batch_iou, dim=0)
    
    return batch_accuracy, batch_iou, batch_intersection, batch_union

if __name__ == '__main__':
    batch_size = 4
    num_points = 2048
    num_parts = 50
    num_classes = 16

    labels_cat = torch.randint(0, num_classes, (batch_size,))
    labels_seg = torch.randint(0, num_parts, (batch_size, num_points))
    prediction = torch.randn(batch_size, num_points, num_parts)

    parts_tensor = torch.randint(0, 1, (batch_size, num_parts))

    zero_tensor = torch.zeros([1])
    one_tensor = torch.ones([1])
    batch_accuracy, batch_iou, batch_union, batch_intersection = compute_performance_metrics(labels_cat, labels_seg, prediction, None, parts_tensor, zero_tensor, one_tensor, test_time=True)