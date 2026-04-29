from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SegmentationSample:
    image_id: str
    pred_mask: list[list[int]]
    gt_mask: list[list[int]]


def _validate_binary_mask(mask: list[list[int]]) -> tuple[int, int]:
    if not mask or not mask[0]:
        raise ValueError("Mask must not be empty.")
    rows = len(mask)
    cols = len(mask[0])
    for row in mask:
        if len(row) != cols:
            raise ValueError("All mask rows must have equal length.")
        for value in row:
            if value not in (0, 1):
                raise ValueError("Binary masks must contain only 0 or 1.")
    return rows, cols


def _validate_same_shape(
    pred_mask: list[list[int]], gt_mask: list[list[int]]
) -> tuple[int, int]:
    pred_rows, pred_cols = _validate_binary_mask(pred_mask)
    gt_rows, gt_cols = _validate_binary_mask(gt_mask)
    if pred_rows != gt_rows or pred_cols != gt_cols:
        raise ValueError("Prediction and ground-truth masks must have identical shapes.")
    return pred_rows, pred_cols


def dice_score(pred_mask: list[list[int]], gt_mask: list[list[int]]) -> float:
    _validate_same_shape(pred_mask, gt_mask)
    intersection = 0
    pred_count = 0
    gt_count = 0
    for pred_row, gt_row in zip(pred_mask, gt_mask):
        for pred_value, gt_value in zip(pred_row, gt_row):
            if pred_value == 1:
                pred_count += 1
            if gt_value == 1:
                gt_count += 1
            if pred_value == 1 and gt_value == 1:
                intersection += 1

    denom = pred_count + gt_count
    if denom == 0:
        return 1.0
    return (2.0 * intersection) / denom


def pixel_accuracy(pred_mask: list[list[int]], gt_mask: list[list[int]]) -> float:
    rows, cols = _validate_same_shape(pred_mask, gt_mask)
    matches = 0
    total = rows * cols
    for pred_row, gt_row in zip(pred_mask, gt_mask):
        for pred_value, gt_value in zip(pred_row, gt_row):
            if pred_value == gt_value:
                matches += 1
    return matches / total if total else 0.0


def mask_iou(pred_mask: list[list[int]], gt_mask: list[list[int]]) -> float:
    _validate_same_shape(pred_mask, gt_mask)
    intersection = 0
    union = 0
    for pred_row, gt_row in zip(pred_mask, gt_mask):
        for pred_value, gt_value in zip(pred_row, gt_row):
            if pred_value == 1 and gt_value == 1:
                intersection += 1
            if pred_value == 1 or gt_value == 1:
                union += 1
    if union == 0:
        return 1.0
    return intersection / union


def mean_iou_multiclass(
    pred_mask: list[list[int]],
    gt_mask: list[list[int]],
    *,
    labels: list[int] | None = None,
) -> float:
    if not pred_mask or not gt_mask:
        raise ValueError("Masks must not be empty.")
    if len(pred_mask) != len(gt_mask) or len(pred_mask[0]) != len(gt_mask[0]):
        raise ValueError("Prediction and ground-truth masks must have identical shapes.")

    for pred_row, gt_row in zip(pred_mask, gt_mask):
        if len(pred_row) != len(gt_row):
            raise ValueError("Prediction and ground-truth masks must have identical shapes.")

    if labels is None:
        labels_set = set()
        for pred_row, gt_row in zip(pred_mask, gt_mask):
            labels_set.update(pred_row)
            labels_set.update(gt_row)
        labels = sorted(labels_set)

    if not labels:
        return 1.0

    ious: list[float] = []
    for label in labels:
        intersection = 0
        union = 0
        for pred_row, gt_row in zip(pred_mask, gt_mask):
            for pred_value, gt_value in zip(pred_row, gt_row):
                pred_match = pred_value == label
                gt_match = gt_value == label
                if pred_match and gt_match:
                    intersection += 1
                if pred_match or gt_match:
                    union += 1
        if union == 0:
            continue
        ious.append(intersection / union)

    if not ious:
        return 1.0
    return sum(ious) / len(ious)


def evaluate_segmentation_suite(
    samples: list[SegmentationSample],
) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples must not be empty.")

    dice_values: list[float] = []
    pixel_values: list[float] = []
    iou_values: list[float] = []
    multiclass_miou_values: list[float] = []

    for sample in samples:
        dice_values.append(dice_score(sample.pred_mask, sample.gt_mask))
        pixel_values.append(pixel_accuracy(sample.pred_mask, sample.gt_mask))
        iou_values.append(mask_iou(sample.pred_mask, sample.gt_mask))
        multiclass_miou_values.append(
            mean_iou_multiclass(sample.pred_mask, sample.gt_mask)
        )

    return {
        "num_samples": len(samples),
        "dice_mean": sum(dice_values) / len(dice_values),
        "pixel_accuracy_mean": sum(pixel_values) / len(pixel_values),
        "binary_iou_mean": sum(iou_values) / len(iou_values),
        "multiclass_miou_mean": sum(multiclass_miou_values) / len(multiclass_miou_values),
    }
