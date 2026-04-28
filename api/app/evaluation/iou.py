from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    label: str
    bbox_xyxy: BBox


def _validate_xyxy(box: BBox) -> None:
    if len(box) != 4:
        raise ValueError("Bounding box must have 4 values: (x1, y1, x2, y2).")

    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid box coordinates {box}. Expected x2 > x1 and y2 > y1."
        )


def xywh_to_xyxy(box_xywh: BBox) -> BBox:
    if len(box_xywh) != 4:
        raise ValueError("Bounding box must have 4 values: (x, y, w, h).")

    x, y, w, h = box_xywh
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid box dimensions {box_xywh}. Expected w > 0 and h > 0.")

    return (x, y, x + w, y + h)


def iou_xyxy(box_a: BBox, box_b: BBox) -> float:
    _validate_xyxy(box_a)
    _validate_xyxy(box_b)

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0

    return inter_area / union_area


def iou_xywh(box_a_xywh: BBox, box_b_xywh: BBox) -> float:
    box_a = xywh_to_xyxy(box_a_xywh)
    box_b = xywh_to_xyxy(box_b_xywh)
    return iou_xyxy(box_a, box_b)


def evaluate_detections(
    predictions: Sequence[Detection],
    ground_truths: Sequence[Detection],
    *,
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> dict[str, float | int]:
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError("iou_threshold must be between 0 and 1.")

    for pred in predictions:
        _validate_xyxy(pred.bbox_xyxy)
    for gt in ground_truths:
        _validate_xyxy(gt.bbox_xyxy)

    candidates: list[tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(predictions):
        for gt_idx, gt in enumerate(ground_truths):
            if class_aware and pred.label != gt.label:
                continue
            iou = iou_xyxy(pred.bbox_xyxy, gt.bbox_xyxy)
            candidates.append((iou, pred_idx, gt_idx))

    candidates.sort(key=lambda item: item[0], reverse=True)

    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    matched_ious: list[float] = []

    for iou, pred_idx, gt_idx in candidates:
        if iou < iou_threshold:
            break
        if pred_idx in matched_preds or gt_idx in matched_gts:
            continue
        matched_preds.add(pred_idx)
        matched_gts.add(gt_idx)
        matched_ious.append(iou)

    true_positives = len(matched_ious)
    false_positives = len(predictions) - true_positives
    false_negatives = len(ground_truths) - true_positives

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
    }
