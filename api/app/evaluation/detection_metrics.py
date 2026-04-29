from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .iou import iou_xyxy

BBox = tuple[float, float, float, float]
BACKGROUND_LABEL = "__background__"


@dataclass(frozen=True)
class GroundTruthBox:
    image_id: str
    label: str
    bbox_xyxy: BBox


@dataclass(frozen=True)
class PredictedBox:
    image_id: str
    label: str
    bbox_xyxy: BBox
    score: float


def _validate_box(box: BBox) -> None:
    if len(box) != 4:
        raise ValueError("Bounding box must have 4 values.")
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid box coordinates {box}. Expected x2 > x1 and y2 > y1."
        )


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _collect_labels(
    predictions: list[PredictedBox], ground_truths: list[GroundTruthBox]
) -> list[str]:
    labels = {pred.label for pred in predictions}
    labels.update(gt.label for gt in ground_truths)
    return sorted(labels)


def _match_detections(
    predictions: list[PredictedBox],
    ground_truths: list[GroundTruthBox],
    *,
    iou_threshold: float,
    class_aware: bool,
) -> tuple[int, int, int]:
    preds = sorted(predictions, key=lambda item: item.score, reverse=True)
    gt_by_image: dict[str, list[GroundTruthBox]] = {}
    for gt in ground_truths:
        gt_by_image.setdefault(gt.image_id, []).append(gt)

    matched_gt: set[tuple[str, int]] = set()
    tp = 0
    fp = 0

    for pred in preds:
        candidates = gt_by_image.get(pred.image_id, [])
        best_idx = -1
        best_iou = -1.0

        for idx, gt in enumerate(candidates):
            gt_key = (pred.image_id, idx)
            if gt_key in matched_gt:
                continue
            if class_aware and gt.label != pred.label:
                continue
            overlap = iou_xyxy(pred.bbox_xyxy, gt.bbox_xyxy)
            if overlap > best_iou:
                best_iou = overlap
                best_idx = idx

        if best_idx >= 0 and best_iou >= iou_threshold:
            matched_gt.add((pred.image_id, best_idx))
            tp += 1
        else:
            fp += 1

    fn = len(ground_truths) - tp
    return tp, fp, fn


def precision_recall_f1_at_iou(
    predictions: list[PredictedBox],
    ground_truths: list[GroundTruthBox],
    *,
    iou_threshold: float = 0.5,
    class_aware: bool = True,
) -> dict[str, float | int]:
    if not (0.0 <= iou_threshold <= 1.0):
        raise ValueError("iou_threshold must be in [0, 1].")

    for gt in ground_truths:
        _validate_box(gt.bbox_xyxy)
    for pred in predictions:
        _validate_box(pred.bbox_xyxy)
        if not (0.0 <= pred.score <= 1.0):
            raise ValueError(f"Prediction score must be in [0, 1], got {pred.score}.")

    tp, fp, fn = _match_detections(
        predictions,
        ground_truths,
        iou_threshold=iou_threshold,
        class_aware=class_aware,
    )
    precision, recall, f1 = _precision_recall_f1(tp, fp, fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def per_class_metrics(
    predictions: list[PredictedBox],
    ground_truths: list[GroundTruthBox],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    labels = _collect_labels(predictions, ground_truths)
    for label in labels:
        label_predictions = [pred for pred in predictions if pred.label == label]
        label_ground_truths = [gt for gt in ground_truths if gt.label == label]
        output[label] = precision_recall_f1_at_iou(
            label_predictions,
            label_ground_truths,
            iou_threshold=iou_threshold,
            class_aware=True,
        )
    return output


def confusion_matrix(
    predictions: list[PredictedBox],
    ground_truths: list[GroundTruthBox],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, dict[str, int]]:
    labels = _collect_labels(predictions, ground_truths)
    labels_with_background = labels + [BACKGROUND_LABEL]

    matrix: dict[str, dict[str, int]] = {
        gt_label: {pred_label: 0 for pred_label in labels_with_background}
        for gt_label in labels_with_background
    }

    preds_by_image: dict[str, list[PredictedBox]] = {}
    gts_by_image: dict[str, list[GroundTruthBox]] = {}
    for pred in predictions:
        preds_by_image.setdefault(pred.image_id, []).append(pred)
    for gt in ground_truths:
        gts_by_image.setdefault(gt.image_id, []).append(gt)

    image_ids = set(preds_by_image.keys()) | set(gts_by_image.keys())
    for image_id in image_ids:
        image_preds = sorted(
            preds_by_image.get(image_id, []), key=lambda item: item.score, reverse=True
        )
        image_gts = gts_by_image.get(image_id, [])

        matched_gt_indices: set[int] = set()
        matched_pred_indices: set[int] = set()

        for pred_index, pred in enumerate(image_preds):
            best_gt_index = -1
            best_iou = -1.0
            for gt_index, gt in enumerate(image_gts):
                if gt_index in matched_gt_indices:
                    continue
                overlap = iou_xyxy(pred.bbox_xyxy, gt.bbox_xyxy)
                if overlap > best_iou:
                    best_iou = overlap
                    best_gt_index = gt_index

            if best_gt_index >= 0 and best_iou >= iou_threshold:
                gt = image_gts[best_gt_index]
                matrix[gt.label][pred.label] += 1
                matched_gt_indices.add(best_gt_index)
                matched_pred_indices.add(pred_index)

        for gt_index, gt in enumerate(image_gts):
            if gt_index not in matched_gt_indices:
                matrix[gt.label][BACKGROUND_LABEL] += 1

        for pred_index, pred in enumerate(image_preds):
            if pred_index not in matched_pred_indices:
                matrix[BACKGROUND_LABEL][pred.label] += 1

    return matrix


def _ap_for_label_at_iou(
    label: str,
    predictions: list[PredictedBox],
    ground_truths: list[GroundTruthBox],
    *,
    iou_threshold: float,
) -> float:
    label_preds = [pred for pred in predictions if pred.label == label]
    label_gts = [gt for gt in ground_truths if gt.label == label]
    if not label_gts:
        return 0.0

    label_preds = sorted(label_preds, key=lambda item: item.score, reverse=True)
    gts_by_image: dict[str, list[GroundTruthBox]] = {}
    for gt in label_gts:
        gts_by_image.setdefault(gt.image_id, []).append(gt)

    matched_gt: set[tuple[str, int]] = set()
    tp_flags: list[int] = []
    fp_flags: list[int] = []

    for pred in label_preds:
        candidates = gts_by_image.get(pred.image_id, [])
        best_index = -1
        best_iou = -1.0
        for idx, gt in enumerate(candidates):
            key = (pred.image_id, idx)
            if key in matched_gt:
                continue
            overlap = iou_xyxy(pred.bbox_xyxy, gt.bbox_xyxy)
            if overlap > best_iou:
                best_iou = overlap
                best_index = idx

        if best_index >= 0 and best_iou >= iou_threshold:
            matched_gt.add((pred.image_id, best_index))
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    if not tp_flags:
        return 0.0

    precisions: list[float] = []
    recalls: list[float] = []
    cum_tp = 0
    cum_fp = 0
    total_gt = len(label_gts)
    for tp_flag, fp_flag in zip(tp_flags, fp_flags):
        cum_tp += tp_flag
        cum_fp += fp_flag
        precision = cum_tp / (cum_tp + cum_fp) if (cum_tp + cum_fp) else 0.0
        recall = cum_tp / total_gt if total_gt else 0.0
        precisions.append(precision)
        recalls.append(recall)

    # COCO-style monotonically-decreasing precision envelope.
    for idx in range(len(precisions) - 2, -1, -1):
        precisions[idx] = max(precisions[idx], precisions[idx + 1])

    ap = 0.0
    previous_recall = 0.0
    for precision, recall in zip(precisions, recalls):
        delta = max(0.0, recall - previous_recall)
        ap += precision * delta
        previous_recall = recall
    return ap


def mean_average_precision(
    predictions: list[PredictedBox],
    ground_truths: list[GroundTruthBox],
    *,
    iou_thresholds: list[float],
) -> dict[str, Any]:
    if not iou_thresholds:
        raise ValueError("iou_thresholds must not be empty.")
    for threshold in iou_thresholds:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("All iou_thresholds must be in [0, 1].")

    labels = _collect_labels(predictions, ground_truths)
    labels = [label for label in labels if any(gt.label == label for gt in ground_truths)]
    if not labels:
        return {"map": 0.0, "per_threshold": {str(t): 0.0 for t in iou_thresholds}}

    per_threshold: dict[str, float] = {}
    threshold_maps: list[float] = []
    for threshold in iou_thresholds:
        per_label_ap = [
            _ap_for_label_at_iou(
                label,
                predictions,
                ground_truths,
                iou_threshold=threshold,
            )
            for label in labels
        ]
        threshold_map = sum(per_label_ap) / len(per_label_ap) if per_label_ap else 0.0
        per_threshold[f"{threshold:.2f}"] = threshold_map
        threshold_maps.append(threshold_map)

    return {
        "map": sum(threshold_maps) / len(threshold_maps) if threshold_maps else 0.0,
        "per_threshold": per_threshold,
    }


def evaluate_detection_suite(
    predictions: list[PredictedBox],
    ground_truths: list[GroundTruthBox],
) -> dict[str, Any]:
    primary = precision_recall_f1_at_iou(
        predictions, ground_truths, iou_threshold=0.5, class_aware=True
    )
    per_class = per_class_metrics(predictions, ground_truths, iou_threshold=0.5)
    conf = confusion_matrix(predictions, ground_truths, iou_threshold=0.5)
    map50 = mean_average_precision(predictions, ground_truths, iou_thresholds=[0.5])
    map5095 = mean_average_precision(
        predictions,
        ground_truths,
        iou_thresholds=[round(value, 2) for value in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]],
    )
    return {
        "primary_at_iou_0_5": primary,
        "per_class_at_iou_0_5": per_class,
        "confusion_matrix_at_iou_0_5": conf,
        "map_0_5": map50["map"],
        "map_0_5_to_0_95": map5095["map"],
        "map_threshold_breakdown": map5095["per_threshold"],
    }
