from __future__ import annotations

import unittest

from app.evaluation.iou import Detection, evaluate_detections, iou_xywh, iou_xyxy


class IoUTestCase(unittest.TestCase):
    def test_iou_identical_boxes_is_one(self) -> None:
        box = (0.0, 0.0, 2.0, 2.0)
        self.assertEqual(iou_xyxy(box, box), 1.0)

    def test_iou_non_overlapping_boxes_is_zero(self) -> None:
        box_a = (0.0, 0.0, 1.0, 1.0)
        box_b = (2.0, 2.0, 3.0, 3.0)
        self.assertEqual(iou_xyxy(box_a, box_b), 0.0)

    def test_iou_partial_overlap(self) -> None:
        # Intersection area = 1, union area = 7, IoU = 1/7.
        box_a = (0.0, 0.0, 2.0, 2.0)
        box_b = (1.0, 1.0, 3.0, 3.0)
        self.assertAlmostEqual(iou_xyxy(box_a, box_b), 1.0 / 7.0, places=6)

    def test_iou_xywh_matches_xyxy_for_same_boxes(self) -> None:
        box_xywh_a = (0.0, 0.0, 2.0, 2.0)
        box_xywh_b = (1.0, 1.0, 2.0, 2.0)
        self.assertAlmostEqual(iou_xywh(box_xywh_a, box_xywh_b), 1.0 / 7.0, places=6)

    def test_evaluate_detections_class_aware(self) -> None:
        predictions = [
            Detection("person", (0.0, 0.0, 2.0, 2.0)),
            Detection("cat", (3.0, 3.0, 5.0, 5.0)),
        ]
        ground_truths = [
            Detection("person", (0.0, 0.0, 2.0, 2.0)),
            Detection("dog", (3.0, 3.0, 5.0, 5.0)),
        ]

        metrics = evaluate_detections(
            predictions, ground_truths, iou_threshold=0.5, class_aware=True
        )

        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertAlmostEqual(float(metrics["precision"]), 0.5, places=6)
        self.assertAlmostEqual(float(metrics["recall"]), 0.5, places=6)
        self.assertAlmostEqual(float(metrics["f1"]), 0.5, places=6)
        self.assertAlmostEqual(float(metrics["mean_iou"]), 1.0, places=6)

    def test_evaluate_detections_non_class_aware(self) -> None:
        predictions = [
            Detection("person", (0.0, 0.0, 2.0, 2.0)),
            Detection("cat", (3.0, 3.0, 5.0, 5.0)),
        ]
        ground_truths = [
            Detection("person", (0.0, 0.0, 2.0, 2.0)),
            Detection("dog", (3.0, 3.0, 5.0, 5.0)),
        ]

        metrics = evaluate_detections(
            predictions, ground_truths, iou_threshold=0.5, class_aware=False
        )

        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertAlmostEqual(float(metrics["precision"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["recall"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["f1"]), 1.0, places=6)
        self.assertAlmostEqual(float(metrics["mean_iou"]), 1.0, places=6)

    def test_iou_threshold_blocks_weak_match(self) -> None:
        predictions = [Detection("person", (0.0, 0.0, 2.0, 2.0))]
        ground_truths = [Detection("person", (1.2, 1.2, 3.2, 3.2))]

        metrics = evaluate_detections(
            predictions, ground_truths, iou_threshold=0.5, class_aware=True
        )
        self.assertEqual(metrics["tp"], 0)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(float(metrics["mean_iou"]), 0.0)

    def test_invalid_threshold_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_detections([], [], iou_threshold=1.2)

    def test_invalid_box_raises(self) -> None:
        predictions = [Detection("person", (0.0, 0.0, 0.0, 1.0))]
        with self.assertRaises(ValueError):
            evaluate_detections(predictions, [])


if __name__ == "__main__":
    unittest.main()
