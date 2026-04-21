from __future__ import annotations

import unittest

from app.evaluation.detection_metrics import (
    BACKGROUND_LABEL,
    GroundTruthBox,
    PredictedBox,
    confusion_matrix,
    evaluate_detection_suite,
    mean_average_precision,
    precision_recall_f1_at_iou,
)


class DetectionMetricsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions = [
            PredictedBox("img1", "person", (0, 0, 2, 2), 0.95),
            PredictedBox("img1", "cat", (3, 3, 5, 5), 0.90),
            PredictedBox("img2", "person", (0, 0, 1, 1), 0.40),
        ]
        self.ground_truths = [
            GroundTruthBox("img1", "person", (0, 0, 2, 2)),
            GroundTruthBox("img1", "dog", (3, 3, 5, 5)),
            GroundTruthBox("img2", "person", (0, 0, 1, 1)),
        ]

    def test_precision_recall_f1_class_aware(self) -> None:
        metrics = precision_recall_f1_at_iou(
            self.predictions, self.ground_truths, iou_threshold=0.5, class_aware=True
        )
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertAlmostEqual(float(metrics["precision"]), 2 / 3, places=6)
        self.assertAlmostEqual(float(metrics["recall"]), 2 / 3, places=6)

    def test_precision_recall_f1_class_agnostic(self) -> None:
        metrics = precision_recall_f1_at_iou(
            self.predictions, self.ground_truths, iou_threshold=0.5, class_aware=False
        )
        self.assertEqual(metrics["tp"], 3)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)

    def test_confusion_matrix_has_background_counts(self) -> None:
        matrix = confusion_matrix(self.predictions, self.ground_truths, iou_threshold=0.5)
        self.assertIn(BACKGROUND_LABEL, matrix)
        self.assertEqual(matrix["dog"]["cat"], 1)
        self.assertEqual(matrix[BACKGROUND_LABEL][BACKGROUND_LABEL], 0)

    def test_map_reports(self) -> None:
        report = mean_average_precision(
            self.predictions,
            self.ground_truths,
            iou_thresholds=[0.5, 0.75],
        )
        self.assertIn("map", report)
        self.assertIn("0.50", report["per_threshold"])
        self.assertIn("0.75", report["per_threshold"])

    def test_suite_contains_required_fields(self) -> None:
        suite = evaluate_detection_suite(self.predictions, self.ground_truths)
        self.assertIn("map_0_5", suite)
        self.assertIn("map_0_5_to_0_95", suite)
        self.assertIn("per_class_at_iou_0_5", suite)
        self.assertIn("confusion_matrix_at_iou_0_5", suite)


if __name__ == "__main__":
    unittest.main()
