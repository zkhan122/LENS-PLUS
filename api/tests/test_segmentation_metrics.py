from __future__ import annotations

import unittest

from app.evaluation.segmentation_metrics import (
    SegmentationSample,
    dice_score,
    evaluate_segmentation_suite,
    mask_iou,
    mean_iou_multiclass,
    pixel_accuracy,
)


class SegmentationMetricsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.pred = [
            [0, 1, 1],
            [0, 1, 0],
            [0, 0, 0],
        ]
        self.gt = [
            [0, 1, 1],
            [0, 1, 1],
            [0, 0, 0],
        ]

    def test_dice_score(self) -> None:
        self.assertAlmostEqual(dice_score(self.pred, self.gt), 6 / 7, places=6)

    def test_pixel_accuracy(self) -> None:
        self.assertAlmostEqual(pixel_accuracy(self.pred, self.gt), 8 / 9, places=6)

    def test_mask_iou(self) -> None:
        self.assertAlmostEqual(mask_iou(self.pred, self.gt), 3 / 4, places=6)

    def test_multiclass_miou(self) -> None:
        pred = [[0, 1], [1, 2]]
        gt = [[0, 1], [2, 2]]
        value = mean_iou_multiclass(pred, gt)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_suite(self) -> None:
        samples = [
            SegmentationSample("a", self.pred, self.gt),
            SegmentationSample("b", self.gt, self.gt),
        ]
        report = evaluate_segmentation_suite(samples)
        self.assertEqual(report["num_samples"], 2)
        self.assertIn("dice_mean", report)
        self.assertIn("multiclass_miou_mean", report)

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            dice_score([[1]], [[1, 1]])


if __name__ == "__main__":
    unittest.main()
