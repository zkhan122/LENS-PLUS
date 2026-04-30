import numpy as np
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def compute_standard_metrics(predicted: np.ndarray, ground_truth: np.ndarray) -> dict:
    valid = ground_truth > 0
    if valid.sum() == 0:
        return {
            "abs_rel": None,
            "rmse": None,
            "delta_1_25": None,
            "delta_1_25_sq": None,
            "delta_1_25_cu": None,
            "scale_factor_used": None,
            "error": "no valid ground truth pixels",
        }
    pred = predicted[valid]
    gt = ground_truth[valid]
    scale = np.median(gt) / (np.median(pred) + 1e-6)
    pred_metric = pred * scale
    abs_rel = float(np.mean(np.abs(pred_metric - gt) / gt))
    rmse = float(np.sqrt(np.mean((pred_metric - gt) ** 2)))
    ratio = np.maximum(pred_metric / gt, gt / pred_metric)
    delta_1 = float(np.mean(ratio < 1.25))
    delta_2 = float(np.mean(ratio < 1.25 ** 2))
    delta_3 = float(np.mean(ratio < 1.25 ** 3))
    return {
        "abs_rel": round(abs_rel, 4),
        "rmse": round(rmse, 4),
        "delta_1_25": round(delta_1, 4),
        "delta_1_25_sq": round(delta_2, 4),
        "delta_1_25_cu": round(delta_3, 4),
        "scale_factor_used": round(float(scale), 4),
    }


if __name__ == "__main__":
    test_results = []

    print("=" * 50)
    print("TEST 1 — Perfect prediction (should score 0 error)")
    gt_perfect = np.array([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [0.0, 7.0, 8.0]])
    pred_perfect = gt_perfect.copy()
    result = compute_standard_metrics(pred_perfect, gt_perfect)
    print(json.dumps(result, indent=2))
    assert result["abs_rel"] == 0.0,    "AbsRel should be 0 for perfect prediction"
    assert result["rmse"] == 0.0,       "RMSE should be 0 for perfect prediction"
    assert result["delta_1_25"] == 1.0, "δ<1.25 should be 1.0 for perfect prediction"
    print("✓ PASSED\n")
    test_results.append({"test": "perfect_prediction", "passed": True, "metrics": result})

    print("=" * 50)
    print("TEST 2 — Scaled prediction (relative depth, should still score well)")
    gt_scaled = np.array([[2.0, 4.0, 6.0],
                          [8.0, 10.0, 12.0]])
    pred_scaled = gt_scaled / 3.5
    result = compute_standard_metrics(pred_scaled, gt_scaled)
    print(json.dumps(result, indent=2))
    assert result["abs_rel"] < 0.01,    "AbsRel should be near 0 after scale alignment"
    assert result["delta_1_25"] == 1.0, "δ<1.25 should be 1.0 after scale alignment"
    print("✓ PASSED\n")
    test_results.append({"test": "scaled_prediction", "passed": True, "metrics": result})

    print("=" * 50)
    print("TEST 3 — Noisy prediction (expect moderate error)")
    np.random.seed(42)
    gt_noisy = np.random.uniform(1.0, 5.0, (480, 640)).astype(np.float32)
    noise = np.random.normal(0, 0.3, gt_noisy.shape).astype(np.float32)
    pred_noisy = (gt_noisy + noise) / 2.0
    result = compute_standard_metrics(pred_noisy, gt_noisy)
    print(json.dumps(result, indent=2))
    assert result["abs_rel"] > 0.0,    "AbsRel should be non-zero for noisy prediction"
    assert result["delta_1_25"] < 1.0, "δ<1.25 should be less than 1.0 for noisy prediction"
    print("✓ PASSED\n")
    test_results.append({"test": "noisy_prediction", "passed": True, "metrics": result})

    print("=" * 50)
    print("TEST 4 — All invalid pixels (should handle gracefully)")
    gt_invalid = np.zeros((100, 100))
    pred_invalid = np.ones((100, 100))
    result = compute_standard_metrics(pred_invalid, gt_invalid)
    print(json.dumps(result, indent=2))
    assert "error" in result,          "Should return error key for all-invalid input"
    assert result["abs_rel"] is None,  "Metrics should be None for all-invalid input"
    print("✓ PASSED\n")
    test_results.append({"test": "all_invalid_pixels", "passed": True, "metrics": result})

    output = {
        "total_tests": len(test_results),
        "all_passed": all(t["passed"] for t in test_results),
        "results": test_results,
    }

    output_path = os.path.join(BASE_DIR, "synthetic_test_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results written to {output_path}")