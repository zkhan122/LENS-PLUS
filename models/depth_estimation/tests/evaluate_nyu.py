import os
import sys
import json
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEPTH_ESTIMATION_DIR = os.path.dirname(BASE_DIR)
DEPTH_ANYTHING_PATH = os.path.join(DEPTH_ESTIMATION_DIR, "Depth-Anything-V2")

os.environ["XFORMERS_DISABLED"] = "1"
sys.modules["xformers"] = None
sys.modules["xformers.ops"] = None

sys.path.insert(0, DEPTH_ANYTHING_PATH)
sys.path.insert(0, DEPTH_ESTIMATION_DIR)

import h5py
import numpy as np
from depth_estimator import DepthEstimator

PAPER_REFERENCE = {
    "abs_rel": 0.053,
    "delta_1_25": 0.973,
    "rmse": 0.261,
}

def compute_standard_metrics(predicted: np.ndarray, ground_truth: np.ndarray) -> dict:
    valid = ground_truth > 0

    if valid.sum() == 0:
        return {
            "abs_rel": None, "rmse": None, "delta_1_25": None,
            "error": "no valid ground truth pixels",
        }

    pred = predicted[valid]
    gt = ground_truth[valid]

    pred = np.clip(pred, 1e-6, None)

    abs_rel = float(np.mean(np.abs(pred - gt) / gt))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    ratio = np.maximum(pred / gt, gt / pred)
    
    delta_1 = float(np.mean(ratio < 1.25))
    delta_2 = float(np.mean(ratio < 1.25 ** 2))
    delta_3 = float(np.mean(ratio < 1.25 ** 3))

    return {
        "abs_rel": round(abs_rel, 4),
        "rmse": round(rmse, 4),
        "delta_1_25": round(delta_1, 4),
        "delta_1_25_sq": round(delta_2, 4),
        "delta_1_25_cu": round(delta_3, 4),
    }

def evaluate_on_nyu(
    mat_path: str,
    checkpoint_path: str,
    num_samples: int = 500,
    output_json: str = None,
):
    if output_json is None:
        output_json = os.path.join(
            DEPTH_ESTIMATION_DIR, "output", "nyu_eval_results.json"
        )

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    print(f"Loading DepthEstimator from {checkpoint_path}...")
    estimator = DepthEstimator(
        checkpoint_path=checkpoint_path,
        frames_root="",
    )
    print(f"Device: {estimator.device}")

    print(f"Loading NYU Depth V2 from {mat_path}...")
    with h5py.File(mat_path, "r") as f:
        images = np.array(f["images"])
        depths = np.array(f["depths"])

    total_available = images.shape[0]
    total = min(num_samples, total_available)
    print(f"Dataset size: {total_available} frames")
    print(f"Evaluating on: {total} samples\n")

    estimator.prev_depth = None

    per_sample = []
    skipped = 0

    for i in range(total):
        img_rgb = images[i].transpose(1, 2, 0).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gt_depth = depths[i].astype(np.float32)

        if gt_depth.max() > 20.0:
            gt_depth = gt_depth / 1000.0

        result = estimator.predict(img_bgr)
        pred_depth = result["raw_depth"]
        proximity_status = result["proximity_status"]

        if i < 5:
            h, w = img_bgr.shape[:2]
            pred_viz_resized = cv2.resize(pred_depth, (w, h))
            gt_viz_resized = cv2.resize(gt_depth, (w, h))
            
            viz_gt = cv2.normalize(gt_viz_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            viz_pred = cv2.normalize(pred_viz_resized, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            viz_gt = cv2.applyColorMap(viz_gt, cv2.COLORMAP_TURBO)
            viz_pred = cv2.applyColorMap(viz_pred, cv2.COLORMAP_TURBO)
            
            combined = np.hstack((img_bgr, viz_gt, viz_pred))
            cv2.imwrite(os.path.join(DEPTH_ESTIMATION_DIR, "output", f"nyu_eval_viz_{i+1}.jpg"), combined)

        metrics = compute_standard_metrics(pred_depth, gt_depth)

        if "error" in metrics:
            print(f"  [{i+1}/{total}] SKIPPED — {metrics['error']}")
            skipped += 1
            continue

        per_sample.append({
            "sample_index": i,
            "proximity_status": proximity_status,
            **metrics,
        })

        print(
            f"  [{i+1}/{total}]  "
            f"AbsRel={metrics['abs_rel']:.4f}  "
            f"RMSE={metrics['rmse']:.4f}  "
            f"δ<1.25={metrics['delta_1_25']:.4f}  "
            f"status={proximity_status}"
        )

    if not per_sample:
        print("No valid samples to summarise.")
        return

    mean_abs_rel = round(float(np.mean([m["abs_rel"]       for m in per_sample])), 4)
    mean_rmse    = round(float(np.mean([m["rmse"]          for m in per_sample])), 4)
    mean_delta_1 = round(float(np.mean([m["delta_1_25"]    for m in per_sample])), 4)
    mean_delta_2 = round(float(np.mean([m["delta_1_25_sq"] for m in per_sample])), 4)
    mean_delta_3 = round(float(np.mean([m["delta_1_25_cu"] for m in per_sample])), 4)

    all_statuses = [m["proximity_status"] for m in per_sample]
    status_distribution = {
        s: all_statuses.count(s)
        for s in ["CLEAR", "FAR", "MID", "CLOSE", "VERY_CLOSE"]
    }

    comparison = {
        "abs_rel": {
            "ours": mean_abs_rel,
            "paper_vits": PAPER_REFERENCE["abs_rel"],
            "difference": round(mean_abs_rel - PAPER_REFERENCE["abs_rel"], 4),
            "note": "lower is better",
        },
        "rmse": {
            "ours": mean_rmse,
            "paper_vits": PAPER_REFERENCE["rmse"],
            "difference": round(mean_rmse - PAPER_REFERENCE["rmse"], 4),
            "note": "lower is better",
        },
        "delta_1_25": {
            "ours": mean_delta_1,
            "paper_vits": PAPER_REFERENCE["delta_1_25"],
            "difference": round(mean_delta_1 - PAPER_REFERENCE["delta_1_25"], 4),
            "note": "higher is better",
        },
    }

    output = {
        "dataset": "NYU Depth V2",
        "model": "DepthEstimator (Depth-Anything-V2 ViT-S + temporal smoothing + zone analysis)",
        "checkpoint": os.path.basename(checkpoint_path),
        "device": estimator.device,
        "num_samples_requested": num_samples,
        "num_samples_evaluated": len(per_sample),
        "num_samples_skipped": skipped,
        "summary": {
            "mean_abs_rel":       mean_abs_rel,
            "mean_rmse":          mean_rmse,
            "mean_delta_1_25":    mean_delta_1,
            "mean_delta_1_25_sq": mean_delta_2,
            "mean_delta_1_25_cu": mean_delta_3,
            "status_distribution": status_distribution,
        },
        "comparison_vs_paper": comparison,
        "per_sample": per_sample,
    }

    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Samples evaluated : {len(per_sample)} / {total}")
    print(f"  Samples skipped   : {skipped}")
    print()
    print(f"  {'Metric':<20} {'Ours':>10} {'Paper ViT-S':>12} {'Diff':>10}")
    print(f"  {'-'*54}")
    print(f"  {'AbsRel ↓':<20} {mean_abs_rel:>10.4f} {PAPER_REFERENCE['abs_rel']:>12.4f} {mean_abs_rel - PAPER_REFERENCE['abs_rel']:>+10.4f}")
    print(f"  {'RMSE ↓':<20} {mean_rmse:>10.4f} {PAPER_REFERENCE['rmse']:>12.4f} {mean_rmse - PAPER_REFERENCE['rmse']:>+10.4f}")
    print(f"  {'δ<1.25 ↑':<20} {mean_delta_1:>10.4f} {PAPER_REFERENCE['delta_1_25']:>12.4f} {mean_delta_1 - PAPER_REFERENCE['delta_1_25']:>+10.4f}")
    print()
    print(f"  Status distribution across {len(per_sample)} frames:")
    for status, count in status_distribution.items():
        print(f"    {status:<12} {count:>4} frames")
    print(f"\n  Results saved to: {output_json}")


if __name__ == "__main__":
    evaluate_on_nyu(
        mat_path=os.path.join(
            DEPTH_ESTIMATION_DIR, "data", "nyu_depth_v2_labeled.mat"
        ),
        checkpoint_path=os.path.join(
            DEPTH_ESTIMATION_DIR, "checkpoints", "depth_anything_v2_metric_hypersim_vits.pth"
        ),
        num_samples=50,
    )