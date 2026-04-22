#!/usr/bin/env python3
"""Evaluation and comparison of geometric vs. YOLO parking detection.

Produces:
- Confusion matrix for the geometric pipeline (vacant vs. occupied)
- Confusion matrix for the YOLO pipeline
- Side-by-side accuracy, precision, recall, F1 comparison
- Per-image result logs
- Summary plots saved to output/evaluation/

This script uses YOLO label files as ground truth (since the PKLot/CNRPark
datasets provide occupancy labels that Tim's pipeline converts to YOLO format).
The geometric pipeline's predictions are compared against these labels.

Usage:
    # evaluate geometric pipeline against YOLO ground truth labels
    python evaluate.py geometric --images data/pklot/yolo_ready/images/test \
                                 --labels data/pklot/yolo_ready/labels/test

    # evaluate YOLO predictions against ground truth
    python evaluate.py yolo --images data/pklot/yolo_ready/images/test \
                            --labels data/pklot/yolo_ready/labels/test \
                            --weights runs/detect/runs/yopo/train/weights/best.pt

    # compare both pipelines side by side
    python evaluate.py compare --images data/pklot/yolo_ready/images/test \
                               --labels data/pklot/yolo_ready/labels/test \
                               --weights runs/detect/runs/yopo/train/weights/best.pt

Authors: Sreenath Prasadh, Timothy Bennett
Course: CS 5330 - Pattern Recognition and Computer Vision
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Ground truth parsing

def load_ground_truth(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> List[Dict]:
    """Load YOLO format ground truth labels.

    Returns list of dicts with keys: class_id, class_name, bbox (x, y, w, h in pixels).
    """
    entries = []
    if not label_path.exists():
        return entries

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx = float(parts[1]) * image_width
            cy = float(parts[2]) * image_height
            w = float(parts[3]) * image_width
            h = float(parts[4]) * image_height
            entries.append({
                "class_id": cls,
                "class_name": config.CLASS_NAMES[cls] if cls < len(config.CLASS_NAMES) else "unknown",
                "bbox": (int(cx - w / 2), int(cy - h / 2), int(w), int(h)),
                "center": (cx, cy),
            })
    return entries


def compute_iou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
) -> float:
    """IoU between two (x, y, w, h) boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


# Confusion matrix

class ConfusionMatrix:
    """2x2 confusion matrix for binary classification (vacant vs. occupied)."""

    def __init__(self):
        # rows = true label, cols = predicted label
        # index 0 = spot (vacant), index 1 = car (occupied)
        self.matrix = np.zeros((2, 2), dtype=int)

    def add(self, true_class: int, pred_class: int) -> None:
        """Add a single prediction. class 0 = spot, class 1 = car."""
        if 0 <= true_class <= 1 and 0 <= pred_class <= 1:
            self.matrix[true_class, pred_class] += 1

    @property
    def total(self) -> int:
        return int(self.matrix.sum())

    @property
    def accuracy(self) -> float:
        return float(np.trace(self.matrix)) / self.total if self.total > 0 else 0.0

    @property
    def precision_per_class(self) -> Tuple[float, float]:
        """(precision_spot, precision_car)"""
        p = []
        for c in range(2):
            col_sum = self.matrix[:, c].sum()
            p.append(float(self.matrix[c, c]) / col_sum if col_sum > 0 else 0.0)
        return (p[0], p[1])

    @property
    def recall_per_class(self) -> Tuple[float, float]:
        """(recall_spot, recall_car)"""
        r = []
        for c in range(2):
            row_sum = self.matrix[c, :].sum()
            r.append(float(self.matrix[c, c]) / row_sum if row_sum > 0 else 0.0)
        return (r[0], r[1])

    @property
    def f1_per_class(self) -> Tuple[float, float]:
        prec = self.precision_per_class
        rec = self.recall_per_class
        f1 = []
        for p, r in zip(prec, rec):
            f1.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
        return (f1[0], f1[1])

    def print_matrix(self, title: str = "Confusion Matrix") -> None:
        print(f"\n{title}")
        print(f"{'':>20} {'Pred Spot':>12} {'Pred Car':>12}")
        print(f"  {'True Spot':>16} {self.matrix[0, 0]:>12} {self.matrix[0, 1]:>12}")
        print(f"  {'True Car':>16} {self.matrix[1, 0]:>12} {self.matrix[1, 1]:>12}")

        prec = self.precision_per_class
        rec = self.recall_per_class
        f1 = self.f1_per_class

        print(f"\n  Accuracy:  {self.accuracy:.{config.METRIC_PRECISION}f}")
        print(f"  Precision: spot={prec[0]:.{config.METRIC_PRECISION}f}  car={prec[1]:.{config.METRIC_PRECISION}f}")
        print(f"  Recall:    spot={rec[0]:.{config.METRIC_PRECISION}f}  car={rec[1]:.{config.METRIC_PRECISION}f}")
        print(f"  F1:        spot={f1[0]:.{config.METRIC_PRECISION}f}  car={f1[1]:.{config.METRIC_PRECISION}f}")


# Geometric pipeline evaluation

def evaluate_geometric(
    images_dir: Path,
    labels_dir: Path,
    threshold_method: str = config.THRESHOLD_METHOD,
    max_images: Optional[int] = None,
) -> ConfusionMatrix:
    """Evaluate the geometric pipeline against ground truth labels.

    For each ground truth bounding box, we check whether the geometric
    pipeline's vacancy detection (based on percent fill within that
    region) agrees with the label.
    """
    from geometricParking import (
        preprocess_frame, extract_line_mask,
        morphological_cleanup, compute_spot_features,
        ParkingSpot,
    )

    cm = ConfusionMatrix()
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in extensions)

    if max_images:
        images = images[:max_images]

    print(f"Evaluating geometric pipeline on {len(images)} images...")

    for i, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        label_path = labels_dir / f"{img_path.stem}.txt"
        h, w = frame.shape[:2]
        gt_entries = load_ground_truth(label_path, w, h)

        if not gt_entries:
            continue

        # preprocess once per image
        _, gray, hsv = preprocess_frame(frame)

        # for each ground truth box, compute geometric features
        # and compare vacancy prediction to the label
        for entry in gt_entries:
            true_class = entry["class_id"]  # 0=spot, 1=car
            bx, by, bw, bh = entry["bbox"]

            # create a synthetic ParkingSpot from the GT bbox
            corners = np.array([
                [bx, by], [bx + bw, by],
                [bx + bw, by + bh], [bx, by + bh],
            ], dtype=np.float32)

            spot = ParkingSpot(
                spot_id=0, corners=corners,
                bbox=(bx, by, bw, bh),
            )

            feat = compute_spot_features(frame, gray, hsv, spot)

            # geometric prediction: vacant (0) or occupied (1)
            pred_class = 0 if feat.is_vacant else 1
            cm.add(true_class, pred_class)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(images)} (running acc: {cm.accuracy:.3f})")

    return cm


# YOLO pipeline evaluation

def evaluate_yolo(
    images_dir: Path,
    labels_dir: Path,
    weights: Path,
    max_images: Optional[int] = None,
) -> ConfusionMatrix:
    """Evaluate YOLO predictions against ground truth labels."""
    from geometricParking import load_yolo_predictions

    cm = ConfusionMatrix()
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in extensions)

    if max_images:
        images = images[:max_images]

    print(f"Evaluating YOLO pipeline on {len(images)} images...")

    for i, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        label_path = labels_dir / f"{img_path.stem}.txt"
        h, w = frame.shape[:2]
        gt_entries = load_ground_truth(label_path, w, h)

        if not gt_entries:
            continue

        # run YOLO inference
        yolo_spots, yolo_cars = load_yolo_predictions(
            weights, frame, conf=config.YOLO_CONFIDENCE
        )
        all_dets = [(d, 0) for d in yolo_spots] + [(d, 1) for d in yolo_cars]

        # match GT to predictions via IoU
        for entry in gt_entries:
            true_class = entry["class_id"]
            gt_bbox = entry["bbox"]

            best_iou = 0.0
            best_pred = -1
            for det, det_class in all_dets:
                iou = compute_iou(gt_bbox, det.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_pred = det_class

            if best_iou >= config.IOU_MATCH_THRESHOLD:
                cm.add(true_class, best_pred)
            else:
                # no match found: count as missed (predict opposite)
                cm.add(true_class, 1 - true_class)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(images)} (running acc: {cm.accuracy:.3f})")

    return cm


# Plotting

def plot_confusion_matrix(
    cm: ConfusionMatrix,
    title: str,
    save_path: Path,
) -> None:
    """Save a confusion matrix heatmap."""
    if not HAS_MATPLOTLIB:
        print(f"Skipping plot (matplotlib not available): {save_path}")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.matrix, cmap="Blues", interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Spot (Vacant)", "Car (Occupied)"])
    ax.set_yticklabels(["Spot (Vacant)", "Car (Occupied)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # annotate cells
    for i in range(2):
        for j in range(2):
            val = cm.matrix[i, j]
            color = "white" if val > cm.matrix.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot: {save_path}")


def plot_comparison(
    geo_cm: ConfusionMatrix,
    yolo_cm: ConfusionMatrix,
    save_path: Path,
) -> None:
    """Save a side-by-side bar chart comparing both pipelines."""
    if not HAS_MATPLOTLIB:
        print(f"Skipping plot (matplotlib not available): {save_path}")
        return

    metrics = ["Accuracy", "Precision\n(spot)", "Precision\n(car)",
               "Recall\n(spot)", "Recall\n(car)", "F1\n(spot)", "F1\n(car)"]

    geo_vals = [
        geo_cm.accuracy,
        *geo_cm.precision_per_class,
        *geo_cm.recall_per_class,
        *geo_cm.f1_per_class,
    ]
    yolo_vals = [
        yolo_cm.accuracy,
        *yolo_cm.precision_per_class,
        *yolo_cm.recall_per_class,
        *yolo_cm.f1_per_class,
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(x - width / 2, geo_vals, width, label="Geometric", color="#2ca02c")
    bars2 = ax.bar(x + width / 2, yolo_vals, width, label="YOLO", color="#1f77b4")

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Geometric vs. YOLO Pipeline Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    # value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0, height + 0.02,
                f"{height:.2f}", ha="center", va="bottom", fontsize=8,
            )

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot: {save_path}")


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate and compare parking detection pipelines"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--images", type=Path, required=True,
                        help="Directory of test images")
    common.add_argument("--labels", type=Path, required=True,
                        help="Directory of YOLO format ground truth labels")
    common.add_argument("--output-dir", type=Path,
                        default=config.OUTPUT_EVAL)
    common.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images (for quick testing)")

    # geometric only
    p_geo = sub.add_parser("geometric", parents=[common],
                           help="Evaluate geometric pipeline")
    p_geo.add_argument("--threshold", choices=["adaptive", "otsu", "isodata"],
                       default=config.THRESHOLD_METHOD)

    # YOLO only
    p_yolo = sub.add_parser("yolo", parents=[common],
                            help="Evaluate YOLO pipeline")
    p_yolo.add_argument("--weights", type=Path,
                        default=config.YOLO_WEIGHTS_PKLOT)

    # compare both
    p_cmp = sub.add_parser("compare", parents=[common],
                           help="Compare both pipelines side by side")
    p_cmp.add_argument("--weights", type=Path,
                       default=config.YOLO_WEIGHTS_PKLOT)
    p_cmp.add_argument("--threshold", choices=["adaptive", "otsu", "isodata"],
                       default=config.THRESHOLD_METHOD)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "geometric":
        cm = evaluate_geometric(
            args.images, args.labels,
            threshold_method=args.threshold,
            max_images=args.max_images,
        )
        cm.print_matrix("Geometric Pipeline Confusion Matrix")
        plot_confusion_matrix(cm, "Geometric Pipeline", args.output_dir / "cm_geometric.png")

    elif args.command == "yolo":
        cm = evaluate_yolo(
            args.images, args.labels, args.weights,
            max_images=args.max_images,
        )
        cm.print_matrix("YOLO Pipeline Confusion Matrix")
        plot_confusion_matrix(cm, "YOLO Pipeline", args.output_dir / "cm_yolo.png")

    elif args.command == "compare":
        print("=" * 50)
        print("GEOMETRIC PIPELINE")
        print("=" * 50)
        geo_cm = evaluate_geometric(
            args.images, args.labels,
            threshold_method=args.threshold,
            max_images=args.max_images,
        )
        geo_cm.print_matrix("Geometric Pipeline Confusion Matrix")

        print("\n" + "=" * 50)
        print("YOLO PIPELINE")
        print("=" * 50)
        yolo_cm = evaluate_yolo(
            args.images, args.labels, args.weights,
            max_images=args.max_images,
        )
        yolo_cm.print_matrix("YOLO Pipeline Confusion Matrix")

        # save plots
        plot_confusion_matrix(geo_cm, "Geometric Pipeline", args.output_dir / "cm_geometric.png")
        plot_confusion_matrix(yolo_cm, "YOLO Pipeline", args.output_dir / "cm_yolo.png")
        plot_comparison(geo_cm, yolo_cm, args.output_dir / "comparison.png")

        # summary
        print("\n" + "=" * 50)
        print("COMPARISON SUMMARY")
        print("=" * 50)
        print(f"{'Metric':<20} {'Geometric':>12} {'YOLO':>12}")
        print("-" * 45)
        print(f"{'Accuracy':<20} {geo_cm.accuracy:>12.4f} {yolo_cm.accuracy:>12.4f}")
        for label, idx in [("Precision (spot)", 0), ("Precision (car)", 1)]:
            print(f"{label:<20} {geo_cm.precision_per_class[idx]:>12.4f} {yolo_cm.precision_per_class[idx]:>12.4f}")
        for label, idx in [("Recall (spot)", 0), ("Recall (car)", 1)]:
            print(f"{label:<20} {geo_cm.recall_per_class[idx]:>12.4f} {yolo_cm.recall_per_class[idx]:>12.4f}")
        for label, idx in [("F1 (spot)", 0), ("F1 (car)", 1)]:
            print(f"{label:<20} {geo_cm.f1_per_class[idx]:>12.4f} {yolo_cm.f1_per_class[idx]:>12.4f}")


if __name__ == "__main__":
    main()