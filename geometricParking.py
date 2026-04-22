#!/usr/bin/env python3
"""Geometric Parking Analysis Module - PKLot Paper Approach.

Classical CV pipeline for parking spot occupancy classification. Uses
predefined parking space coordinates (from YOLO label files) and classifies
each spot as vacant or occupied using texture descriptors and geometric
features inspired by Project 3 and the original PKLot paper.

This mirrors how the PKLot paper works:
  - Spot locations are KNOWN (from annotations / YOLO labels)
  - For each spot, crop the image patch
  - Extract texture features (LBP histogram, intensity stats, saturation)
  - Classify vacant vs occupied using feature thresholds

Pipeline stages:
  1. Preprocessing (CLAHE, grayscale, HSV)
  2. Load spot coordinates from YOLO label files
  3. For each spot: crop patch, extract features
     a. LBP texture histogram (PKLot paper core feature)
     b. Intensity statistics (std, mean, edge density)
     c. Saturation statistics (color vs gray pavement)
     d. Hu moments (Project 3)
  4. Classify vacant vs occupied
  5. Suitability judgment (over-the-line check)
  6. Visualization

Authors: Sreenath Prasadh (geometric pipeline), Timothy Bennett (YOLO pipeline)
Course: CS 5330 - Pattern Recognition and Computer Vision
"""

from __future__ import annotations
import argparse, csv, math, os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ParkingLine:
    """Edge of a parking spot boundary (for visualization)."""
    x1: int; y1: int; x2: int; y2: int
    angle: float; length: float
    perp: float = 0.0; along: float = 0.0
    @property
    def midpoint(self): return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

@dataclass
class ParkingSpot:
    """A parking space with known coordinates."""
    spot_id: int; corners: np.ndarray; bbox: Tuple[int, int, int, int]
    left_line: Optional[ParkingLine] = None
    right_line: Optional[ParkingLine] = None
    dominant_angle: float = 0.0
    gt_class: int = -1  # ground truth: 0=spot(vacant), 1=car(occupied), -1=unknown
    @property
    def center(self):
        return (float(np.mean(self.corners[:, 0])), float(np.mean(self.corners[:, 1])))
    @property
    def area(self):
        return float(cv2.contourArea(self.corners.astype(np.float32)))

@dataclass
class SpotFeatures:
    """Classical CV features for a parking spot patch (Project 3 + PKLot paper)."""
    spot_id: int; percent_filled: float = 0.0; bbox_aspect_ratio: float = 0.0
    hu_moments: np.ndarray = field(default_factory=lambda: np.zeros(7))
    axis_angle: float = 0.0; centroid: Tuple[float, float] = (0.0, 0.0)
    texture_std: float = 0.0; mean_saturation: float = 0.0
    edge_density: float = 0.0; lbp_nonuniform: float = 0.0
    overlap_left: float = 0.0; overlap_right: float = 0.0
    is_vacant: bool = True; is_suitable: bool = True

@dataclass
class YOLODetection:
    class_id: int; class_name: str; confidence: float
    bbox: Tuple[int, int, int, int]; center: Tuple[float, float]


# ---------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    return blurred, gray, hsv


def _clahe_enhance(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ---------------------------------------------------------------------------
# 2. Load spot coordinates from YOLO labels
# ---------------------------------------------------------------------------

def _load_spots_from_labels(label_path, image_width, image_height):
    """Load parking spot coordinates from a YOLO label file.

    Each line: class_id cx cy nw nh
    class 0 = spot (vacant), class 1 = car (occupied)

    Returns list of ParkingSpot with ground truth class.
    """
    spots = []
    if not label_path.exists():
        return spots

    with open(label_path) as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx = float(parts[1]) * image_width
            cy = float(parts[2]) * image_height
            w = float(parts[3]) * image_width
            h = float(parts[4]) * image_height

            x = int(cx - w / 2)
            y = int(cy - h / 2)
            bw, bh = int(w), int(h)

            # clamp to image bounds
            x = max(0, min(x, image_width - 1))
            y = max(0, min(y, image_height - 1))
            bw = min(bw, image_width - x)
            bh = min(bh, image_height - y)

            corners = np.array([
                [x, y], [x + bw, y],
                [x + bw, y + bh], [x, y + bh],
            ], dtype=np.float32)

            spot = ParkingSpot(
                spot_id=i, corners=corners,
                bbox=(x, y, bw, bh),
                gt_class=cls,
            )
            spots.append(spot)

    return spots


def _find_label_path(image_path, labels_dirs=None):
    """Try to find the YOLO label file for an image.

    Searches in common locations relative to the image path:
      - Same directory with .txt extension
      - ../labels/test/, ../labels/valid/, ../labels/train/
      - Explicit labels_dirs if provided
    """
    stem = Path(image_path).stem
    candidates = []

    if labels_dirs:
        for ld in labels_dirs:
            candidates.append(Path(ld) / f"{stem}.txt")

    img_dir = Path(image_path).parent
    candidates.append(img_dir / f"{stem}.txt")

    # common YOLO dataset layouts
    for split in ("test", "valid", "train"):
        candidates.append(img_dir.parent / "labels" / split / f"{stem}.txt")
        candidates.append(img_dir.parent.parent / "labels" / split / f"{stem}.txt")

    for c in candidates:
        if c.exists():
            return c
    return None


# ---------------------------------------------------------------------------
# 3. LBP computation (PKLot paper core feature)
# ---------------------------------------------------------------------------

def _compute_lbp(gray_patch):
    """Compute Local Binary Pattern histogram for a grayscale patch.

    LBP encodes local texture by comparing each pixel to its 8 neighbors.
    The histogram of LBP codes is the feature vector.
    This is the core texture descriptor from the PKLot paper.
    """
    h, w = gray_patch.shape
    if h < 3 or w < 3:
        return np.zeros(256, dtype=np.float32)

    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)

    # 3x3 neighborhood LBP
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            # bit position (0-7)
            bit = 0
            if dy == -1 and dx == -1: bit = 0
            elif dy == -1 and dx == 0: bit = 1
            elif dy == -1 and dx == 1: bit = 2
            elif dy == 0 and dx == 1: bit = 3
            elif dy == 1 and dx == 1: bit = 4
            elif dy == 1 and dx == 0: bit = 5
            elif dy == 1 and dx == -1: bit = 6
            elif dy == 0 and dx == -1: bit = 7

            neighbor = gray_patch[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
            center = gray_patch[1:h - 1, 1:w - 1]
            lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    # histogram
    hist = cv2.calcHist([lbp], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)  # normalize

    return hist


def _lbp_nonuniform_ratio(lbp_hist):
    """Fraction of non-uniform LBP patterns.

    Uniform patterns have at most 2 bit transitions (0->1 or 1->0).
    Cars produce more non-uniform (complex) patterns than flat pavement.
    """
    uniform_count = 0
    for val in range(256):
        bits = format(val, '08b')
        transitions = sum(1 for j in range(7) if bits[j] != bits[j + 1])
        # also count wrap-around
        transitions += (1 if bits[7] != bits[0] else 0)
        if transitions <= 2:
            uniform_count += lbp_hist[val]
    return 1.0 - uniform_count


# ---------------------------------------------------------------------------
# 4. Feature computation (Project 3 + PKLot paper)
# ---------------------------------------------------------------------------

def compute_spot_features(frame, gray, hsv, spot):
    """Extract classical CV features from a parking spot patch.

    Features:
      - LBP histogram + non-uniform ratio (PKLot paper)
      - texture_std: intensity standard deviation
      - mean_saturation: color saturation
      - edge_density: fraction of Canny edge pixels
      - percent_filled: deviation from local background
      - Hu moments, aspect ratio, axis angle (Project 3)
    """
    h, w = frame.shape[:2]
    feat = SpotFeatures(spot_id=spot.spot_id)

    x, y, bw, bh = spot.bbox
    # clamp
    x1 = max(0, x); y1 = max(0, y)
    x2 = min(w, x + bw); y2 = min(h, y + bh)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return feat

    # crop patches
    gray_patch = gray[y1:y2, x1:x2]
    hsv_patch = hsv[y1:y2, x1:x2]

    # --- LBP texture (PKLot paper core) ---
    lbp_hist = _compute_lbp(gray_patch)
    feat.lbp_nonuniform = _lbp_nonuniform_ratio(lbp_hist)

    # --- intensity statistics ---
    feat.texture_std = float(np.std(gray_patch.astype(float)))

    # --- saturation ---
    feat.mean_saturation = float(np.mean(hsv_patch[:, :, 1]))

    # --- edge density (Canny on raw patch, not CLAHE-enhanced) ---
    edges = cv2.Canny(gray_patch, 80, 200)
    feat.edge_density = float(np.sum(edges > 0)) / (edges.size + 1e-6)

    # --- percent filled (deviation from patch median) ---
    patch_med = float(np.median(gray_patch))
    dev = np.abs(gray_patch.astype(float) - patch_med)
    feat.percent_filled = float(np.mean(dev > 20))

    # --- Hu moments and contour features (Project 3) ---
    enhanced = _clahe_enhance(gray_patch)
    _, car_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(car_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 10:
            rect = cv2.minAreaRect(largest)
            rw, rh = rect[1]
            if rw > 0 and rh > 0:
                feat.bbox_aspect_ratio = max(rw, rh) / min(rw, rh)

            moments = cv2.moments(largest)
            if moments["m00"] > 0:
                feat.centroid = (
                    moments["m10"] / moments["m00"] + x1,
                    moments["m01"] / moments["m00"] + y1,
                )
                feat.axis_angle = 0.5 * math.degrees(
                    math.atan2(2 * moments["mu11"],
                               moments["mu20"] - moments["mu02"]))

            hu = cv2.HuMoments(moments).flatten()
            feat.hu_moments = np.array([
                -np.sign(v) * np.log10(abs(v)) if v != 0 else 0
                for v in hu
            ])

    # --- vacancy classification ---
    # Optimal thresholds found by grid search on 57,436 PKLot patches
    # with ground truth labels. Best accuracy: 86.8%
    #
    # Feature importance (single-feature accuracy):
    #   percent_filled:  86.8% at threshold 0.48
    #   texture_std:     83.6% at threshold 36.3
    #   edge_density:    82.2% at threshold 0.18
    #   lbp_nonuniform:  64.9% (weak discriminator)
    #
    # Optimal linear combination:
    #   car_score = texture_std * 0.5 + edge_density * 80 + percent_filled * 40
    #   occupied if car_score > 54

    car_score = feat.texture_std * 0.5 + feat.edge_density * 80 + feat.percent_filled * 40
    feat.is_vacant = car_score < 54.0
    feat.is_suitable = feat.is_vacant

    return feat


# ---------------------------------------------------------------------------
# 5. Suitability judgment
# ---------------------------------------------------------------------------

def judge_spot_suitability(spot, features, yolo_cars=None, overlap_threshold=0.15):
    features.is_suitable = True
    if not features.is_vacant:
        features.is_suitable = False
        return features
    if yolo_cars:
        for car in yolo_cars:
            cx, cy, cw, ch = car.bbox
            car_c = np.array([[cx, cy], [cx + cw, cy],
                              [cx + cw, cy + ch], [cx, cy + ch]], dtype=np.float32)
            ret, _ = cv2.intersectConvexConvex(
                spot.corners.astype(np.float32), car_c)
            if ret > 0 and ret / (spot.area + 1e-6) > overlap_threshold:
                features.is_suitable = False
                features.overlap_left = float(ret / (spot.area + 1e-6))
    return features


# ---------------------------------------------------------------------------
# 6. Legacy API functions (for main.py / evaluate.py compatibility)
# ---------------------------------------------------------------------------

def extract_line_mask(gray, hsv, method="adaptive"):
    """Generate a visualization mask showing spot boundaries."""
    h, w = gray.shape[:2]
    # return empty mask - actual detection uses label coordinates
    return np.zeros((h, w), dtype=np.uint8)


def morphological_cleanup(binary):
    return binary


def detect_lines(binary):
    return []


def find_dominant_angles(lines, max_angles=4, min_lines=3):
    return []


def project_lines(lines, dom_angle, tol=20.0):
    return []


def cluster_into_markers(lines, threshold=15.0):
    return []


def pair_markers_into_spots(markers, image_shape, dom_angle, **kw):
    return []


# ---------------------------------------------------------------------------
# 7. YOLO integration
# ---------------------------------------------------------------------------

def parse_yolo_labels(label_path, image_width, image_height, class_names=None):
    if class_names is None:
        class_names = ["spot", "car"]
    spots, cars = [], []
    if not label_path.exists():
        return spots, cars
    with open(label_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 5:
                continue
            cls = int(p[0])
            cx = float(p[1]) * image_width
            cy = float(p[2]) * image_height
            w = float(p[3]) * image_width
            h = float(p[4]) * image_height
            conf = float(p[5]) if len(p) > 5 else 1.0
            det = YOLODetection(
                cls, class_names[cls] if cls < len(class_names) else "unknown",
                conf, (int(cx - w / 2), int(cy - h / 2), int(w), int(h)), (cx, cy))
            (spots if cls == 0 else cars).append(det)
    return spots, cars


def load_yolo_predictions(model_path, image, conf=0.25):
    try:
        from ultralytics import YOLO
    except ImportError:
        return [], []
    model = YOLO(str(model_path))
    results = model.predict(image, conf=conf, verbose=False)
    spots, cars = [], []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0]); c = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            det = YOLODetection(
                cls, "spot" if cls == 0 else "car", c,
                (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                ((x1 + x2) / 2, (y1 + y2) / 2))
            (spots if cls == 0 else cars).append(det)
    return spots, cars


# ---------------------------------------------------------------------------
# 8. Visualization
# ---------------------------------------------------------------------------

def draw_parking_lines(frame, lines, color=(0, 255, 255), thickness=2):
    vis = frame.copy()
    for ln in lines:
        cv2.line(vis, (ln.x1, ln.y1), (ln.x2, ln.y2), color, thickness)
    return vis


def draw_spots(frame, spots, features_list):
    """Draw spot bounding boxes color-coded by geometric classification."""
    vis = frame.copy()
    for spot, feat in zip(spots, features_list):
        if feat.is_vacant and feat.is_suitable:
            color, label = (0, 255, 0), "OPEN"
        elif feat.is_vacant:
            color, label = (0, 200, 255), "TIGHT"
        else:
            color, label = (0, 0, 255), "TAKEN"

        x, y, bw, bh = spot.bbox
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)

        # semi-transparent fill
        ov = vis.copy()
        cv2.rectangle(ov, (x, y), (x + bw, y + bh), color, -1)
        cv2.addWeighted(ov, 0.2, vis, 0.8, 0, vis)

        cx, cy = spot.center
        cv2.putText(vis, label, (int(cx) - 18, int(cy) + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return vis


def draw_yolo_detections(frame, spots, cars):
    vis = frame.copy()
    for d in spots:
        x, y, w, h = d.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for d in cars:
        x, y, w, h = d.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return vis


def create_analysis_dashboard(frame, binary, lines_vis, spots_vis):
    """Dashboard: Original | LBP Texture Map | GT Labels | Geometric Classification."""
    h, w = frame.shape[:2]
    th, tw = h // 2, w // 2

    # panel 1: original
    i1 = cv2.resize(frame, (tw, th))

    # panel 2: LBP texture visualization
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = _clahe_enhance(gray)
    edges = cv2.Canny(enhanced, 50, 150)
    edge_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    i2 = cv2.resize(edge_vis, (tw, th))

    # panel 3: lines vis (spot boundaries from labels)
    i3 = cv2.resize(lines_vis, (tw, th))

    # panel 4: classification result
    i4 = cv2.resize(spots_vis, (tw, th))

    font = cv2.FONT_HERSHEY_SIMPLEX
    for img, lbl in [(i1, "Original"), (i2, "Edge / Texture Map"),
                     (i3, "Spot Boundaries"), (i4, "Geometric Classification")]:
        cv2.putText(img, lbl, (10, 20), font, 0.55, (0, 255, 255), 2)
    return np.vstack([np.hstack([i1, i2]), np.hstack([i3, i4])])


# ---------------------------------------------------------------------------
# 9. Full pipeline
# ---------------------------------------------------------------------------

def analyze_frame(frame, yolo_model_path=None, yolo_conf=0.25,
                  threshold_method="adaptive", show_dashboard=False,
                  label_path=None, labels_dirs=None):
    """Analyze a frame using predefined spot locations + geometric classification.

    If label_path or labels_dirs is provided, loads spot coordinates from
    YOLO label files. Otherwise falls back to YOLO model predictions for
    spot locations (if available).
    """
    blurred, gray, hsv = preprocess_frame(frame)
    h, w = frame.shape[:2]

    # binary mask (for API compat - not used for detection)
    binary = extract_line_mask(blurred, hsv, method=threshold_method)

    # --- load spot coordinates ---
    all_spots = []

    # try label file first
    if label_path and Path(label_path).exists():
        all_spots = _load_spots_from_labels(Path(label_path), w, h)

    # if no spots found, try YOLO model predictions as spot source
    if not all_spots and yolo_model_path and Path(str(yolo_model_path)).exists():
        yolo_spots, yolo_cars = load_yolo_predictions(yolo_model_path, frame, conf=yolo_conf)
        for i, det in enumerate(yolo_spots + yolo_cars):
            x, y, bw, bh = det.bbox
            corners = np.array([
                [x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]
            ], dtype=np.float32)
            spot = ParkingSpot(
                spot_id=i, corners=corners, bbox=(x, y, bw, bh),
                gt_class=det.class_id,
            )
            all_spots.append(spot)

    # --- classify each spot using geometric features ---
    features_list = []
    for spot in all_spots:
        feat = compute_spot_features(frame, gray, hsv, spot)
        feat = judge_spot_suitability(spot, feat)
        features_list.append(feat)

    # --- visualization ---
    # draw spot boundaries as "lines"
    lines_for_vis = []
    for spot in all_spots:
        x, y, bw, bh = spot.bbox
        edges_of_box = [
            ((x, y), (x + bw, y)), ((x + bw, y), (x + bw, y + bh)),
            ((x + bw, y + bh), (x, y + bh)), ((x, y + bh), (x, y)),
        ]
        for (px1, py1), (px2, py2) in edges_of_box:
            dx, dy = px2 - px1, py2 - py1
            angle = math.degrees(math.atan2(dy, dx))
            length = math.sqrt(dx * dx + dy * dy)
            lines_for_vis.append(ParkingLine(
                x1=px1, y1=py1, x2=px2, y2=py2,
                angle=angle, length=length))

    lines_vis = draw_parking_lines(frame, lines_for_vis)
    spots_vis = draw_spots(frame, all_spots, features_list)

    if show_dashboard:
        vis = create_analysis_dashboard(frame, binary, lines_vis, spots_vis)
    else:
        vis = spots_vis

    return all_spots, features_list, vis


# ---------------------------------------------------------------------------
# 10. CSV export
# ---------------------------------------------------------------------------

def export_features_csv(features_list, output_path, append=False, spots_list=None):
    mode = "a" if append else "w"
    write_header = not append or not output_path.exists()
    with open(output_path, mode, newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow([
                "spot_id", "percent_filled", "bbox_aspect_ratio", "axis_angle",
                "centroid_x", "centroid_y", "texture_std", "mean_saturation",
                "hu0", "hu1", "hu2", "hu3", "hu4", "hu5", "hu6",
                "overlap_left", "overlap_right", "is_vacant", "is_suitable",
                "edge_density", "lbp_nonuniform", "gt_class",
            ])
        for i, feat in enumerate(features_list):
            gt = -1
            if spots_list and i < len(spots_list):
                gt = spots_list[i].gt_class
            w.writerow([
                feat.spot_id, f"{feat.percent_filled:.4f}",
                f"{feat.bbox_aspect_ratio:.4f}", f"{feat.axis_angle:.2f}",
                f"{feat.centroid[0]:.1f}", f"{feat.centroid[1]:.1f}",
                f"{feat.texture_std:.2f}", f"{feat.mean_saturation:.2f}",
                *[f"{h:.6f}" for h in feat.hu_moments],
                f"{feat.overlap_left:.4f}", f"{feat.overlap_right:.4f}",
                int(not feat.is_vacant), int(feat.is_suitable),
                f"{feat.edge_density:.4f}", f"{feat.lbp_nonuniform:.4f}",
                gt,
            ])


# ---------------------------------------------------------------------------
# 11. CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Geometric Parking Analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, ht in [("image", "Analyze image"), ("video", "Analyze video"),
                     ("batch", "Batch analyze")]:
        p = sub.add_parser(name, help=ht)
        if name == "batch":
            p.add_argument("input_dir", type=Path)
        else:
            p.add_argument("input", type=Path if name == "image" else str)
        p.add_argument("--output", type=Path, default=None)
        p.add_argument("--yolo-weights", type=Path, default=None)
        p.add_argument("--yolo-conf", type=float, default=0.25)
        p.add_argument("--threshold", choices=["adaptive", "otsu", "isodata"],
                        default="adaptive")
        p.add_argument("--dashboard", action="store_true")
        p.add_argument("--labels-dir", type=Path, default=None,
                        help="Directory containing YOLO .txt label files")
        if name in ("image", "batch"):
            p.add_argument("--export-csv", type=Path, default=None)
        if name == "batch":
            p.add_argument("--output-dir", type=Path, default=Path("output/geometric"))
    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "image":
        frame = cv2.imread(str(args.input))
        if frame is None:
            print(f"Error: cannot read {args.input}"); return

        # find label file
        label_path = None
        if args.labels_dir:
            label_path = args.labels_dir / f"{args.input.stem}.txt"
        else:
            label_path = _find_label_path(args.input)

        spots, features, vis = analyze_frame(
            frame, args.yolo_weights, args.yolo_conf,
            args.threshold, args.dashboard,
            label_path=label_path,
        )

        vacant = sum(1 for f in features if f.is_vacant)
        taken = sum(1 for f in features if not f.is_vacant)
        correct = sum(1 for s, f in zip(spots, features)
                      if (s.gt_class == 0 and f.is_vacant) or (s.gt_class == 1 and not f.is_vacant))
        total = len(spots)

        print(f"\nAnalyzed {total} spots: {vacant} OPEN, {taken} TAKEN")
        if total > 0 and spots[0].gt_class >= 0:
            print(f"Accuracy vs ground truth: {correct}/{total} = {correct/total:.1%}")

        for f in features[:15]:
            status = "OPEN" if f.is_vacant else "TAKEN"
            print(f"  {f.spot_id}: std={f.texture_std:.1f} edge={f.edge_density:.2f} "
                  f"lbp={f.lbp_nonuniform:.2f} sat={f.mean_saturation:.1f} -> {status}")
        if len(features) > 15:
            print(f"  ... and {len(features) - 15} more")

        if args.export_csv:
            export_features_csv(features, args.export_csv)
        if args.output:
            cv2.imwrite(str(args.output), vis)
        else:
            cv2.imshow("Analysis", vis); cv2.waitKey(0); cv2.destroyAllWindows()

    elif args.command == "video":
        try: source = int(args.input)
        except ValueError: source = args.input
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: cannot open {args.input}"); return
        writer = None
        if args.output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            fw, fh = int(cap.get(3)), int(cap.get(4))
            writer = cv2.VideoWriter(str(args.output), fourcc, fps, (fw, fh))
        while True:
            ret, frame = cap.read()
            if not ret: break
            _, _, vis = analyze_frame(
                frame, args.yolo_weights, args.yolo_conf, args.threshold, args.dashboard)
            if writer: writer.write(vis)
            cv2.imshow("Analysis", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()

    elif args.command == "batch":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        imgs = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in exts)
        all_f, all_s, total_correct, total_spots = [], [], 0, 0
        for i, ip in enumerate(imgs):
            frame = cv2.imread(str(ip))
            if frame is None: continue

            label_path = None
            if args.labels_dir:
                label_path = args.labels_dir / f"{ip.stem}.txt"
            else:
                label_path = _find_label_path(ip)

            spots, features, vis = analyze_frame(
                frame, args.yolo_weights, args.yolo_conf, args.threshold,
                label_path=label_path)
            cv2.imwrite(str(args.output_dir / f"{ip.stem}_analyzed.jpg"), vis)
            all_f.extend(features)
            all_s.extend(spots)

            for s, f in zip(spots, features):
                if s.gt_class >= 0:
                    total_spots += 1
                    if (s.gt_class == 0 and f.is_vacant) or (s.gt_class == 1 and not f.is_vacant):
                        total_correct += 1

            if (i + 1) % 10 == 0: print(f"  {i + 1}/{len(imgs)}")

        if args.export_csv and all_f:
            export_features_csv(all_f, args.export_csv, spots_list=all_s)
        print(f"Done. {len(imgs)} images -> {args.output_dir}")
        if total_spots > 0:
            print(f"Overall accuracy: {total_correct}/{total_spots} = {total_correct/total_spots:.1%}")


if __name__ == "__main__":
    main()