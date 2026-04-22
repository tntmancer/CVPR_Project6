#!/usr/bin/env python3
"""Unified entry point for the parking spot analysis project.

Runs both the geometric (classical CV) pipeline and the YOLO (deep learning)
pipeline on the same input, then fuses their results for a final decision
on spot vacancy and suitability.

Usage:
    # analyze a single image with both pipelines
    python main.py image parking.jpg

    # analyze with dashboard visualization
    python main.py image parking.jpg --dashboard --save output/result.jpg

    # analyze a video or webcam
    python main.py video 0
    python main.py video parking_lot.mp4 --save output/demo.mp4

    # batch process a directory
    python main.py batch data/pklot/test --save-dir output/results

    # use only geometric pipeline (no YOLO)
    python main.py image parking.jpg --geometric-only

    # use only YOLO pipeline
    python main.py image parking.jpg --yolo-only

Authors: Sreenath Prasadh, Timothy Bennett
Course: CS 5330 - Pattern Recognition and Computer Vision
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from geometricParking import (
    ParkingSpot,
    SpotFeatures,
    YOLODetection,
    analyze_frame,
    compute_spot_features,
    create_analysis_dashboard,
    draw_parking_lines,
    draw_spots,
    draw_yolo_detections,
    export_features_csv,
    judge_spot_suitability,
    load_yolo_predictions,
)


# Fused result data class

@dataclass
class FusedSpotResult:
    """Combined result from both pipelines for a single parking spot."""
    spot_id: int
    # geometric pipeline results
    geo_vacant: Optional[bool] = None
    geo_suitable: Optional[bool] = None
    geo_fill: float = 0.0
    # YOLO pipeline results
    yolo_class: Optional[str] = None       # "spot" or "car"
    yolo_confidence: float = 0.0
    # fused decision
    final_vacant: bool = True
    final_suitable: bool = True
    # source of truth for the final decision
    decision_source: str = "none"


# IoU computation for matching geometric spots to YOLO detections

def compute_iou(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
) -> float:
    """Compute intersection over union between two (x, y, w, h) boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # convert to (x_min, y_min, x_max, y_max)
    a_min_x, a_min_y = x1, y1
    a_max_x, a_max_y = x1 + w1, y1 + h1
    b_min_x, b_min_y = x2, y2
    b_max_x, b_max_y = x2 + w2, y2 + h2

    inter_x1 = max(a_min_x, b_min_x)
    inter_y1 = max(a_min_y, b_min_y)
    inter_x2 = min(a_max_x, b_max_x)
    inter_y2 = min(a_max_y, b_max_y)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = w1 * h1
    area_b = w2 * h2
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


# Fusion logic

def fuse_results(
    geo_spots: List[ParkingSpot],
    geo_features: List[SpotFeatures],
    yolo_spots: List[YOLODetection],
    yolo_cars: List[YOLODetection],
    iou_threshold: float = config.IOU_MATCH_THRESHOLD,
) -> List[FusedSpotResult]:
    """Fuse geometric and YOLO results for each detected spot.

    Strategy:
    - Match each geometric spot to YOLO detections via IoU
    - If both pipelines agree, use that decision
    - If they disagree, prefer YOLO for vacancy (trained on data)
      but prefer geometric for suitability (line-based analysis)
    """
    results = []

    for spot, feat in zip(geo_spots, geo_features):
        fused = FusedSpotResult(spot_id=spot.spot_id)

        # geometric results
        fused.geo_vacant = feat.is_vacant
        fused.geo_suitable = feat.is_suitable
        fused.geo_fill = feat.percent_filled

        # find best matching YOLO detection for this spot
        best_iou = 0.0
        best_det = None

        all_yolo = yolo_spots + yolo_cars
        for det in all_yolo:
            iou = compute_iou(spot.bbox, det.bbox)
            if iou > best_iou:
                best_iou = iou
                best_det = det

        if best_det and best_iou >= iou_threshold:
            fused.yolo_class = best_det.class_name
            fused.yolo_confidence = best_det.confidence

        # fusion decision
        if fused.yolo_class is not None:
            # YOLO matched: use YOLO for vacancy, geometric for suitability
            fused.final_vacant = (fused.yolo_class == "spot")
            fused.final_suitable = feat.is_suitable if fused.final_vacant else False
            fused.decision_source = "fused"
        elif fused.geo_vacant is not None:
            # YOLO did not match: fall back to geometric
            fused.final_vacant = fused.geo_vacant
            fused.final_suitable = fused.geo_suitable
            fused.decision_source = "geometric"
        else:
            fused.decision_source = "none"

        results.append(fused)

    return results


# Fused visualization

def draw_fused_results(
    frame: np.ndarray,
    geo_spots: List[ParkingSpot],
    fused_results: List[FusedSpotResult],
    yolo_cars: List[YOLODetection],
) -> np.ndarray:
    """Draw the fused analysis results on the frame."""
    vis = frame.copy()

    for spot, fused in zip(geo_spots, fused_results):
        if fused.final_vacant and fused.final_suitable:
            color = config.COLOR_OPEN
            label = "OPEN"
        elif fused.final_vacant and not fused.final_suitable:
            color = config.COLOR_TIGHT
            label = "TIGHT"
        else:
            color = config.COLOR_TAKEN
            label = "TAKEN"

        corners = spot.corners.astype(np.int32)
        cv2.polylines(vis, [corners], True, color, 2)

        # semi-transparent fill
        overlay = vis.copy()
        cv2.fillPoly(overlay, [corners], color)
        cv2.addWeighted(overlay, config.OVERLAY_ALPHA, vis, 1 - config.OVERLAY_ALPHA, 0, vis)

        # label with source indicator
        cx, cy = spot.center
        source_tag = fused.decision_source[0].upper()  # F, G, or N
        display_text = f"{label} [{source_tag}]"
        cv2.putText(
            vis, display_text, (int(cx) - 30, int(cy) + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2,
        )

        # show YOLO confidence if available
        if fused.yolo_confidence > 0:
            conf_text = f"{fused.yolo_confidence:.0%}"
            cv2.putText(
                vis, conf_text, (int(cx) - 15, int(cy) + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
            )

    # draw YOLO car detections
    for car in yolo_cars:
        x, y, w, h = car.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), config.COLOR_YOLO_CAR, 2)
        cv2.putText(
            vis, f"car {car.confidence:.2f}", (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, config.COLOR_YOLO_CAR, 1,
        )

    return vis


# Summary printing

def print_summary(fused_results: List[FusedSpotResult]) -> None:
    """Print a table summarizing all spot results."""
    total = len(fused_results)
    vacant = sum(1 for r in fused_results if r.final_vacant)
    suitable = sum(1 for r in fused_results if r.final_vacant and r.final_suitable)
    tight = sum(1 for r in fused_results if r.final_vacant and not r.final_suitable)
    occupied = total - vacant

    print(f"\n{'='*65}")
    print(f" PARKING ANALYSIS SUMMARY")
    print(f"{'='*65}")
    print(f" Total spots: {total}")
    print(f" Open (suitable):   {suitable}")
    print(f" Open (tight):      {tight}")
    print(f" Occupied:          {occupied}")
    print(f"{'='*65}")

    print(f"\n{'ID':>4} {'GeoFill':>8} {'GeoVac':>7} {'YOLO':>6} "
          f"{'YConf':>6} {'Final':>7} {'Suit':>6} {'Source':>8}")
    print("-" * 60)
    for r in fused_results:
        geo_vac = "Yes" if r.geo_vacant else "No" if r.geo_vacant is not None else "-"
        yolo_cls = r.yolo_class or "-"
        final = "Vacant" if r.final_vacant else "Taken"
        suit = "Yes" if r.final_suitable else "No"
        print(
            f"{r.spot_id:>4} "
            f"{r.geo_fill:>7.1%} "
            f"{geo_vac:>7} "
            f"{yolo_cls:>6} "
            f"{r.yolo_confidence:>5.0%} "
            f"{final:>7} "
            f"{suit:>6} "
            f"{r.decision_source:>8}"
        )
    print()


# Process a single frame (core logic)

def process_frame(
    frame: np.ndarray,
    yolo_weights: Optional[Path] = None,
    geometric_only: bool = False,
    yolo_only: bool = False,
    threshold_method: str = config.THRESHOLD_METHOD,
    show_dashboard: bool = False,
) -> Tuple[List[FusedSpotResult], np.ndarray]:
    """Process a single frame through both pipelines and fuse results.

    Returns:
        (fused_results, visualization)
    """
    geo_spots = []
    geo_features = []
    yolo_spots_det = []
    yolo_cars_det = []

    # run geometric pipeline
    if not yolo_only:
        geo_spots, geo_features, geo_vis = analyze_frame(
            frame,
            yolo_model_path=None,  # handle YOLO separately for fusion
            threshold_method=threshold_method,
            show_dashboard=False,
        )

    # run YOLO pipeline
    if not geometric_only and yolo_weights and yolo_weights.exists():
        yolo_spots_det, yolo_cars_det = load_yolo_predictions(
            yolo_weights, frame, conf=config.YOLO_CONFIDENCE,
        )

    # fuse results
    if geo_spots:
        # re-judge suitability with YOLO car positions
        for spot, feat in zip(geo_spots, geo_features):
            judge_spot_suitability(spot, feat, yolo_cars=yolo_cars_det)

        fused = fuse_results(
            geo_spots, geo_features, yolo_spots_det, yolo_cars_det
        )
        vis = draw_fused_results(frame, geo_spots, fused, yolo_cars_det)
    elif yolo_spots_det or yolo_cars_det:
        # YOLO only mode
        vis = draw_yolo_detections(frame, yolo_spots_det, yolo_cars_det)
        fused = []
        for i, det in enumerate(yolo_spots_det + yolo_cars_det):
            r = FusedSpotResult(
                spot_id=i,
                yolo_class=det.class_name,
                yolo_confidence=det.confidence,
                final_vacant=(det.class_name == "spot"),
                decision_source="yolo",
            )
            fused.append(r)
    else:
        vis = frame.copy()
        fused = []

    # build dashboard if requested
    if show_dashboard and not yolo_only and geo_spots:
        from geometricParking import (
            preprocess_frame, extract_line_mask,
            morphological_cleanup, detect_lines,
            draw_parking_lines as _draw_lines,
        )
        blurred, gray, hsv = preprocess_frame(frame)
        binary = extract_line_mask(blurred, hsv, method=threshold_method)
        cleaned = morphological_cleanup(binary)
        lines = detect_lines(cleaned)
        lines_vis = _draw_lines(frame, lines)

        vis = create_analysis_dashboard(frame, cleaned, lines_vis, vis)

    return fused, vis


# CLI commands

def cmd_image(args: argparse.Namespace) -> None:
    """Analyze a single image."""
    frame = cv2.imread(str(args.input))
    if frame is None:
        print(f"Error: cannot read image {args.input}")
        sys.exit(1)

    yolo_w = None if args.geometric_only else (args.yolo_weights or config.YOLO_WEIGHTS_PKLOT)

    start = time.time()
    fused, vis = process_frame(
        frame,
        yolo_weights=yolo_w,
        geometric_only=args.geometric_only,
        yolo_only=args.yolo_only,
        threshold_method=args.threshold,
        show_dashboard=args.dashboard,
    )
    elapsed = time.time() - start

    print_summary(fused)
    print(f"Processing time: {elapsed:.3f}s")

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save), vis)
        print(f"Saved to {args.save}")

    if args.export_csv:
        from geometricParking import export_features_csv as _export
        # re-extract geometric features for CSV
        _, geo_feats, _ = analyze_frame(
            frame, threshold_method=args.threshold
        )
        _export(geo_feats, args.export_csv)
        print(f"Features exported to {args.export_csv}")

    # display
    if not args.save:
        cv2.imshow("Parking Analysis", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def cmd_video(args: argparse.Namespace) -> None:
    """Analyze a video or webcam feed."""
    try:
        source = int(args.input)
    except ValueError:
        source = args.input

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: cannot open {args.input}")
        sys.exit(1)

    yolo_w = None if args.geometric_only else (args.yolo_weights or config.YOLO_WEIGHTS_PKLOT)

    writer = None
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
        fps = cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(args.save), fourcc, fps, (w, h))

    frame_count = 0
    total_time = 0.0

    print("Press 'q' to quit, 's' to save current frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        fused, vis = process_frame(
            frame,
            yolo_weights=yolo_w,
            geometric_only=args.geometric_only,
            yolo_only=args.yolo_only,
            threshold_method=args.threshold,
            show_dashboard=args.dashboard,
        )
        elapsed = time.time() - start
        total_time += elapsed
        frame_count += 1

        # overlay FPS
        fps = 1.0 / elapsed if elapsed > 0 else 0
        cv2.putText(
            vis, f"FPS: {fps:.1f}", (10, vis.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )

        if writer:
            writer.write(vis)

        cv2.imshow("Parking Analysis", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            snap = f"snapshot_{frame_count:05d}.jpg"
            cv2.imwrite(snap, vis)
            print(f"Saved {snap}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\nProcessed {frame_count} frames, avg {avg_fps:.1f} FPS")


def cmd_batch(args: argparse.Namespace) -> None:
    """Batch analyze all images in a directory."""
    save_dir = args.save_dir or config.OUTPUT_FUSED
    save_dir.mkdir(parents=True, exist_ok=True)

    yolo_w = None if args.geometric_only else (args.yolo_weights or config.YOLO_WEIGHTS_PKLOT)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in extensions)

    if not images:
        print(f"No images found in {args.input_dir}")
        sys.exit(1)

    print(f"Processing {len(images)} images...")
    all_fused = []

    for i, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        fused, vis = process_frame(
            frame,
            yolo_weights=yolo_w,
            geometric_only=args.geometric_only,
            yolo_only=args.yolo_only,
            threshold_method=args.threshold,
        )

        out_path = save_dir / f"{img_path.stem}_analyzed.jpg"
        cv2.imwrite(str(out_path), vis)
        all_fused.extend(fused)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(images)}")

    # overall summary
    total = len(all_fused)
    vacant = sum(1 for r in all_fused if r.final_vacant)
    suitable = sum(1 for r in all_fused if r.final_vacant and r.final_suitable)
    print(f"\nBatch complete: {total} spots analyzed across {len(images)} images")
    print(f"  Vacant: {vacant}, Suitable: {suitable}, Occupied: {total - vacant}")
    print(f"  Results saved to {save_dir}")


# Argument parsing

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parking Spot Analysis - Unified Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py image lot.jpg --dashboard --save output/result.jpg
  python main.py video 0 --geometric-only
  python main.py batch data/test --save-dir output/batch
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # shared arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--yolo-weights", type=Path, default=None,
                        help="Path to YOLO weights (default: auto-detect)")
    common.add_argument("--threshold", choices=["adaptive", "otsu", "isodata"],
                        default=config.THRESHOLD_METHOD)
    common.add_argument("--dashboard", action="store_true",
                        help="Show 2x2 processing stage dashboard")
    common.add_argument("--geometric-only", action="store_true",
                        help="Use only the geometric pipeline")
    common.add_argument("--yolo-only", action="store_true",
                        help="Use only the YOLO pipeline")

    # image
    p_img = sub.add_parser("image", parents=[common], help="Analyze a single image")
    p_img.add_argument("input", type=Path)
    p_img.add_argument("--save", type=Path, default=None)
    p_img.add_argument("--export-csv", type=Path, default=None)

    # video
    p_vid = sub.add_parser("video", parents=[common], help="Analyze video or webcam")
    p_vid.add_argument("input", type=str)
    p_vid.add_argument("--save", type=Path, default=None)

    # batch
    p_bat = sub.add_parser("batch", parents=[common], help="Batch analyze a directory")
    p_bat.add_argument("input_dir", type=Path)
    p_bat.add_argument("--save-dir", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.geometric_only and args.yolo_only:
        print("Error: cannot use both --geometric-only and --yolo-only")
        sys.exit(1)

    if args.command == "image":
        cmd_image(args)
    elif args.command == "video":
        cmd_video(args)
    elif args.command == "batch":
        cmd_batch(args)


if __name__ == "__main__":
    main()