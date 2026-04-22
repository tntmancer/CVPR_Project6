#!/usr/bin/env python3
"""Demo video generator for the parking analysis project.

Creates an annotated demo video (or processes webcam) showing the
full pipeline in action. Designed for the project presentation
and report submission.

Modes:
    live     - Real-time webcam demo with dashboard overlay
    video    - Process an existing video file
    montage  - Create a montage from a folder of images (slideshow style)

Usage:
    # live webcam demo
    python demo.py live

    # process a video with dashboard
    python demo.py video parking_lot.mp4 --output demo_output.mp4

    # create image montage demo
    python demo.py montage data/pklot/yolo_ready/images/test \
                   --output demo_montage.mp4 --max-images 20

Authors: Sreenath Prasadh, Timothy Bennett
Course: CS 5330 - Pattern Recognition and Computer Vision
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config
from main import process_frame, print_summary


# Title card

def create_title_card(
    width: int = 1280,
    height: int = 720,
    duration_frames: int = 90,
) -> np.ndarray:
    """Create a title card frame for the demo video."""
    card = np.zeros((height, width, 3), dtype=np.uint8)

    # dark gradient background
    for y in range(height):
        shade = int(40 + 30 * (y / height))
        card[y, :] = (shade, shade - 10, shade - 20)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # title
    title = "Parking Spot Analysis"
    (tw, th), _ = cv2.getTextSize(title, font, 1.5, 3)
    cv2.putText(card, title, ((width - tw) // 2, height // 3),
                font, 1.5, (0, 255, 255), 3)

    # subtitle
    sub = "Geometric + YOLO Fusion Pipeline"
    (sw, sh), _ = cv2.getTextSize(sub, font, 0.8, 2)
    cv2.putText(card, sub, ((width - sw) // 2, height // 3 + 50),
                font, 0.8, (200, 200, 200), 2)

    # authors
    authors = "Sreenath Prasadh & Timothy Bennett"
    (aw, ah), _ = cv2.getTextSize(authors, font, 0.6, 1)
    cv2.putText(card, authors, ((width - aw) // 2, height // 2 + 30),
                font, 0.6, (180, 180, 180), 1)

    # course
    course = "CS 5330 - Pattern Recognition and Computer Vision"
    (cw, ch), _ = cv2.getTextSize(course, font, 0.5, 1)
    cv2.putText(card, course, ((width - cw) // 2, height // 2 + 60),
                font, 0.5, (150, 150, 150), 1)

    # legend
    legend_y = int(height * 0.7)
    legend_x = width // 4
    legend_items = [
        (config.COLOR_OPEN, "OPEN - Vacant and suitable"),
        (config.COLOR_TIGHT, "TIGHT - Vacant but neighbor over line"),
        (config.COLOR_TAKEN, "TAKEN - Occupied"),
    ]
    for color, text in legend_items:
        cv2.rectangle(card, (legend_x, legend_y - 12), (legend_x + 20, legend_y + 4), color, -1)
        cv2.putText(card, text, (legend_x + 30, legend_y),
                    font, 0.5, (200, 200, 200), 1)
        legend_y += 30

    return card


# HUD overlay

def add_hud(
    frame: np.ndarray,
    frame_num: int,
    fps: float,
    num_spots: int,
    num_vacant: int,
    num_suitable: int,
) -> np.ndarray:
    """Add a heads-up display overlay to the frame."""
    vis = frame.copy()
    h, w = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # semi-transparent black bar at top
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    # HUD text
    cv2.putText(vis, f"Frame: {frame_num}", (10, 28),
                font, 0.5, (200, 200, 200), 1)
    cv2.putText(vis, f"FPS: {fps:.1f}", (150, 28),
                font, 0.5, (0, 255, 255), 1)
    cv2.putText(vis, f"Spots: {num_spots}", (280, 28),
                font, 0.5, (200, 200, 200), 1)

    # color-coded vacancy info
    cv2.putText(vis, f"Open: {num_suitable}", (400, 28),
                font, 0.5, config.COLOR_OPEN, 1)
    tight = num_vacant - num_suitable
    cv2.putText(vis, f"Tight: {tight}", (520, 28),
                font, 0.5, config.COLOR_TIGHT, 1)
    occupied = num_spots - num_vacant
    cv2.putText(vis, f"Taken: {occupied}", (640, 28),
                font, 0.5, config.COLOR_TAKEN, 1)

    return vis


# Demo modes


def demo_live(args: argparse.Namespace) -> None:
    """Live webcam demo."""
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: cannot open camera {args.camera}")
        sys.exit(1)

    yolo_w = args.yolo_weights or config.YOLO_WEIGHTS_PKLOT

    writer = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
        fps = cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (w, h))

        # write title card
        title = create_title_card(w, h)
        for _ in range(int(fps * 3)):  # 3 seconds
            writer.write(title)

    print("Live demo started. Press 'q' to quit, 's' to snapshot.")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        fused, vis = process_frame(
            frame,
            yolo_weights=yolo_w if yolo_w.exists() else None,
            threshold_method=args.threshold,
            show_dashboard=args.dashboard,
        )
        elapsed = time.time() - start
        frame_count += 1
        fps_val = 1.0 / elapsed if elapsed > 0 else 0

        total = len(fused)
        vacant = sum(1 for r in fused if r.final_vacant)
        suitable = sum(1 for r in fused if r.final_vacant and r.final_suitable)

        vis = add_hud(vis, frame_count, fps_val, total, vacant, suitable)

        if writer:
            writer.write(vis)

        cv2.imshow("Parking Demo", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            snap = f"demo_snap_{frame_count:05d}.jpg"
            cv2.imwrite(snap, vis)
            print(f"  Saved {snap}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Demo ended. {frame_count} frames processed.")


def demo_video(args: argparse.Namespace) -> None:
    """Process an existing video file."""
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"Error: cannot open {args.input}")
        sys.exit(1)

    yolo_w = args.yolo_weights or config.YOLO_WEIGHTS_PKLOT

    fps = cap.get(cv2.CAP_PROP_FPS) or config.DEFAULT_FPS
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
        out_w, out_h = w, h
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (out_w, out_h))

        # title card
        title = create_title_card(out_w, out_h)
        for _ in range(int(fps * 3)):
            writer.write(title)

    print(f"Processing {total_frames} frames at {fps:.1f} FPS...")
    frame_count = 0
    total_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        fused, vis = process_frame(
            frame,
            yolo_weights=yolo_w if yolo_w.exists() else None,
            threshold_method=args.threshold,
            show_dashboard=args.dashboard,
        )
        elapsed = time.time() - start
        total_time += elapsed
        frame_count += 1

        fps_val = 1.0 / elapsed if elapsed > 0 else 0
        total = len(fused)
        vacant = sum(1 for r in fused if r.final_vacant)
        suitable = sum(1 for r in fused if r.final_vacant and r.final_suitable)

        vis = add_hud(vis, frame_count, fps_val, total, vacant, suitable)

        if writer:
            writer.write(vis)

        if frame_count % 100 == 0:
            pct = frame_count / total_frames * 100 if total_frames > 0 else 0
            print(f"  Frame {frame_count}/{total_frames} ({pct:.0f}%)")

    cap.release()
    if writer:
        writer.release()

    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\nDone. {frame_count} frames, avg {avg_fps:.1f} FPS")
    if args.output:
        print(f"Output saved to {args.output}")


def demo_montage(args: argparse.Namespace) -> None:
    """Create a montage demo from a folder of still images."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in extensions)

    if args.max_images:
        images = images[:args.max_images]

    if not images:
        print(f"No images found in {args.input_dir}")
        sys.exit(1)

    yolo_w = args.yolo_weights or config.YOLO_WEIGHTS_PKLOT

    # determine output size from first image
    first = cv2.imread(str(images[0]))
    h, w = first.shape[:2]
    fps = 2.0  # 2 frames per second for slideshow

    writer = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (w, h))

        # title card (3 seconds)
        title = create_title_card(w, h)
        for _ in range(int(fps * 3)):
            writer.write(title)

    print(f"Creating montage from {len(images)} images...")

    for i, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        # resize to match output
        frame = cv2.resize(frame, (w, h))

        fused, vis = process_frame(
            frame,
            yolo_weights=yolo_w if yolo_w.exists() else None,
            threshold_method=args.threshold,
            show_dashboard=args.dashboard,
        )

        total = len(fused)
        vacant = sum(1 for r in fused if r.final_vacant)
        suitable = sum(1 for r in fused if r.final_vacant and r.final_suitable)

        vis = add_hud(vis, i + 1, 0, total, vacant, suitable)

        # add filename
        cv2.putText(
            vis, img_path.name, (10, vis.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
        )

        if writer:
            # hold each frame for 1 second (fps frames)
            for _ in range(max(1, int(fps))):
                writer.write(vis)

        # also save individual analyzed images
        if args.save_frames:
            out_dir = config.OUTPUT_DEMO / "frames"
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_demo.jpg"), vis)

        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(images)}")

    if writer:
        writer.release()
        print(f"Montage saved to {args.output}")

    print("Done.")


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo video generator for parking analysis"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # shared
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--yolo-weights", type=Path, default=None)
    common.add_argument("--threshold", choices=["adaptive", "otsu", "isodata"],
                        default=config.THRESHOLD_METHOD)
    common.add_argument("--dashboard", action="store_true")
    common.add_argument("--output", type=Path, default=None,
                        help="Output video file path")

    # live
    p_live = sub.add_parser("live", parents=[common], help="Live webcam demo")
    p_live.add_argument("--camera", type=int, default=0)

    # video
    p_vid = sub.add_parser("video", parents=[common], help="Process video file")
    p_vid.add_argument("input", type=Path)

    # montage
    p_mont = sub.add_parser("montage", parents=[common],
                            help="Create slideshow from images")
    p_mont.add_argument("input_dir", type=Path)
    p_mont.add_argument("--max-images", type=int, default=None)
    p_mont.add_argument("--save-frames", action="store_true",
                        help="Also save individual analyzed frames")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "live":
        demo_live(args)
    elif args.command == "video":
        demo_video(args)
    elif args.command == "montage":
        demo_montage(args)


if __name__ == "__main__":
    main()