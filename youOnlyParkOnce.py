#!/usr/bin/env python3
"""YOLOv11 parking detector.

This script:
1) Converts COCO annotations in data/{train,valid,test} to YOLO format.
2) Trains a YOLOv11 model to detect two classes: spot and car.
3) Runs prediction and saves images/videos with bounding boxes.

Dataset mapping used:
- spot: spaces, space-empty, spot-like labels
- car: space-occupied, car/vehicle-like labels
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from ultralytics import YOLO


CLASS_NAMES = ["spot", "car"]


def resolve_device(requested_device: str) -> str:
	"""Return a safe Ultralytics device string based on runtime availability."""
	dev = requested_device.strip().lower()

	if dev == "auto":
		if torch.cuda.is_available() and torch.cuda.device_count() > 0:
			return "0"
		print("CUDA not available. Falling back to CPU.")
		return "cpu"

	if dev == "cpu":
		return "cpu"

	# User asked for CUDA explicitly (e.g., '0' or '0,1').
	if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
		print(f"Requested device '{requested_device}' is not available. Falling back to CPU.")
		return "cpu"

	return requested_device


def _map_category_to_target(cat_name: str) -> int | None:
	"""Map source category names to target classes: 0=spot, 1=car."""
	name = cat_name.strip().lower()

	# Occupied space means a car is present.
	if "occupied" in name or "car" in name or "vehicle" in name:
		return 1

	# Empty/general space labels are treated as parking spots.
	if "space" in name or "spot" in name or "empty" in name:
		return 0

	return None


def _safe_link_or_copy(src: Path, dst: Path) -> None:
	dst.parent.mkdir(parents=True, exist_ok=True)
	if dst.exists():
		return
	try:
		os.link(src, dst)
	except OSError:
		shutil.copy2(src, dst)


def _xywh_to_yolo(bbox: List[float], width: int, height: int) -> Tuple[float, float, float, float]:
	x, y, w, h = bbox
	x = max(0.0, min(x, float(width)))
	y = max(0.0, min(y, float(height)))
	w = max(0.0, min(w, float(width) - x))
	h = max(0.0, min(h, float(height) - y))

	cx = (x + w / 2.0) / float(width)
	cy = (y + h / 2.0) / float(height)
	nw = w / float(width)
	nh = h / float(height)
	return cx, cy, nw, nh


def convert_coco_split(split_dir: Path, out_root: Path) -> int:
	"""Convert one split from COCO json to YOLO txt labels.

	Returns number of images processed.
	"""
	ann_path = split_dir / "_annotations.coco.json"
	if not ann_path.exists():
		raise FileNotFoundError(f"Missing annotation file: {ann_path}")

	with ann_path.open("r", encoding="utf-8") as f:
		coco = json.load(f)

	categories: Dict[int, int | None] = {}
	for cat in coco.get("categories", []):
		categories[int(cat["id"])] = _map_category_to_target(str(cat["name"]))

	images_by_id: Dict[int, Dict] = {int(im["id"]): im for im in coco.get("images", [])}
	anns_by_image: Dict[int, List[Dict]] = {}
	for ann in coco.get("annotations", []):
		image_id = int(ann["image_id"])
		anns_by_image.setdefault(image_id, []).append(ann)

	split_name = split_dir.name
	out_images = out_root / "images" / split_name
	out_labels = out_root / "labels" / split_name
	out_images.mkdir(parents=True, exist_ok=True)
	out_labels.mkdir(parents=True, exist_ok=True)

	processed = 0
	for image_id, image_meta in images_by_id.items():
		file_name = image_meta["file_name"]
		width = int(image_meta["width"])
		height = int(image_meta["height"])

		src_image = split_dir / file_name
		if not src_image.exists():
			continue

		dst_image = out_images / file_name
		_safe_link_or_copy(src_image, dst_image)

		label_path = out_labels / f"{Path(file_name).stem}.txt"
		lines: List[str] = []

		for ann in anns_by_image.get(image_id, []):
			src_class = categories.get(int(ann["category_id"]))
			if src_class is None:
				continue

			bbox = ann.get("bbox")
			if not bbox or len(bbox) != 4:
				continue

			cx, cy, nw, nh = _xywh_to_yolo(bbox, width, height)
			if nw <= 0.0 or nh <= 0.0:
				continue

			lines.append(f"{src_class} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

		with label_path.open("w", encoding="utf-8") as lf:
			lf.write("\n".join(lines))

		processed += 1

	return processed


def write_dataset_yaml(out_root: Path) -> Path:
	yaml_path = out_root / "parking_dataset.yaml"
	content = "\n".join(
		[
			f"path: {out_root.resolve()}",
			"train: images/train",
			"val: images/valid",
			"test: images/test",
			"names:",
			f"  0: {CLASS_NAMES[0]}",
			f"  1: {CLASS_NAMES[1]}",
			"",
		]
	)
	yaml_path.write_text(content, encoding="utf-8")
	return yaml_path


def prepare_dataset(data_root: Path, out_root: Path) -> Path:
	for split in ("train", "valid", "test"):
		split_dir = data_root / split
		if not split_dir.exists():
			raise FileNotFoundError(f"Expected split directory not found: {split_dir}")
		count = convert_coco_split(split_dir, out_root)
		print(f"Prepared split '{split}' with {count} images")

	yaml_path = write_dataset_yaml(out_root)
	print(f"Wrote dataset yaml: {yaml_path}")
	return yaml_path


def train_model(
	dataset_yaml: Path,
	model_name: str,
	epochs: int,
	imgsz: int,
	batch: int,
	device: str,
	project_dir: Path,
) -> Path:
	model = YOLO(model_name)
	result = model.train(
		data=str(dataset_yaml),
		epochs=epochs,
		imgsz=imgsz,
		batch=batch,
		device=device,
		project=str(project_dir),
		name="train",
		exist_ok=True,
	)

	best_path = Path(result.save_dir) / "weights" / "best.pt"
	print(f"Training done. Best weights: {best_path}")
	return best_path


def predict(
	weights: Path,
	source: str,
	conf: float,
	device: str,
	project_dir: Path,
) -> None:
	model = YOLO(str(weights))
	model.predict(
		source=source,
		conf=conf,
		device=device,
		save=True,
		project=str(project_dir),
		name="predict",
		exist_ok=True,
	)
	print(f"Predictions saved under: {project_dir / 'predict'}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="YOLOv11 parking spot + car detector")
	sub = parser.add_subparsers(dest="command", required=True)

	common = argparse.ArgumentParser(add_help=False)
	common.add_argument("--data-root", type=Path, default=Path("data"), help="Dataset root with train/valid/test")
	common.add_argument(
		"--prepared-root",
		type=Path,
		default=Path("data/yolo_ready"),
		help="Output folder for converted YOLO dataset",
	)
	common.add_argument("--runs-dir", type=Path, default=Path("runs/yopo"), help="Output folder for train/predict runs")
	common.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', or CUDA device id like '0'")

	p_prepare = sub.add_parser("prepare", parents=[common], help="Convert COCO to YOLO dataset")
	p_prepare.set_defaults(func="prepare")

	p_train = sub.add_parser("train", parents=[common], help="Prepare dataset and train YOLOv11")
	p_train.add_argument("--model", type=str, default="yolo11n.pt", help="YOLOv11 checkpoint")
	p_train.add_argument("--epochs", type=int, default=50)
	p_train.add_argument("--imgsz", type=int, default=640)
	p_train.add_argument("--batch", type=int, default=16)
	p_train.set_defaults(func="train")

	p_predict = sub.add_parser("predict", parents=[common], help="Run prediction with bounding boxes")
	p_predict.add_argument("--weights", type=Path, default=Path("runs/yopo/train/weights/best.pt"))
	p_predict.add_argument("--source", type=str, default="data/test", help="Image/video path, folder, or webcam index")
	p_predict.add_argument("--conf", type=float, default=0.25)
	p_predict.set_defaults(func="predict")

	p_all = sub.add_parser("all", parents=[common], help="Prepare, train, and predict in one command")
	p_all.add_argument("--model", type=str, default="yolo11n.pt")
	p_all.add_argument("--epochs", type=int, default=50)
	p_all.add_argument("--imgsz", type=int, default=640)
	p_all.add_argument("--batch", type=int, default=16)
	p_all.add_argument("--source", type=str, default="data/test")
	p_all.add_argument("--conf", type=float, default=0.25)
	p_all.set_defaults(func="all")

	return parser.parse_args()


def main() -> None:
	args = parse_args()
	args.device = resolve_device(args.device)

	if args.func == "prepare":
		prepare_dataset(args.data_root, args.prepared_root)
		return

	if args.func == "train":
		dataset_yaml = prepare_dataset(args.data_root, args.prepared_root)
		train_model(
			dataset_yaml=dataset_yaml,
			model_name=args.model,
			epochs=args.epochs,
			imgsz=args.imgsz,
			batch=args.batch,
			device=args.device,
			project_dir=args.runs_dir,
		)
		return

	if args.func == "predict":
		predict(
			weights=args.weights,
			source=args.source,
			conf=args.conf,
			device=args.device,
			project_dir=args.runs_dir,
		)
		return

	if args.func == "all":
		dataset_yaml = prepare_dataset(args.data_root, args.prepared_root)
		weights = train_model(
			dataset_yaml=dataset_yaml,
			model_name=args.model,
			epochs=args.epochs,
			imgsz=args.imgsz,
			batch=args.batch,
			device=args.device,
			project_dir=args.runs_dir,
		)
		predict(
			weights=weights,
			source=args.source,
			conf=args.conf,
			device=args.device,
			project_dir=args.runs_dir,
		)
		return

	raise RuntimeError("Unknown command")


if __name__ == "__main__":
	main()
