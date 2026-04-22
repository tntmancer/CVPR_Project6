#!/usr/bin/env python3
"""YOLOv11 parking detector for CNRPark+EXT.

This script has multiple modes:
1) Prepare YOLO labels from CNRPark+EXT CSV annotations and camera slot boxes.
2) Train a YOLOv11 model to detect two classes: spot and car.
3) Run prediction and save images/videos with bounding boxes.
4) Evaluate on a held-out test split.

Dataset mapping used:
- spot: occupancy == 0 (empty slot)
- car: occupancy == 1 (occupied slot)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

# for visualization and reporting use umbrella class names
CLASS_NAMES = ["spot", "car"]

WEATHER_MAP = {
	"S": "SUNNY",
	"O": "OVERCAST",
	"R": "RAINY",
}


def resolve_device(requested_device: str) -> str:
	"""Resolve device string and fall back to CPU when needed."""
	dev = requested_device.strip().lower()

	# use GPU if available, otherwise CPU
	if dev == "auto":
		if torch.cuda.is_available() and torch.cuda.device_count() > 0:
			return "0"
		print("CUDA not available. Falling back to CPU.")
		return "cpu"

	# explicit CPU request
	if dev == "cpu":
		return "cpu"

	# user asked for CUDA explicitly (e.g., '0' or '0,1').
	if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
		print(f"Requested device '{requested_device}' is not available. Falling back to CPU.")
		return "cpu"

	return requested_device


def _safe_link_or_copy(src: Path, dst: Path) -> None:
	dst.parent.mkdir(parents=True, exist_ok=True)
	if dst.exists():
		return
	try:
		os.link(src, dst)
	except OSError:
		shutil.copy2(src, dst)


def _xywh_to_yolo(bbox: Tuple[float, float, float, float], width: int, height: int) -> Tuple[float, float, float, float]:
	"""Convert (x, y, w, h) in pixel coordinates to YOLO format (cx, cy, nw, nh) normalized to [0, 1]."""
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


def _camera_folder_name(camera_code: str) -> str | None:
	# CNRPark+EXT uses camera codes 01..09 for full-image subset rows.
	value = camera_code.strip()
	if value.isdigit():
		idx = int(value)
		if 1 <= idx <= 9:
			return f"camera{idx}"
	return None


def _decode_datetime(datetime_value: str) -> Tuple[str, str] | None:
	"""Return (date_yyyy_mm_dd, hhmm) from CNRPark datetime formats."""
	dt = datetime_value.strip()

	if "_" not in dt:
		return None

	date_part, time_part = dt.split("_", 1)
	# supports both legacy format (20150703_0805) and EXT format (2015-11-12_07.09).
	if "-" in date_part:
		date_value = date_part
	else:
		if len(date_part) != 8 or not date_part.isdigit():
			return None
		date_value = f"{date_part[0:4]}-{date_part[4:6]}-{date_part[6:8]}"

	hhmm = time_part.replace(".", "")
	if len(hhmm) != 4 or not hhmm.isdigit():
		return None
	return date_value, hhmm


def _split_for_key(key: str, train_ratio: float = 0.7, val_ratio: float = 0.15) -> str:
	"""Deterministic split assignment from image key."""
	# Hash-based split keeps assignment stable across reruns and machines.
	digest = hashlib.md5(key.encode("utf-8")).hexdigest()
	score = int(digest[:8], 16) / float(0xFFFFFFFF)
	if score < train_ratio:
		return "train"
	if score < train_ratio + val_ratio:
		return "valid"
	return "test"


def _load_camera_slot_boxes(
	data_root: Path,
	target_width: int,
	target_height: int,
	source_width: int,
	source_height: int,
) -> Dict[str, Dict[int, Tuple[float, float, float, float]]]:
	"""Load slot boxes from camera1.csv..camera9.csv and scale to target image size."""
	slots_by_camera: Dict[str, Dict[int, Tuple[float, float, float, float]]] = {}
	# camera CSV coordinates are defined on source resolution (default 2592x1944)
	# full images are 1000x750, so we rescale boxes at load time
	scale_x = float(target_width) / float(source_width)
	scale_y = float(target_height) / float(source_height)

	for i in range(1, 10):
		camera_name = f"camera{i}"
		csv_path = data_root / f"camera{i}.csv"
		if not csv_path.exists():
			continue

		camera_slots: Dict[int, Tuple[float, float, float, float]] = {}
		with csv_path.open("r", encoding="utf-8") as f:
			reader = csv.DictReader(f)
			for row in reader:
				slot_id = int(str(row["SlotId"]).strip())
				x = float(row["X"]) * scale_x
				y = float(row["Y"]) * scale_y
				w = float(row["W"]) * scale_x
				h = float(row["H"]) * scale_y
				camera_slots[slot_id] = (x, y, w, h)

		slots_by_camera[camera_name] = camera_slots

	return slots_by_camera


def prepare_dataset(
	data_root: Path,
	out_root: Path,
	source_width: int,
	source_height: int,
	target_width: int,
	target_height: int,
) -> Path:
	"""Build YOLO dataset from CNRPark+EXT CSV + camera slot boxes + full images."""
	# full-frame images used for training labels
	full_images_root = data_root / "FULL_IMAGE_1000x750"
	# occupancy labels (one row per slot per timestamp)
	ann_csv = data_root / "CNRPark+EXT.csv"

	if not full_images_root.exists():
		raise FileNotFoundError(f"Missing full-image directory: {full_images_root}")
	if not ann_csv.exists():
		raise FileNotFoundError(f"Missing annotation CSV: {ann_csv}")

	slots_by_camera = _load_camera_slot_boxes(
		data_root=data_root,
		target_width=target_width,
		target_height=target_height,
		source_width=source_width,
		source_height=source_height,
	)
	if not slots_by_camera:
		raise RuntimeError("No camera slot CSV files found (camera1.csv..camera9.csv).")

	for split in ("train", "valid", "test"):
		(out_root / "images" / split).mkdir(parents=True, exist_ok=True)
		(out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

	image_records: Dict[str, Dict[str, object]] = {}
	missing_images = 0
	missing_slots = 0
	skipped_rows = 0

	with ann_csv.open("r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			# map camera/weather/time metadata in CSV to concrete full image path
			camera_dir = _camera_folder_name(str(row["camera"]))
			if camera_dir is None:
				continue

			weather_dir = WEATHER_MAP.get(str(row["weather"]).strip())
			if weather_dir is None:
				skipped_rows += 1
				continue

			decoded = _decode_datetime(str(row["datetime"]))
			if decoded is None:
				skipped_rows += 1
				continue
			date_value, hhmm = decoded

			try:
				slot_id = int(str(row["slot_id"]).strip())
				occupancy = int(str(row["occupancy"]).strip())
			except ValueError:
				skipped_rows += 1
				continue

			# lookup per-camera slot ROI, then assign class from occupancy
			bbox = slots_by_camera.get(camera_dir, {}).get(slot_id)
			if bbox is None:
				missing_slots += 1
				continue

			image_name = f"{date_value}_{hhmm}.jpg"
			src_image = full_images_root / weather_dir / date_value / camera_dir / image_name
			if not src_image.exists():
				missing_images += 1
				continue

			image_key = f"{weather_dir}_{date_value}_{camera_dir}_{hhmm}"
			split = _split_for_key(image_key)
			out_name = f"{image_key}.jpg"

			record = image_records.setdefault(
				image_key,
				{
					"src": src_image,
					"split": split,
					"out_name": out_name,
					"lines": [],
				},
			)

			# 0=spot (empty), 1=car (occupied)
			class_id = 1 if occupancy == 1 else 0
			cx, cy, nw, nh = _xywh_to_yolo(bbox, target_width, target_height)
			if nw <= 0.0 or nh <= 0.0:
				continue
			record["lines"].append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

	split_counts = {"train": 0, "valid": 0, "test": 0}
	for image_key, record in image_records.items():
		split = str(record["split"])
		out_name = str(record["out_name"])
		src_image = Path(record["src"])

		dst_image = out_root / "images" / split / out_name
		_safe_link_or_copy(src_image, dst_image)

		label_path = out_root / "labels" / split / f"{Path(out_name).stem}.txt"
		# de-duplicate label rows in case of duplicate CSV entries
		lines = list(dict.fromkeys(record["lines"]))
		with label_path.open("w", encoding="utf-8") as lf:
			lf.write("\n".join(lines))

		split_counts[split] += 1

	yaml_path = write_dataset_yaml(out_root)
	print(
		"Prepared CNRParkEXT dataset with "
		f"train={split_counts['train']}, valid={split_counts['valid']}, test={split_counts['test']} images"
	)
	print(f"Skipped rows: {skipped_rows}, missing slot boxes: {missing_slots}, missing images: {missing_images}")
	print(f"Wrote dataset yaml: {yaml_path}")
	return yaml_path


def write_dataset_yaml(out_root: Path) -> Path:
	"""Write dataset yaml for YOLO training."""
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


def _metric_value(metrics: object, names: Tuple[str, ...], default: float = 0.0) -> float:
	for name in names:
		value = getattr(metrics, name, None)
		if value is not None:
			try:
				return float(value)
			except (TypeError, ValueError):
				continue
	return default


def _save_metric_summary_plot(metrics: object, save_dir: Path) -> Path:
	box_metrics = getattr(metrics, "box", metrics)
	values = [
		_metric_value(box_metrics, ("mp", "precision", "p")),
		_metric_value(box_metrics, ("mr", "recall", "r")),
		_metric_value(box_metrics, ("map50", "map_50", "mAP50")),
		_metric_value(box_metrics, ("map", "map5095", "mAP50-95")),
	]
	labels = ["Precision", "Recall", "mAP@50", "mAP@50:95"]

	fig, ax = plt.subplots(figsize=(8, 4.5))
	bars = ax.bar(labels, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
	ax.set_ylim(0.0, 1.0)
	ax.set_ylabel("Score")
	ax.set_title("Test Set Detection Metrics")
	ax.grid(axis="y", alpha=0.25)
	for bar, value in zip(bars, values, strict=False):
		ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
	fig.tight_layout()

	plot_path = save_dir / "metrics_summary.png"
	fig.savefig(plot_path, dpi=200)
	plt.close(fig)
	return plot_path


def evaluate_model(
	weights: Path,
	dataset_yaml: Path,
	device: str,
	project_dir: Path,
) -> None:
	model = YOLO(str(weights))
	metrics = model.val(
		data=str(dataset_yaml),
		split="test",
		device=device,
		plots=True,
		save_json=True,
		project=str(project_dir),
		name="test_eval",
		exist_ok=True,
	)

	save_dir = Path(metrics.save_dir)
	plot_path = _save_metric_summary_plot(metrics, save_dir)
	box_metrics = getattr(metrics, "box", metrics)
	precision = _metric_value(box_metrics, ("mp", "precision", "p"))
	recall = _metric_value(box_metrics, ("mr", "recall", "r"))
	map50 = _metric_value(box_metrics, ("map50", "map_50", "mAP50"))
	map5095 = _metric_value(box_metrics, ("map", "map5095", "mAP50-95"))

	print("Test-set metrics:")
	print(f"  Precision:   {precision:.4f}")
	print(f"  Recall:      {recall:.4f}")
	print(f"  mAP@50:      {map50:.4f}")
	print(f"  mAP@50:95:   {map5095:.4f}")
	print(f"Saved metric summary plot: {plot_path}")
	print(f"Ultralytics evaluation artifacts: {save_dir}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="YOLOv11 CNRParkEXT spot + car detector")
	sub = parser.add_subparsers(dest="command", required=True)

	# common arguments for all subcommands
	common = argparse.ArgumentParser(add_help=False)
	# root folder expected to contain FULL_IMAGE_1000x750, CNRPark+EXT.csv, and camera*.csv files
	common.add_argument("--data-root", type=Path, default=Path("data/CNRParkEXT"), help="CNRParkEXT dataset root")
	# output root for prepared YOLO dataset
	# will contain images/ and labels/ subdirectories
	common.add_argument(
		"--prepared-root",
		type=Path,
		default=Path("data/CNRParkEXT/yolo_ready"),
		help="Output folder for converted YOLO dataset",
	)
	# output root for training runs, predictions, and evaluation artifacts
	common.add_argument("--runs-dir", type=Path, default=Path("runs/cnrpark"), help="Output folder for train/predict runs")
	# device can be 'auto' to select GPU if available, 'cpu' to force CPU, or a specific CUDA device id like '0'
	# falls back to CPU if requested GPU is not available
	common.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', or CUDA device id like '0'")
	# annotation CSV coordinates are in the original camera resolution
	common.add_argument("--source-width", type=int, default=2592, help="Original annotation coordinate width")
	common.add_argument("--source-height", type=int, default=1944, help="Original annotation coordinate height")
	# full image dimensions where labels are projected
	common.add_argument("--target-width", type=int, default=1000, help="Full image width in pixels")
	common.add_argument("--target-height", type=int, default=750, help="Full image height in pixels")

	# subcommands
	# convert CNRPark+EXT CSV labels into YOLO format
	p_prepare = sub.add_parser("prepare", parents=[common], help="Convert CNRParkEXT CSV to YOLO dataset")
	p_prepare.set_defaults(func="prepare")

	# train YOLOv11 model
	p_train = sub.add_parser("train", parents=[common], help="Prepare dataset and train YOLOv11")
	# model checkpoint can be a pretrained YOLOv11 variant like 'yolo11n.pt' or a custom checkpoint path
	p_train.add_argument("--model", type=str, default="yolo11n.pt", help="YOLOv11 checkpoint")
	# training hyperparameters
	p_train.add_argument("--epochs", type=int, default=50)
	p_train.add_argument("--imgsz", type=int, default=640)
	p_train.add_argument("--batch", type=int, default=16)
	p_train.set_defaults(func="train")

	# run prediction with bounding boxes
	p_predict = sub.add_parser("predict", parents=[common], help="Run prediction with bounding boxes")
	# source for weights defaults to the best.pt from training, but can be overridden to use any checkpoint
	p_predict.add_argument("--weights", type=Path, default=Path("runs/cnrpark/train/weights/best.pt"))
	# source can be an image, video, folder of images/videos, or webcam index (e.g., '0')
	p_predict.add_argument(
		"--source",
		type=str,
		default="data/CNRParkEXT/FULL_IMAGE_1000x750",
		help="Image/video path, folder, or webcam index",
	)
	# confidence threshold for predictions
	p_predict.add_argument("--conf", type=float, default=0.25)
	p_predict.set_defaults(func="predict")

	# evaluate model on test split and save metrics/plots
	p_eval = sub.add_parser("evaluate", parents=[common], help="Evaluate the model on the test split and save plots")
	p_eval.add_argument("--weights", type=Path, default=Path("runs/cnrpark/train/weights/best.pt"))
	p_eval.add_argument("--source", type=str, default="data/CNRParkEXT/FULL_IMAGE_1000x750", help="Accepted for compatibility")
	p_eval.set_defaults(func="evaluate")

	# run the entire pipeline: prepare, train, and predict
	p_all = sub.add_parser("all", parents=[common], help="Prepare, train, and predict in one command")
	p_all.add_argument("--model", type=str, default="yolo11n.pt")
	p_all.add_argument("--epochs", type=int, default=50)
	p_all.add_argument("--imgsz", type=int, default=640)
	p_all.add_argument("--batch", type=int, default=16)
	p_all.add_argument("--source", type=str, default="data/CNRParkEXT/FULL_IMAGE_1000x750")
	p_all.add_argument("--conf", type=float, default=0.25)
	p_all.set_defaults(func="all")

	return parser.parse_args()


def main() -> None:
	args = parse_args()
	args.device = resolve_device(args.device)

	if args.func == "prepare":
		prepare_dataset(
			data_root=args.data_root,
			out_root=args.prepared_root,
			source_width=args.source_width,
			source_height=args.source_height,
			target_width=args.target_width,
			target_height=args.target_height,
		)
		return

	if args.func == "train":
		dataset_yaml = prepare_dataset(
			data_root=args.data_root,
			out_root=args.prepared_root,
			source_width=args.source_width,
			source_height=args.source_height,
			target_width=args.target_width,
			target_height=args.target_height,
		)
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

	if args.func == "evaluate":
		dataset_yaml = prepare_dataset(
			data_root=args.data_root,
			out_root=args.prepared_root,
			source_width=args.source_width,
			source_height=args.source_height,
			target_width=args.target_width,
			target_height=args.target_height,
		)
		evaluate_model(
			weights=args.weights,
			dataset_yaml=dataset_yaml,
			device=args.device,
			project_dir=args.runs_dir,
		)
		return

	if args.func == "all":
		dataset_yaml = prepare_dataset(
			data_root=args.data_root,
			out_root=args.prepared_root,
			source_width=args.source_width,
			source_height=args.source_height,
			target_width=args.target_width,
			target_height=args.target_height,
		)
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
