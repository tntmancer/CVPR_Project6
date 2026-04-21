#!/usr/bin/env python3
"""YOLOv11 parking detector.

This script hasmultiple modes:
1) Convert COCO annotations in data/pklot/{train,valid,test} to YOLO format.
2) Train a YOLOv11 model to detect two classes: spot and car.
3) Run prediction and saves images/videos with bounding boxes.

Dataset mapping used:
- spot: spaces, space-empty, spot-like labels
- car: space-occupied; car/vehicle-like labels
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from itertools import islice
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

# for visualization and reporting use umbrella class names
CLASS_NAMES = ["spot", "car"]

def resolve_device(requested_device: str) -> str:
	'''
	Safely handle the requested device string, allowing 'auto' to select GPU if available, and falling back to CPU if not
	This makes the script environment-agnostic
	'''
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


def _is_webcam_source(source: str) -> bool:
	"""True when source is a webcam index such as '0'."""
	return source.strip().isdigit()


def _iter_media_paths(source: str | Path) -> Iterator[str]:
	"""Yield image/video paths recursively from a file or directory source."""
	path = Path(source)
	media_extensions = {
		".png",
		".jpg",
		".jpeg",
		".bmp",
		".webp",
		".mp4",
		".avi",
		".mov",
		".mkv",
		".wmv",
		".m4v",
	}

	if not path.exists():
		return

	if path.is_file():
		yield str(path)
		return

	for p in sorted(path.rglob("*")):
		if p.is_file() and p.suffix.lower() in media_extensions:
			yield str(p)


def _iter_chunks(items: Iterator[str], chunk_size: int) -> Iterator[List[str]]:
	"""Yield fixed-size lists from an iterator without loading everything into memory."""
	while True:
		chunk = list(islice(items, chunk_size))
		if not chunk:
			break
		yield chunk


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
	# explicit name pulled from dataset
	ann_path = split_dir / "_annotations.coco.json"
	if not ann_path.exists():
		raise FileNotFoundError(f"Missing annotation file: {ann_path}")

	# load COCO annotations
	with ann_path.open("r", encoding="utf-8") as f:
		coco = json.load(f)

	# map COCO category IDs to target class indices (0 or 1), or None to ignore
	categories: Dict[int, int | None] = {}
	for cat in coco.get("categories", []):
		categories[int(cat["id"])] = _map_category_to_target(str(cat["name"]))

	# map image IDs to their metadata for quick lookup
	images_by_id: Dict[int, Dict] = {int(im["id"]): im for im in coco.get("images", [])}
	# group annotations by image ID for efficient access
	anns_by_image: Dict[int, List[Dict]] = {}
	for ann in coco.get("annotations", []):
		image_id = int(ann["image_id"])
		anns_by_image.setdefault(image_id, []).append(ann)

	# prepare output directories for split
	split_name = split_dir.name
	out_images = out_root / "images" / split_name
	out_labels = out_root / "labels" / split_name
	out_images.mkdir(parents=True, exist_ok=True)
	out_labels.mkdir(parents=True, exist_ok=True)

	processed = 0
	# iterate over all images in the split, convert annotations, and save labels
	for image_id, image_meta in images_by_id.items():
		file_name = image_meta["file_name"]
		width = int(image_meta["width"])
		height = int(image_meta["height"])

		# get source image path and ensure it exists
		src_image = split_dir / file_name
		if not src_image.exists():
			continue

		# save image to output directory 
		# link if possible, copy if not
		# for system compatibility while optimizing for large datasets
		dst_image = out_images / file_name
		_safe_link_or_copy(src_image, dst_image)

		# convert annotations for this image to YOLO format and save to label file
		label_path = out_labels / f"{Path(file_name).stem}.txt"
		lines: List[str] = []

		# process each annotation for this image, mapping categories and converting bboxes
		for ann in anns_by_image.get(image_id, []):
			src_class = categories.get(int(ann["category_id"]))
			if src_class is None:
				continue

			bbox = ann.get("bbox")
			if not bbox or len(bbox) != 4:
				continue

			# convert COCO bbox to YOLO format and ensure valid dimensions
			cx, cy, nw, nh = _xywh_to_yolo(bbox, width, height)
			if nw <= 0.0 or nh <= 0.0:
				continue

			lines.append(f"{src_class} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

		with label_path.open("w", encoding="utf-8") as lf:
			lf.write("\n".join(lines))

		processed += 1

	return processed


def write_dataset_yaml(out_root: Path) -> Path:
	"""
	Write the dataset.yaml file for YOLO training, pointing to the prepared images and labels.
	"""
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
	"""
	Convert the dataset from COCO format in data_root to YOLO format in out_root, and write the dataset.yaml file.
	Returns the path to the dataset.yaml file."""
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
	"""
	The best.pt file is saved by Ultralytics in the run directory under weights/best.pt relative to the save_dir
	Return the path to this file for use in prediction and evaluation steps.
	"""
	best_path = Path(result.save_dir) / "weights" / "best.pt"
	print(f"Training done. Best weights: {best_path}")
	return best_path


def predict(
	weights: Path,
	source: str,
	conf: float,
	device: str,
	project_dir: Path,
	batch_files: int,
) -> None:
	model = YOLO(str(weights))

	if _is_webcam_source(source):
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
		return

	if batch_files < 1:
		raise ValueError("batch_files must be >= 1")

	media_iter = _iter_media_paths(source)
	processed = 0
	for idx, chunk in enumerate(_iter_chunks(media_iter, batch_files), start=1):
		model.predict(
			source=chunk,
			conf=conf,
			device=device,
			save=True,
			project=str(project_dir),
			name="predict",
			exist_ok=True,
		)
		processed += len(chunk)
		print(f"Processed batch {idx}: +{len(chunk)} files (total={processed})")

	if processed == 0:
		raise FileNotFoundError(f"No images or videos found in {source}")

	print(f"Predictions saved under: {project_dir / 'predict'}")


def _metric_value(metrics: object, names: Tuple[str, ...], default: float = 0.0) -> float:
	"""
	Helper function to extract a metric value from the metrics object, trying multiple possible attribute names for Ultralytics compatibility.
	Returns the first valid float value found, or a default if none are found.
	"""
	for name in names:
		value = getattr(metrics, name, None)
		if value is not None:
			try:
				return float(value)
			except (TypeError, ValueError):
				continue
	return default


def _save_metric_summary_plot(metrics: object, save_dir: Path) -> Path:
	"""
	save a bar plot summarizing key detection metrics (precision, recall, mAP@50, mAP@50:95) extracted from the metrics object.
	"""
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
	"""
	evaluate the model on the test split, extract key metrics, and save a summary plot.
	The metrics object contains attributes for precision, recall, mAP@50, and mAP@50:95, which we will extract and display.
	The plot will be saved in the evaluation run directory for easy reference.
	"""

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
	"""
	Parse command-line arguments for the YOLOv11 parking spot and car detector script, supporting multiple subcommands for dataset preparation, training, prediction, and evaluation.
	Each subcommand has its own specific arguments, while common arguments are shared across all commands.
	"""
	parser = argparse.ArgumentParser(description="YOLOv11 parking spot + car detector")
	sub = parser.add_subparsers(dest="command", required=True)

	# common arguments for all subcommands
	common = argparse.ArgumentParser(add_help=False)
	# dataset root is expected to have train/valid/test subdirectories with COCO annotations and images
	common.add_argument("--data-root", type=Path, default=Path("data/pklot"), help="Dataset root with train/valid/test")
	# output root for prepared YOLO dataset
	# will contain images/ and labels/ subdirectories
	common.add_argument(
		"--prepared-root",
		type=Path,
		default=Path("data/pklot/yolo_ready"),
		help="Output folder for converted YOLO dataset",
	)
	# output root for training runs, predictions, and evaluation artifacts
	common.add_argument("--runs-dir", type=Path, default=Path("runs/yopo"), help="Output folder for train/predict runs")
	# device can be 'auto' to select GPU if available, 'cpu' to force CPU, or a specific CUDA device id like '0'
	# falls back to CPU if requested GPU is not available
	common.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', or CUDA device id like '0'")

	# subcommands
	# convert COCO to YOLO dataset
	p_prepare = sub.add_parser("prepare", parents=[common], help="Convert COCO to YOLO dataset")
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
	p_predict.add_argument("--weights", type=Path, default=Path("runs/detect/runs/yopo/train/weights/best.pt"))
	# source can be an image, video, folder of images/videos, or webcam index (e.g., '0')
	p_predict.add_argument("--source", type=str, default="data/pklot/test", help="Image/video path, folder, or webcam index")
	# confidence threshold for predictions
	p_predict.add_argument("--conf", type=float, default=0.25)
	p_predict.add_argument(
		"--batch-files",
		type=int,
		default=200,
		help="Number of media files to send to YOLO per prediction call to reduce peak RAM",
	)
	p_predict.set_defaults(func="predict")

	# evaluate model on test split and save metrics/plots
	p_eval = sub.add_parser("evaluate", parents=[common], help="Evaluate the model on the test split and save plots")
	p_eval.add_argument("--weights", type=Path, default=Path("runs/detect/runs/yopo/train/weights/best.pt"))
	p_eval.add_argument("--source", type=str, default="data/pklot/test", help="Accepted for compatibility; the test split is evaluated")
	p_eval.set_defaults(func="evaluate")

	# run the entire pipeline: prepare, train, and predict
	p_all = sub.add_parser("all", parents=[common], help="Prepare, train, and predict in one command")
	p_all.add_argument("--model", type=str, default="yolo11n.pt")
	p_all.add_argument("--epochs", type=int, default=50)
	p_all.add_argument("--imgsz", type=int, default=640)
	p_all.add_argument("--batch", type=int, default=16)
	p_all.add_argument("--source", type=str, default="data/test")
	p_all.add_argument("--conf", type=float, default=0.25)
	p_all.add_argument("--batch-files", type=int, default=200)
	p_all.set_defaults(func="all")

	return parser.parse_args()


def main() -> None:
	"""
	Main entry point for the YOLOv11 parking spot and car detector script. 
	Parses command-line arguments, resolves the device, and dispatches to the appropriate function based on the selected subcommand:
	prepare, train, predict, evaluate, or all).
	"""
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
			batch_files=args.batch_files,
		)
		return

	if args.func == "evaluate":
		dataset_yaml = prepare_dataset(args.data_root, args.prepared_root)
		evaluate_model(
			weights=args.weights,
			dataset_yaml=dataset_yaml,
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
			batch_files=args.batch_files,
		)
		return

	raise RuntimeError("Unknown command")


if __name__ == "__main__":
	main()
