#!/usr/bin/env python3
"""YOLOv11 detector for the KITTI dataset.

This script has multiple modes:
1) Prepare dataset configuration for KITTI (Ultralytics built-in dataset yaml).
2) Train a YOLOv11 model on KITTI.
3) Run prediction and save images/videos with bounding boxes.
4) Evaluate on a held-out split (default: val).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO, settings


def resolve_device(requested_device: str) -> str:
	"""
	Safely handle the requested device string, allowing 'auto' to select GPU if available, and falling back to CPU if not.
	This makes the script environment-agnostic.
	"""
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


def prepare_dataset(dataset_spec: str, datasets_dir: Path) -> str:
	"""
	Prepare step for KITTI.

	For Ultralytics datasets such as kitti.yaml, no manual conversion is required.
	Ultralytics handles dataset resolution/download on first use.
	Returns the dataset spec string to be used by training/evaluation.
	"""
	# configure Ultralytics dataset root so built-in datasets (e.g., kitti.yaml) are placed under data/
	datasets_dir.mkdir(parents=True, exist_ok=True)
	settings.update({"datasets_dir": str(datasets_dir.resolve())})
	print(f"Ultralytics datasets_dir set to: {datasets_dir.resolve()}")

	# use a local dataset yaml with an absolute path to guarantee KITTI resolves under data/kitti
	if Path(dataset_spec).name.lower() == "kitti.yaml":
		kitti_root = (datasets_dir / "kitti").resolve()
		local_yaml = (datasets_dir / "kitti_local.yaml").resolve()
		content = "\n".join(
			[
				f"path: {kitti_root}",
				"train: images/train",
				"val: images/val",
				"names:",
				"  0: car",
				"  1: van",
				"  2: truck",
				"  3: pedestrian",
				"  4: person_sitting",
				"  5: cyclist",
				"  6: tram",
				"  7: misc",
				"download: https://github.com/ultralytics/assets/releases/download/v0.0.0/kitti.zip",
				"",
			]
		)
		local_yaml.write_text(content, encoding="utf-8")
		print(f"Using KITTI dataset spec: {local_yaml}")
		print("No explicit conversion step is required for Ultralytics KITTI dataset yaml.")
		return str(local_yaml)

	print(f"Using dataset spec: {dataset_spec}")
	return dataset_spec


def train_model(
	dataset_spec: str,
	model_name: str,
	epochs: int,
	imgsz: int,
	batch: int,
	device: str,
	project_dir: Path,
) -> Path:
	model = YOLO(model_name)
	result = model.train(
		data=dataset_spec,
		epochs=epochs,
		imgsz=imgsz,
		batch=batch,
		device=device,
		project=str(project_dir),
		name="train",
		exist_ok=True,
	)
	"""
	The best.pt file is saved by Ultralytics in the run directory under weights/best.pt relative to the save_dir.
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
	classes: List[int] | None = None,
) -> None:
	model = YOLO(str(weights))
	model.predict(
		source=source,
		conf=conf,
		device=device,
		classes=classes,
		save=True,
		project=str(project_dir),
		name="predict",
		exist_ok=True,
	)
	"""
	The predict method saves images/videos with bounding boxes in the run directory under predict relative to the save_dir.
	"""
	print(f"Predictions saved under: {project_dir / 'predict'}")


def _parse_classes_arg(value: str | None) -> List[int] | None:
	"""Parse a comma-separated class list like '0,1,2' into [0, 1, 2]."""
	if value is None:
		return None
	text = value.strip()
	if not text:
		return None
	classes: List[int] = []
	for part in text.split(","):
		item = part.strip()
		if not item:
			continue
		classes.append(int(item))
	return classes if classes else None


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


def _save_metric_summary_plot(metrics: object, save_dir: Path, split_name: str) -> Path:
	"""
	Save a bar plot summarizing key detection metrics (precision, recall, mAP@50, mAP@50:95) extracted from the metrics object.
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
	ax.set_title(f"{split_name.title()} Split Detection Metrics")
	ax.grid(axis="y", alpha=0.25)
	for bar, value in zip(bars, values, strict=False):
		ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
	fig.tight_layout()

	plot_path = save_dir / f"metrics_summary_{split_name}.png"
	fig.savefig(plot_path, dpi=200)
	plt.close(fig)
	return plot_path


def evaluate_model(
	weights: Path,
	dataset_spec: str,
	split: str,
	device: str,
	project_dir: Path,
) -> None:
	model = YOLO(str(weights))
	metrics = model.val(
		data=dataset_spec,
		split=split,
		device=device,
		plots=True,
		save_json=True,
		project=str(project_dir),
		name=f"{split}_eval",
		exist_ok=True,
	)
	"""
	Evaluate the model on the requested split, extract key metrics, and save a summary plot.
	The metrics object contains attributes for precision, recall, mAP@50, and mAP@50:95, which we will extract and display.
	The plot is saved in the evaluation run directory for easy reference.
	"""

	save_dir = Path(metrics.save_dir)
	plot_path = _save_metric_summary_plot(metrics, save_dir, split)
	box_metrics = getattr(metrics, "box", metrics)
	precision = _metric_value(box_metrics, ("mp", "precision", "p"))
	recall = _metric_value(box_metrics, ("mr", "recall", "r"))
	map50 = _metric_value(box_metrics, ("map50", "map_50", "mAP50"))
	map5095 = _metric_value(box_metrics, ("map", "map5095", "mAP50-95"))

	print(f"{split}-split metrics:")
	print(f"  Precision:   {precision:.4f}")
	print(f"  Recall:      {recall:.4f}")
	print(f"  mAP@50:      {map50:.4f}")
	print(f"  mAP@50:95:   {map5095:.4f}")
	print(f"Saved metric summary plot: {plot_path}")
	print(f"Ultralytics evaluation artifacts: {save_dir}")


def parse_args() -> argparse.Namespace:
	"""
	Parse command-line arguments for the YOLOv11 KITTI detector script, supporting multiple subcommands for dataset preparation, training, prediction, and evaluation.
	Each subcommand has its own specific arguments, while common arguments are shared across all commands.
	"""
	parser = argparse.ArgumentParser(description="YOLOv11 KITTI detector")
	sub = parser.add_subparsers(dest="command", required=True)

	# common arguments for all subcommands
	common = argparse.ArgumentParser(add_help=False)
	# dataset can be an Ultralytics dataset yaml id (e.g., kitti.yaml) or a local yaml path
	common.add_argument("--data", type=str, default="kitti.yaml", help="Dataset yaml/id (e.g., kitti.yaml)")
	# output root for training runs, predictions, and evaluation artifacts
	common.add_argument("--runs-dir", type=Path, default=Path("runs/kitti"), help="Output folder for train/predict runs")
	# Ultralytics datasets directory; KITTI will be downloaded/resolved under this folder (e.g., data/kitti)
	common.add_argument("--datasets-dir", type=Path, default=Path("data"), help="Root folder for Ultralytics datasets")
	# device can be 'auto' to select GPU if available, 'cpu' to force CPU, or a specific CUDA device id like '0'
	# falls back to CPU if requested GPU is not available
	common.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', or CUDA device id like '0'")

	# subcommands
	# prepare dataset spec for KITTI training
	p_prepare = sub.add_parser("prepare", parents=[common], help="Prepare KITTI dataset spec")
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
	p_predict.add_argument("--weights", type=Path, default=Path("runs/detect/runs/kitti/train/weights/best.pt"))
	# source can be an image, video, folder of images/videos, or webcam index (e.g., '0')
	p_predict.add_argument("--source", type=str, default="data/street_view_images", help="Image/video path, folder, or webcam index")
	# confidence threshold for predictions
	p_predict.add_argument("--conf", type=float, default=0.25)
	# optional class filter, e.g. '2' for KITTI car class depending on class index mapping
	p_predict.add_argument("--classes", type=str, default=None, help="Optional comma-separated class ids to keep")
	p_predict.set_defaults(func="predict")

	# evaluate model on split and save metrics/plots
	p_eval = sub.add_parser("evaluate", parents=[common], help="Evaluate the model on a split and save plots")
	p_eval.add_argument("--weights", type=Path, default=Path("runs/detect/runs/kitti/train/weights/best.pt"))
	p_eval.add_argument("--split", type=str, default="val", help="Evaluation split name (e.g., val, test)")
	p_eval.add_argument("--source", type=str, default="", help="Accepted for compatibility")
	p_eval.set_defaults(func="evaluate")

	# run the entire pipeline: prepare, train, and predict
	p_all = sub.add_parser("all", parents=[common], help="Prepare, train, and predict in one command")
	p_all.add_argument("--model", type=str, default="yolo11n.pt")
	p_all.add_argument("--epochs", type=int, default=50)
	p_all.add_argument("--imgsz", type=int, default=640)
	p_all.add_argument("--batch", type=int, default=16)
	p_all.add_argument("--source", type=str, default="data/street_view_images")
	p_all.add_argument("--conf", type=float, default=0.25)
	p_all.add_argument("--classes", type=str, default=None, help="Optional comma-separated class ids to keep")
	p_all.set_defaults(func="all")

	return parser.parse_args()


def main() -> None:
	"""
	Main entry point for the YOLOv11 KITTI detector script.
	Parses command-line arguments, resolves the device, and dispatches to the appropriate function based on the selected subcommand:
	prepare, train, predict, evaluate, or all.
	"""
	args = parse_args()
	args.device = resolve_device(args.device)

	if args.func == "prepare":
		prepare_dataset(args.data, args.datasets_dir)
		return

	if args.func == "train":
		dataset_spec = prepare_dataset(args.data, args.datasets_dir)
		train_model(
			dataset_spec=dataset_spec,
			model_name=args.model,
			epochs=args.epochs,
			imgsz=args.imgsz,
			batch=args.batch,
			device=args.device,
			project_dir=args.runs_dir,
		)
		return

	if args.func == "predict":
		classes = _parse_classes_arg(args.classes)
		predict(
			weights=args.weights,
			source=args.source,
			conf=args.conf,
			device=args.device,
			project_dir=args.runs_dir,
			classes=classes,
		)
		return

	if args.func == "evaluate":
		dataset_spec = prepare_dataset(args.data, args.datasets_dir)
		evaluate_model(
			weights=args.weights,
			dataset_spec=dataset_spec,
			split=args.split,
			device=args.device,
			project_dir=args.runs_dir,
		)
		return

	if args.func == "all":
		classes = _parse_classes_arg(args.classes)
		dataset_spec = prepare_dataset(args.data, args.datasets_dir)
		weights = train_model(
			dataset_spec=dataset_spec,
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
			classes=classes,
		)
		return

	raise RuntimeError("Unknown command")


if __name__ == "__main__":
	main()
