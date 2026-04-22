"""Centralized configuration for the parking analysis project.

All tunable parameters live here so both the geometric pipeline
and the YOLO pipeline can reference them. Adjust these values
when testing on different parking lots, lighting conditions,
or camera angles.

Authors: Sreenath Prasadh, Timothy Bennett
Course: CS 5330 - Pattern Recognition and Computer Vision
"""

from pathlib import Path


# Project paths

# YOLO model weights 
YOLO_WEIGHTS_PKLOT = Path("runs/detect/runs/yopo/train/weights/best.pt")
YOLO_WEIGHTS_CNRPARK = Path("runs/cnrpark/train/weights/best.pt")
YOLO_WEIGHTS_DEFAULT = Path("yolo11n.pt")

# dataset roots
DATA_ROOT_PKLOT = Path("data/pklot")
DATA_ROOT_CNRPARK = Path("data/CNRParkEXT")

# output directories
OUTPUT_DIR = Path("output")
OUTPUT_GEOMETRIC = OUTPUT_DIR / "geometric"
OUTPUT_YOLO = OUTPUT_DIR / "yolo"
OUTPUT_FUSED = OUTPUT_DIR / "fused"
OUTPUT_EVAL = OUTPUT_DIR / "evaluation"
OUTPUT_DEMO = OUTPUT_DIR / "demo"

# class names (shared with Tim's YOLO config)
CLASS_NAMES = ["spot", "car"]


# Preprocessing

# Gaussian blur kernel size and sigma
BLUR_KERNEL_SIZE = (5, 5)
BLUR_SIGMA = 1.5


# Thresholding

# which method to use: "adaptive", "otsu", or "isodata"
THRESHOLD_METHOD = "adaptive"

# adaptive threshold parameters
ADAPTIVE_BLOCK_SIZE = 25
ADAPTIVE_C = -10

# ISODATA (k-means) subsampling factor
ISODATA_SUBSAMPLE = 16


# Morphological cleanup

# small kernel for general noise removal (erosion/dilation)
MORPH_KERNEL_SMALL = (3, 3)

# rectangular kernel for line continuity preservation
MORPH_KERNEL_LINE = (5, 3)

# number of iterations for opening and closing
MORPH_OPEN_ITERATIONS = 1
MORPH_CLOSE_ITERATIONS = 2
MORPH_DILATE_ITERATIONS = 1


# Hough line detection

# Canny edge detection thresholds
CANNY_LOW = 50
CANNY_HIGH = 150

# HoughLinesP parameters
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LENGTH = 50
HOUGH_MAX_LINE_GAP = 20

# angle tolerance in degrees for filtering lines
# lines within this many degrees of 0 (horizontal) or 90 (vertical) are kept
ANGLE_TOLERANCE = 30.0


# Line clustering

# max perpendicular distance (pixels) to merge nearby lines
LINE_CLUSTER_THRESHOLD = 30.0


# Spot extraction

# min and max pixel width between two vertical lines to count as a spot
MIN_SPOT_WIDTH = 40
MAX_SPOT_WIDTH = 200

# padding around detected vertical line extent for spot boundaries
SPOT_VERTICAL_PADDING = 10


# Feature computation and vacancy detection

# percent fill threshold: above this, spot is considered occupied
VACANCY_FILL_THRESHOLD = 0.35

# minimum contour area (pixels) to consider as a real object
MIN_CONTOUR_AREA = 100

# car mask thresholds (for standalone geometric detection)
CAR_INTENSITY_THRESHOLD = 100   # pixels darker than this might be cars
CAR_SATURATION_THRESHOLD = 40   # pixels more saturated than this might be cars



# Over-the-line suitability judgment

# pixel tolerance before counting as overlap
LINE_OVERLAP_TOLERANCE = 10

# fraction of car width past the line to flag as unsuitable
OVERLAP_THRESHOLD = 0.15

# partial fill threshold for debris/obstruction detection
PARTIAL_FILL_THRESHOLD = 0.15


# YOLO inference

# confidence threshold for YOLO predictions
YOLO_CONFIDENCE = 0.25

# image size for YOLO inference
YOLO_IMGSZ = 640


# Visualization

# colors (BGR format for OpenCV)
COLOR_OPEN = (0, 255, 0)       # green: vacant and suitable
COLOR_TIGHT = (0, 200, 255)    # yellow/orange: vacant but tight
COLOR_TAKEN = (0, 0, 255)      # red: occupied
COLOR_LINES = (0, 255, 255)    # cyan: detected parking lines
COLOR_CENTROID = (255, 0, 255) # magenta: feature centroids
COLOR_YOLO_SPOT = (0, 255, 0)  # green: YOLO spot detections
COLOR_YOLO_CAR = (0, 0, 255)   # red: YOLO car detections

# overlay transparency (0.0 = invisible, 1.0 = opaque)
OVERLAY_ALPHA = 0.2

# font settings
FONT_SCALE = 0.5
FONT_THICKNESS = 2


# Video / demo

# output video codec
VIDEO_CODEC = "mp4v"

# fallback FPS if camera does not report
DEFAULT_FPS = 30.0

# dashboard label font scale
DASHBOARD_FONT_SCALE = 0.7


# Evaluation

# IoU threshold to match a geometric spot with a YOLO detection
IOU_MATCH_THRESHOLD = 0.3

# number of decimal places for metric reporting
METRIC_PRECISION = 4