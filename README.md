# Creating a Virtual Environment
## 1) Create the virtual environment

From the project folder:

`python3 -m venv parking_env`

## 2) Activate the virtual environment

Linux or macOS:

`source parking_env/bin/activate`

Windows PowerShell:

`parking_env\Scripts\Activate.ps1`

Windows Command Prompt:

`parking_env\Scripts\activate.bat`

When activated, the shell should be preceded with `(parking_env)`

## 3) Upgrade pip

`python -m pip install --upgrade pip`

## 4) Install required packages

`pip install -r requirements.txt`

## 5) Verify installation

`python -c "import torch, torchvision, ultralytics, cv2, numpy, pandas, yaml, tqdm, scipy; from PIL import Image; import matplotlib; from matplotlib import pyplot as plt; print('All imports OK')"`

## 6) Deactivate when done

`deactivate`

## 7) Launch the environment later

Each new terminal session:

1. Open terminal in this project folder
2. Run: `source parking_env/bin/activate`
3. Run the Python scripts

# Getting the Data
## PKLot Data
The PKLot database is licensed under a Creative Commons Attribution 4.0 License and may be used provided you acknowledge the source by citing the [PKLot paper](http://www.inf.ufpr.br/lesoliveira/download/ESWA2015.pdf) in publications about your research:

```
Almeida, P., Oliveira, L. S., Silva Jr, E., Britto Jr, A., Koerich, A., PKLot – A robust dataset for parking lot classification, Expert Systems with Applications, 42(11):4937-4949, 2015.
```

The dataset was downloaded from https://public.roboflow.ai/object-detection/pklot, then moved to the `data` folder of the WSL environment and unzipped.

## CNRPark+EXT Data
[CNRPark+EXT](http://cnrpark.it/) has the data for this training run. [These are the labels](https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNRPark+EXT.csv) and [these are the full images](https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNR-EXT_FULL_IMAGE_1000x750.tar).

The CNRPark-EXT dataset is licensed under a [Open Data Commons Open Database License (ODbL) v1.0.](https://opendatacommons.org/licenses/odbl/1-0/)

The data from this set was moved to `data/CNRParkEXT` along with the labels.

## Merging the Data
From the root directory of the project, run `mkdir -p data/combined/images/{train,valid,test} data/combined/labels/{train,valid,test}` to create train/test/validate directories for the merged PKLot and CNRPark dataset.

Then copy images and labels from both prepared datasets into the combined directory with dataset-specific filename prefixes to avoid collisions:

```bash
shopt -s nullglob

for split in train valid test; do
	for ds in CNRParkEXT pklot; do
		src="data/$ds/yolo_ready"
		prefix="$(echo "$ds" | tr '[:upper:]' '[:lower:]')_"

		for img in "$src/images/$split"/*; do
			[ -f "$img" ] || continue
			base="$(basename "$img")"
			stem="${base%.*}"

			cp --update=none "$img" "data/combined/images/$split/${prefix}${base}"

			lbl="$src/labels/$split/${stem}.txt"
			if [ -f "$lbl" ]; then
				cp --update=none "$lbl" "data/combined/labels/$split/${prefix}${stem}.txt"
			fi
		done
	done
done
```

Create a dataset YAML for the merged directory:

```bash
cat > /root/project6/data/combined/parking_dataset.yaml << 'EOF'
path: /root/project6/data/combined
train: images/train
val: images/valid
test: images/test
names:
  0: spot
  1: car
EOF
```

To verify the merge counts, run:

```bash
for split in train valid test; do
	echo "split=$split images=$(find /root/project6/data/combined/images/$split -maxdepth 1 -type f | wc -l) labels=$(find /root/project6/data/combined/labels/$split -maxdepth 1 -type f | wc -l)"
done
```

# Running the YOLO Model for PKLot
Run `python youOnlyParkOncePKLot.py prepare` to parse the dataset into a YOLO11-compatible format.

Run `python youOnlyParkOncePKLot.py train --model yolo11n.pt --epochs <num_epochs>` to train the YOLO model on the dataset. You can pass `--device auto` to use the GPU if available, `--device cpu` to use the CPU, or `--device <number>` to specify a specific device.

Run `python youOnlyParkOncePKLot.py predict --weights runs/detect/runs/pklot/train/weights/best.pt --source data/pklot/test` to predict using the trained weights. `--conf` can be passed for a specific confidence level in predictions.

# Running the YOLO Model for CNRPark+EXT
Run `python youOnlyParkOnceCNRPark.py prepare` to parse the dataset into a YOLO11-compatible format.

Run `python youOnlyParkOnceCNRPark.py train --model yolo11n.pt --epochs <num_epochs>` to train the YOLO model on the dataset. You can pass `--device auto` to use the GPU if available, `--device cpu` to use the CPU, or `--device <number>` to specify a specific device.

Run `python youOnlyParkOnceCNRPark.py predict --weights runs/detect/runs/cnrpark/train/weights/best.pt --source data/CNRParkEXT/FULL_IMAGE_1000x750` to predict using the trained weights. `--conf` can be passed for a specific confidence level in predictions.

Run `python youOnlyParkOnceCNRPark.py evaluate --weights runs/detect/runs/cnrpark/train/weights/best.pt` to evaluate the model on the test split and save metrics/plots.

# Running the YOLO Model on Merged Data (PKLot + CNRPark+EXT)
Train directly with Ultralytics on the merged YAML:

```bash
yolo detect train \
	model=/root/project6/yolo11n.pt \
	data=/root/project6/data/combined/parking_dataset.yaml \
	epochs=20 \
	imgsz=640 \
	batch=16 \
	device=cpu \
	workers=0 \
	project=/root/project6/runs/detect/runs \
	name=combined_cnr_pklot \
	exist_ok=True
```

Evaluate the merged model on each original dataset:

```bash
yolo detect val \
	model=/root/project6/runs/detect/runs/combined_cnr_pklot/weights/best.pt \
	data=/root/project6/data/pklot/yolo_ready/parking_dataset.yaml \
	split=test \
	project=/root/project6/runs/cross_eval \
	name=combined_on_pklot \
	device=cpu \
    workers=0 \
	batch=8
```

```bash
yolo detect val \
	model=/root/project6/runs/detect/runs/combined_cnr_pklot/weights/best.pt \
	data=/root/project6/data/CNRParkEXT/yolo_ready/parking_dataset.yaml \
	split=test \
	project=/root/project6/runs/cross_eval \
	name=combined_on_cnr \
	device=cpu \
    workers=0 \
	batch=8
```

Evaluate the model on the test split of the combined dataset:
```bash
yolo detect val \
	model=/root/project6/runs/detect/runs/combined_cnr_pklot/weights/best.pt \
	data=/root/project6/data/combined/parking_dataset.yaml \
	split=test \
	project=/root/project6/runs/cross_eval \
	name=combined_on_combined \
	device=cpu \
	workers=0 \
	batch=8
```

Note: using `device=cpu` and `workers=0` is safer for WSL stability. If you have setup GPU in your environment, instead use `device=0`.