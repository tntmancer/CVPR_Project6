# Creating a Virtual Environment
## 1) Create the virtual environment

From the project folder:

`python3 -m venv .venv`

## 2) Activate the virtual environment

Linux or macOS:

`source .venv/bin/activate`

Windows PowerShell:

`.venv\Scripts\Activate.ps1`

Windows Command Prompt:

`.venv\Scripts\activate.bat`

When activated, the shell should be preceded with `(.venv)`

## 3) Upgrade pip

`python -m pip install --upgrade pip`

## 4) Install required packages

`pip install -r requirements.txt`

## 5) Verify installation

`python -c "import torch, torchvision, matplotlib, cv2; from matplotlib import pyplot as plt; print('All imports OK')"`

## 6) Deactivate when done

`deactivate`

## 7) Launch the environment later

Each new terminal session:

1. Open terminal in this project folder
2. Run: `source .venv/bin/activate`
3. Run the Python scripts

# Getting the Data
The PKLot database is licensed under a Creative Commons Attribution 4.0 License and may be used provided you acknowledge the source by citing the [PKLot paper](http://www.inf.ufpr.br/lesoliveira/download/ESWA2015.pdf) in publications about your research:

```
Almeida, P., Oliveira, L. S., Silva Jr, E., Britto Jr, A., Koerich, A., PKLot – A robust dataset for parking lot classification, Expert Systems with Applications, 42(11):4937-4949, 2015.
```

The dataset was downloaded from https://public.roboflow.ai/object-detection/pklot, then moved to the `data` folder of the WSL environment and unzipped.

# Running the YOLO Model
Run `python youOnlyParkOnce.py prepare` to parse the dataset into a YOLO11-compatible format.

Run `python youOnlyParkOnce.py train --model yolo11n.pt --epochs <num_epochs>` to train the YOLO model on the dataset. You can pass `--device auto` to use the GPU if available, `--device cpu` to use the CPU, or `--device <number>` to specify a specific device.

Run `python youOnlyParkOnce.py predict --weights runs/detect/runs/yopo/train/weights/best.pt --source data/test` to predict using the trained weights.