# PAN Card Field Detection using YOLO

## Overview

This project uses **YOLO (Ultralytics)** to detect important fields from a PAN card image such as PAN number, name, father's name, and date of birth.
The model is trained on a custom dataset and outputs bounding boxes around detected fields.

---

## Project Structure

```
YOLO
│
├── dataset
│   ├── images
│   │   ├── train
│   │   └── val
│   └── labels
│       ├── train
│       └── val
│
├── data.yaml
├── train.py
├── detect.py
└── runs/detect/train/weights/best.pt
```

---

## Requirements

Install dependencies:

```
pip install -r requirement.txt
```

---

## Dataset Format

```
dataset
├── images/train
├── images/val
├── labels/train
└── labels/val
```

Label format (YOLO):

```
class_id x_center y_center width height
```

---

## Train Model

```
python train.py
```

Trained model will be saved in:

```
runs/detect/train/weights/best.pt
```

---

## Run Detection

```
python detect.py
```

The model will detect PAN card fields and display bounding boxes on the image.

---
