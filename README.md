# Handwritten Symbol Classifier — Assignment 2

Computer Vision Assignment 2 project.

The project continues Assignment 1 and classifies three handwritten mathematical symbols:

- equals (=)
- minus (-)
- plus (+)

## Model

The application uses EfficientNetV2B0 with ImageNet pretrained weights and transfer learning.

The model was trained on the same self-collected dataset used in Assignment 1:

- 100 equals images
- 100 minus images
- 100 plus images

Total: 300 images.

## Features

The web application supports:

- image upload
- webcam input
- symbol prediction
- confidence score
- Grad-CAM visual explanation
- inference latency measurement

## Files

- `Assignment2.ipynb` — training, experiments and evaluation
- `app.py` — Gradio web application
- `efficientnetv2_symbol_classifier.keras` — trained model
- `requirements.txt` — required Python packages

## Installation

Install the required packages:

```bash
pip install -r requirements.txt