# YOLOv8 Custom Object Detection Training

This repository provides a complete workflow for training **YOLOv8** (by Ultralytics) on custom datasets. It includes a Jupyter Notebook designed to guide you through environment setup, data preparation, model training, and inference using the latest state-of-the-art YOLO architecture.

## 📄 Repository Contents

* `yolov8_custom_training.ipynb`: The primary notebook that orchestrates the training pipeline:
    * **Setup**: Installing the `ultralytics` package.
    * **Data Handling**: Configuring the `data.yaml` file for custom classes.
    * **Training**: Fine-tuning a pre-trained model (e.g., `yolov8n.pt`, `yolov8s.pt`) on your data.
    * **Validation**: Evaluating model performance (mAP, confusion matrix).
    * **Inference**: Running the trained model on new images/videos.

## 🛠️ Prerequisites

To run this project, you need the following:

* [Python 3.8+](https://www.python.org/downloads/)
* **PyTorch** (with CUDA support recommended for GPU acceleration)
* **Ultralytics** library

### Installation
If running locally, install the dependencies:

```bash
pip install ultralytics notebook
```

## 🚀 Usage Guide
Option A: Google Colab (Recommended)
Upload yolov8_custom_training.ipynb to Google Colab
.

Enable GPU: Go to Runtime > Change runtime type > T4 GPU.

Follow the cells to mount your dataset (e.g., from Google Drive) and start training.

Option B: Local Training
Clone the repository:

```bash
git clone [https://github.com/felixyustian/yolo_v8_test.git](https://github.com/felixyustian/yolo_v8_test.git)
cd yolo_v8_test
```

Launch Jupyter:
```bash
jupyter notebook yolov8_custom_training.ipynb
```

## 📊 Training Workflow
YOLOv8 uses a simpler CLI/Python API compared to previous versions. The notebook generally covers these steps:

Initialize Model:

Python
```bash
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (nano version)
Train:

Python
# Train the model
results = model.train(data="data.yaml", epochs=100, imgsz=640)
Validate & Predict:

Python
# Validate
model.val()

# Predict on a new image
model.predict("path/to/image.jpg", save=True)
```

## 📁 Dataset Structure
Ensure your dataset is organized as follows for YOLOv8:

dataset/
├── data.yaml  # Configuration file
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
data.yaml 

Example:
YAML
path: ../dataset  # dataset root dir
train: train/images
val: valid/images

names:
  0: person
  1: bicycle
  2: car
  
## 📄 License
This project is licensed under the GPL-3.0 License. See the LICENSE file for details.

```bash
Would you like me to generate a specific `data.yaml` template file to go along with this README?

For a visual walkthrough on how to set this up, you might find this [YOLOv8 Custom Dataset Tutorial](https://www.youtube.com/watch?v=-QRVPDjfCYc) helpful as it covers the end-to-end process from data preparation to training.

http://googleusercontent.com/youtube_content/0
```
