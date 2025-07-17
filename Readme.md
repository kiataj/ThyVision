# Thyroid Classifier: Web-based 3D Inference & Visualization

This project is a lightweight web application for running patchwise inference on 3D volumetric medical images (e.g., micro-CT scans of thyroid tumors), with optional TIFF export of predictions, and interactive Napari visualization.

## Features

- **Patch-based Inference** with MONAI and PyTorch
- **TIFF Export** of multi-task prediction maps
- **Interactive 3D Visualization** via Napari
- **End-to-End Pipeline** from upload → inference → visualization → export
- **MLflow Integration** for model versioning

---

## Getting Started

### 1. Clone Repository

<pre>git clone https://github.com/kiataj/thyroid-classifier.git
cd thyroid-classifier</pre>


### 2. Backend Setuo (FastAPI)

## Prerequisites
- Python 3.8+
- Anaconda (recommended)
- MLflow, MONAI, PyTorch, FastAPI, etc.

<pre>conda create -n thyroidenv python=3.10
conda activate thyroidenv
pip install -r requirements.txt</pre>

## Run MLflow Server
<pre>mlflow server --backend-store-uri ./mlruns --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000</pre>

## Launch FastAPI Backemd
<pre>python main.py</pre>

### Frontend Setup (React)
<pre>
cd fromtend
npm install
npm run dev
</pre>
App runs on: http://localhost:5173/

### Usage Workflow

1. **Upload** a 3D `.tiff` volume (e.g., micro-CT scan).
2. **Configure** patch and stride size.
3. **Click** `Run Inference` – progress will be shown.
4. **Click** `Launch Napari` to explore prediction maps.
5. **Enter** a save directory and click `Save Predictions` to export prediction maps as `uint8 .tiff` files.



#### Directory structure
├── main.py                   # FastAPI backend
├── inference_module.py       # Inference logic (with tqdm)
├── napari_viewer.py          # Napari UI
├── uploads/                  # Temporary data storage
├── frontend/                 # React frontend (Vite)
└── mlruns/                   # MLflow tracking directory

