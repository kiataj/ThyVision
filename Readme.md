# Thyroid Classifier: Web-based 3D Inference for micro-CT images of thyroid cancer
This project is a lightweight web application for running patchwise inference of diagnostics and BRAF V600E mutation on 3D volumetric micro-CT scans of thyroid tumors.

## Features
- **Patch-based Inference** on 3D micro-CT images.
- **Multi-task Prediction:** BRAF V600E mutation status and diagnostic classification.
- **TIFF Export** of prediction maps.
- **End-to-End Pipeline** from upload → inference → export
---

## Getting Started

### 1. Clone Repository

<pre>
git clone https://github.com/kiataj/ThyVision.git
cd ThyVision
</pre>

### 2. Build the Docker Image

<pre>
docker build -t thyvision .
</pre>

## Run the Docker Container
<pre>
docker run -p 8000:8000 -v C:\User\app:/data thyvision 
</pre>
You can replace `C:\User\app` with a local derive of your preference for saving the predictions.

## Launch Frontend
Once the Docker container is running, the web interface will be available at: <br>
[http://localhost:8000/](http://localhost:8000/)

### Usage Workflow

1. **Upload** a 3D `.tiff` volume.
2. **Configure** patch and stride size.
3. **Click** `Run Inference` - wait until the buttons are activated again, the progress bar can be seen in the command prompt that the docker container is running in.
4. **Click** `Save Predictions` to save the predictions in the mounted derive.

<pre>
#### Directory structure
├── main.py                   # FastAPI backend
├── inference_module.py       # Inference logic (with tqdm)
├── inference-ui/             # React frontend (Vite)
└── Input example/            # An input example .tif file you can use to run inference.
</pre>

