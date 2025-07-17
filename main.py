from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

import numpy as np
import tifffile as tiff
import io, os
from pathlib import Path
from shutil import copy2
import subprocess
import zarr
from numcodecs import Blosc
from typing import Dict
import traceback
import logging
import sys

import torch
import mlflow.pyfunc
from huggingface_hub import snapshot_download
from inference_module import infer_patchwise

# --------------------------------------------------------------
# Logging
# --------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.info("🚀 Logging initialized from main.py")

app = FastAPI()

# --------------------------------------------------------------
# FastAPI + CORS
# --------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------
# Download model at runtime (once)
# --------------------------------------------------------------
local_dir = Path("/app/hf_model")
local_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id="HippoCanFly/ct-thyroid-classifier",
    repo_type="model",
    local_dir=str(local_dir),
    token="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",
)

# --------------------------------------------------------------
# Create Windows-style alias files expected by pickled model
# --------------------------------------------------------------
artifacts_dir = local_dir / "artifacts"
for name in ["encoder_state.pt", "classifier_state.pt"]:
    src = artifacts_dir / name
    dst = local_dir / f"artifacts\\{name}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists() and not dst.exists():
        try:
            os.link(src, dst)
        except OSError:
            copy2(src, dst)

# --------------------------------------------------------------
# Patch torch.load exactly once (CPU-safe)
# --------------------------------------------------------------
if not getattr(torch.load, "__patched_for_cpu__", False):
    _orig_torch_load = torch.load

    def _cpu_safe_load(*args, **kwargs):
        kwargs["map_location"] = torch.device("cpu")
        return _orig_torch_load(*args, **kwargs)

    _cpu_safe_load.__patched_for_cpu__ = True
    torch.load = _cpu_safe_load

# --------------------------------------------------------------
# Load MLflow PyFunc model
# --------------------------------------------------------------
model = mlflow.pyfunc.load_model(str(local_dir))
encoder = model._model_impl.python_model.encoder
classifier = model._model_impl.python_model.classifier
task_list = list(classifier.heads.keys())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on device: {device}")

# --------------------------------------------------------------
# I/O paths
# --------------------------------------------------------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
ZARR_PATH = UPLOAD_DIR / "predictions.zarr"
logger.info(f"📁 UPLOAD_DIR exists: {UPLOAD_DIR.resolve()}")


@app.get("/ping")
def ping():
    logger.info("✅ Ping route hit")
    return {"message": "pong"}

@app.get("/debug-uploads")
def debug_uploads():
    path = Path("/app/uploads")
    return {
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "contents": [p.name for p in path.glob("*")]
    }


#-------------------------------------#
#Inference endpoint
#-------------------------------------#

@app.post("/run-inference/")
async def run_inference(
    file: UploadFile = File(...),
    patch_d: int = Form(...),
    patch_h: int = Form(...),
    patch_w: int = Form(...),
    stride_d: int = Form(...),
    stride_h: int = Form(...),
    stride_w: int = Form(...)
):
    logger.info("/run-inference/ called")

    try:
        logger.info("⏳ Reading and normalizing uploaded volume...")
        file_data = await file.read()
        vol = tiff.imread(io.BytesIO(file_data)).astype(np.float32)
        vol = (vol - vol.min()) / (vol.max() - vol.min())
        logger.info(f"✅ Volume loaded with shape: {vol.shape}")

        logger.info("💾 Saving normalized volume to disk...")
        np.save(UPLOAD_DIR / "volume.npy", vol)

        logger.info("🚀 Running patch-based inference...")
        prob_maps = infer_patchwise(
            encoder, classifier, vol,
            patch_size=(patch_d, patch_h, patch_w),
            stride=(stride_d, stride_h, stride_w),
            task_list=task_list, device=device
        )
        logger.info(f"✅ Inference completed for tasks: {list(prob_maps.keys())}")

        logger.info("💾 Writing Zarr output...")
        root = zarr.open_group(str(ZARR_PATH), mode="w")
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
        for task in task_list:
            logger.info(f"   - Saving task: {task}, shape: {prob_maps[task].shape}")
            root.create_dataset(name=task, data=prob_maps[task], compressor=compressor, overwrite=True)

        logger.info("✅ Zarr saved successfully.")
        return {"status": "done", "zarr_path": str(ZARR_PATH)}

    except Exception as e:
        tb = traceback.format_exc()
        logger.error("❌ Exception during inference:\n" + tb)
        return {"status": "error", "message": str(e)}


# --------------------------------------------------------------
# Save predictions as TIFFs
# --------------------------------------------------------------
@app.post("/save-predictions/")
async def save_predictions():
    try:
        save_dir = Path("/data")  # Docker-mounted folder
        fname = "prediction"

        save_dir.mkdir(parents=True, exist_ok=True)
        z = zarr.open_group(str(ZARR_PATH), mode="r")

        for key in z.keys():
            prob_uint8 = (z[key][:] * 255).clip(0, 255).astype(np.uint8)
            tiff.imwrite(str(save_dir / f"{fname}_{key}.tiff"), prob_uint8)

        logger.info(f"✅ Saved predictions to {save_dir}")
        return {"status": "saved", "path": str(save_dir)}
    except Exception as e:
        logger.error(f"❌ Failed to save predictions: {e}")
        return {"status": "error", "message": str(e)}

# --------------------------------------------------------------
# Serve React build
# --------------------------------------------------------------
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# --------------------------------------------------------------
# Entry-point (only when run directly)
# --------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
