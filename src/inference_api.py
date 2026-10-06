import io
import json
import os
from pathlib import Path
from typing import Annotated

import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image


DEFAULT_METADATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "checkpoints"
    / "fashion_model.metadata.json"
)
METADATA_PATH = Path(
    os.environ.get("MODEL_METADATA_PATH", str(DEFAULT_METADATA_PATH))
)
TRITON_URL = os.environ.get("TRITON_URL", "http://localhost:8000").rstrip("/")
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", 10 * 1024 * 1024))

if not METADATA_PATH.is_file():
    raise FileNotFoundError(f"Model metadata not found: {METADATA_PATH}")

with METADATA_PATH.open(encoding="utf-8") as metadata_file:
    MODEL_METADATA = json.load(metadata_file)

INPUT_SIZE = int(MODEL_METADATA["input"]["shape"][-1])
NORMALIZATION_MEAN = np.asarray(
    MODEL_METADATA["input"]["normalization"]["mean"], dtype=np.float32
).reshape(1, 1, 3)
NORMALIZATION_STD = np.asarray(
    MODEL_METADATA["input"]["normalization"]["std"], dtype=np.float32
).reshape(1, 1, 3)
OUTPUTS = MODEL_METADATA["outputs"]

app = FastAPI(
    title="Fashion Merchandising Inference API",
    description="Upload a product image to get fashion attribute predictions.",
    version="0.1.0",
)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


def _preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.width * source.height > 40_000_000:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions are too large.",
                )
            image = source.convert("RGB").resize(
                (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR
            )
    except OSError as error:
        raise HTTPException(
            status_code=400,
            detail="Upload a valid JPEG, PNG, or other supported image.",
        ) from error

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    pixels = (pixels - NORMALIZATION_MEAN) / NORMALIZATION_STD
    return np.transpose(pixels, (2, 0, 1))[np.newaxis, ...].astype(
        np.float32, copy=False
    )


def _decode_prediction(output_data: list, output_name: str) -> dict:
    logits = np.asarray(output_data, dtype=np.float32).reshape(-1)
    shifted_logits = logits - np.max(logits)
    probabilities = np.exp(shifted_logits)
    probabilities /= probabilities.sum()
    class_index = int(np.argmax(probabilities))

    output_metadata = OUTPUTS[output_name]
    classes = output_metadata["classes"]
    label = classes[class_index] if classes is not None else None
    prediction = {
        "label": label,
        "class_index": class_index,
        "confidence": round(float(probabilities[class_index]), 4),
    }
    if label is None:
        prediction["label_mapping_status"] = output_metadata[
            "label_mapping_status"
        ]
    return prediction


@app.get("/health/live")
def liveness():
    return {"status": "live"}


@app.get(
    "/health/ready",
    responses={503: {"description": "Triton model is not ready."}},
)
def readiness():
    try:
        response = httpx.get(
            f"{TRITON_URL}/v2/models/fashion_model/ready", timeout=2.0
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=503, detail="Triton model is not ready."
        ) from error
    return {"status": "ready", "model": "fashion_model"}


@app.post(
    "/predict",
    responses={
        400: {"description": "The upload is empty or is not a valid image."},
        413: {"description": "The uploaded image exceeds an allowed size limit."},
        502: {"description": "Triton returned an error or invalid response."},
        503: {"description": "The Triton inference server is unavailable."},
    },
)
def predict(file: Annotated[UploadFile, File(...)]):
    image_bytes = file.file.read(MAX_IMAGE_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {MAX_IMAGE_BYTES}-byte upload limit.",
        )

    image_tensor = _preprocess_image(image_bytes)
    payload = {
        "inputs": [
            {
                "name": "INPUT__0",
                "shape": list(image_tensor.shape),
                "datatype": "FP32",
                "data": image_tensor.reshape(-1).tolist(),
            }
        ],
        "outputs": [
            {"name": f"OUTPUT__{index}"} for index in range(len(OUTPUTS))
        ],
    }

    try:
        response = httpx.post(
            f"{TRITON_URL}/v2/models/fashion_model/infer",
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()
    except httpx.RequestError as error:
        raise HTTPException(
            status_code=503, detail="Could not reach the Triton inference server."
        ) from error
    except httpx.HTTPStatusError as error:
        raise HTTPException(
            status_code=502,
            detail=f"Triton inference failed: {error.response.text[:500]}",
        ) from error

    try:
        response_outputs = response.json()["outputs"]
        predictions = {
            name: _decode_prediction(
                response_outputs[metadata["tuple_index"]]["data"], name
            )
            for name, metadata in OUTPUTS.items()
        }
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=502,
            detail="Triton returned an invalid prediction response.",
        ) from error

    return {"model": "fashion_model", "predictions": predictions}
