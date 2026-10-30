"""DriveGuard AI Flask interface for driver drowsiness predictions."""
import base64
from functools import lru_cache
from io import BytesIO
import json
import os

import numpy as np
from flask import Flask, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError

# OpenCV is optional: it is only used for the face auto-crop (and exact resizing).
# If it is missing or broken (e.g. OpenCV 5.x has no CascadeClassifier), the app still works.
try:
    import cv2
    HAS_CV2 = hasattr(cv2, "resize")
    HAS_CASCADE = hasattr(cv2, "CascadeClassifier")
except Exception:
    cv2, HAS_CV2, HAS_CASCADE = None, False, False

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT, "cnn_best.keras")
META_PATH = os.path.join(ROOT, "model_meta.json")
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ----------------------------------------------------------------------------- helpers
@lru_cache(maxsize=1)
def load_model_and_meta():
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    meta = {"img_size": int(model.input_shape[1]), "class_names": ["Not Drowsy", "Drowsy"]}
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            meta.update(json.load(f))
    return model, meta


@lru_cache(maxsize=1)
def load_face_detector():
    """Returns the Haar face detector, or None if OpenCV cannot provide it."""
    if not HAS_CASCADE:
        return None
    try:
        det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        return None if det.empty() else det
    except Exception:
        return None


def read_image(file) -> np.ndarray:
    """Uploaded file -> RGB uint8 array (respects phone-photo rotation)."""
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    return np.array(img)


def crop_face(rgb: np.ndarray, detector):
    """Crop the largest detected face with a small margin. Returns (image, found_face)."""
    if detector is None:
        return rgb, False
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return rgb, False
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    m = int(0.15 * max(w, h))
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(rgb.shape[1], x + w + m), min(rgb.shape[0], y + h + m)
    return rgb[y0:y1, x0:x1], True


def preprocess(rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize to the network size. Scaling (x / 255) happens inside the model."""
    if HAS_CV2:
        img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)       # same as training
    else:
        img = np.array(Image.fromarray(rgb).resize((size, size), Image.BOX))     # close equivalent without OpenCV
    return img[None].astype("float32")


def predict_drowsy(model, batch: np.ndarray) -> float:
    return float(model.predict(batch, verbose=0).ravel()[0])


def image_data_uri(rgb: np.ndarray) -> str:
    image = Image.fromarray(rgb)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def accuracy_label(meta: dict) -> str:
    value = meta.get("test_accuracy")
    if value is None:
        return "Not recorded"
    try:
        accuracy = float(value)
    except (TypeError, ValueError):
        return "Not recorded"
    if 0 <= accuracy <= 1:
        return f"{accuracy:.1%}"
    if 1 < accuracy <= 100:
        return f"{accuracy:.1f}%"
    return "Not recorded"


def load_page_data():
    meta = {"img_size": "--", "best_variant": "CNN", "threshold": 0.5}
    error = None
    if not os.path.exists(MODEL_PATH):
        error = "Model file cnn_best.keras was not found beside app.py."
    else:
        try:
            _, meta = load_model_and_meta()
        except Exception as exc:
            error = f"The model could not be loaded: {exc}"
    return meta, error


@app.route("/", methods=["GET", "POST"])
def index():
    meta, model_error = load_page_data()
    prediction = None
    input_image = None
    model_image = None
    error = model_error

    if request.method == "POST" and not model_error:
        try:
            image_file = request.files.get("image")
            camera_data = request.form.get("camera_data", "")
            if image_file and image_file.filename:
                rgb = read_image(image_file.stream)
            elif camera_data.startswith("data:image/"):
                _, encoded = camera_data.split(",", 1)
                image_bytes = base64.b64decode(encoded, validate=True)
                rgb = read_image(BytesIO(image_bytes))
            else:
                raise ValueError("Choose an image or take a photo before analyzing.")

            auto_crop = request.form.get("auto_crop") == "on"
            threshold = float(request.form.get("threshold", meta.get("threshold", 0.5)))
            threshold = min(max(threshold, 0.05), 0.95)
            detector = load_face_detector() if auto_crop else None
            face, found = crop_face(rgb, detector) if auto_crop else (rgb, False)
            model, meta = load_model_and_meta()
            probability = predict_drowsy(model, preprocess(face, int(meta["img_size"])))
            drowsy = probability >= threshold

            prediction = {
                "drowsy": drowsy,
                "confidence": probability if drowsy else 1 - probability,
                "probability": probability,
                "threshold": threshold,
                "face_warning": "No face was detected; the full image was used." if auto_crop and detector and not found else None,
                "detector_warning": "Face detection is unavailable; the full image was used." if auto_crop and detector is None else None,
            }
            input_image = image_data_uri(rgb)
            model_image = image_data_uri(face)
        except (UnidentifiedImageError, OSError, ValueError, TypeError) as exc:
            error = str(exc) or "The selected file is not a supported image."

    f1 = meta.get("test_f1")
    if isinstance(f1, (int, float)):
        f1 = f"{float(f1):.1%}"
    else:
        f1 = ""

    return render_template(
        "index.html",
        meta=meta,
        accuracy=accuracy_label(meta),
        f1=f1,
        error=error,
        prediction=prediction,
        input_image=input_image,
        model_image=model_image,
    )


@app.errorhandler(413)
def file_too_large(_error):
    meta, model_error = load_page_data()
    return render_template(
        "index.html",
        meta=meta,
        accuracy=accuracy_label(meta),
        f1="Not recorded",
        error=model_error or "Image is too large. Maximum upload size is 10 MB.",
        prediction=None,
        input_image=None,
        model_image=None,
    ), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
