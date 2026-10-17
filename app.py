import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
MODEL_PATH = BASE_DIR / "model" / "freshness_model.keras"
LABELS_PATH = BASE_DIR / "model" / "class_names.json"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
IMAGE_SIZE = (224, 224)

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-for-local-development"
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_model = None
_class_names = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model():
    global _model, _class_names
    if _model is None:
        if not MODEL_PATH.exists() or not LABELS_PATH.exists():
            raise FileNotFoundError(
                "The trained model is not available yet. Add a dataset and run: python train_model.py"
            )
        _model = load_model(MODEL_PATH)
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            _class_names = json.load(f)
    return _model, _class_names


def preprocess_image(path):
    # OpenCV loads the image as BGR; convert to RGB before passing it to the model.
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("The uploaded file could not be read as an image.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files.get("image")
        if uploaded_file is None or uploaded_file.filename == "":
            flash("Please choose an image first.", "error")
            return redirect(url_for("index"))
        if not allowed_file(uploaded_file.filename):
            flash("Please upload a JPG, JPEG, PNG, or WEBP image.", "error")
            return redirect(url_for("index"))

        safe_name = secure_filename(uploaded_file.filename)
        # Give each upload a unique name to avoid accidental overwrites.
        import uuid
        saved_name = f"{uuid.uuid4().hex}_{safe_name}"
        image_path = UPLOAD_DIR / saved_name
        uploaded_file.save(image_path)

        try:
            model, class_names = get_model()
            processed = preprocess_image(image_path)
            probabilities = model.predict(processed, verbose=0)[0]
            best_index = int(np.argmax(probabilities))
            label = str(class_names[best_index])
            confidence = float(probabilities[best_index]) * 100
            return render_template(
                "result.html",
                image_url=url_for("static", filename=f"uploads/{saved_name}"),
                label=label.replace("_", " ").title(),
                confidence=f"{confidence:.1f}",
                probabilities=sorted(
                    [
                        {"label": str(name).replace("_", " ").title(),
                         "confidence": f"{float(prob) * 100:.1f}"}
                        for name, prob in zip(class_names, probabilities)
                    ],
                    key=lambda item: float(item["confidence"]),
                    reverse=True,
                ),
                note="This is an experimental image-based estimate, not a food-safety guarantee."
            )
        except (FileNotFoundError, ValueError) as exc:
            image_path.unlink(missing_ok=True)
            flash(str(exc), "error")
            return redirect(url_for("index"))
        except Exception:
            image_path.unlink(missing_ok=True)
            app.logger.exception("Prediction failed")
            flash("Prediction failed. Check that the model was trained correctly.", "error")
            return redirect(url_for("index"))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
