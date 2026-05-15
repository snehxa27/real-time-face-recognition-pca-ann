"""
predict.py — Face Prediction Module
=====================================
Loads saved model, PCA, and label encoder to predict the identity
of a new face image.

Features:
  - Single-image prediction with confidence score
  - Batch prediction from a folder
  - Unknown-face detection (confidence threshold)
  - Saves annotated prediction image

Usage:
    python predict.py --image path/to/face.jpg
    python predict.py --folder path/to/test_folder/
    python predict.py --image face.jpg --threshold 0.55

Author  : PCA-ANN Face Recognition Project
"""

import os
import sys
import pickle
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from utils import preprocess_single_image, display_prediction

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
MODELS_DIR      = "models"
OUTPUTS_DIR     = "outputs"
MODEL_PATH      = os.path.join(MODELS_DIR, "ann_model.keras")
PCA_PATH        = os.path.join(MODELS_DIR, "pca.pkl")
ENCODER_PATH    = os.path.join(MODELS_DIR, "label_encoder.pkl")

# Confidence below this → label as "Unknown"
DEFAULT_THRESHOLD = 0.50


# ─────────────────────────────────────────────
# 1. Load saved artefacts
# ─────────────────────────────────────────────
def load_model_and_artefacts():
    """Load ANN model, PCA transformer, and label encoder from disk."""
    if not all(os.path.exists(p) for p in [MODEL_PATH, PCA_PATH, ENCODER_PATH]):
        sys.exit(
            "[ERROR] Trained model not found.\n"
            "  Please run  python train_model.py  first."
        )

    print("[INFO] Loading model and artefacts …")
    model = keras.models.load_model(MODEL_PATH)

    with open(PCA_PATH, "rb") as f:
        pca = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)

    print(f"[INFO] Model loaded   : {MODEL_PATH}")
    print(f"[INFO] Classes known  : {list(le.classes_)}")
    return model, pca, le


# ─────────────────────────────────────────────
# 2. Core prediction function
# ─────────────────────────────────────────────
def predict_face(img_path: str, model, pca, le,
                 threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Predict the identity of a face in *img_path*.

    Returns
    -------
    result : dict with keys
        label       : predicted person name (or "Unknown")
        confidence  : float in [0, 1]
        all_probs   : dict {class_name: probability}
        is_known    : bool
    """
    # Pre-process image  →  1-D vector  →  PCA space
    x_flat = preprocess_single_image(img_path)  # shape (1, n_features)
    x_pca  = pca.transform(x_flat)              # shape (1, n_components)

    # ANN prediction
    probs   = model.predict(x_pca, verbose=0)[0]   # shape (n_classes,)
    pred_id = int(np.argmax(probs))
    conf    = float(probs[pred_id])

    label    = le.classes_[pred_id] if conf >= threshold else "Unknown"
    is_known = conf >= threshold

    all_probs = {le.classes_[i]: float(probs[i]) for i in range(len(le.classes_))}

    return {
        "label":      label,
        "confidence": conf,
        "all_probs":  all_probs,
        "is_known":   is_known,
    }


# ─────────────────────────────────────────────
# 3. CLI helpers
# ─────────────────────────────────────────────
def predict_single(img_path: str, threshold: float):
    """Predict one image, print result, and save annotated output."""
    model, pca, le = load_model_and_artefacts()

    result = predict_face(img_path, model, pca, le, threshold)

    print(f"\n{'='*50}")
    print(f"  Image     : {img_path}")
    print(f"  Predicted : {result['label']}")
    print(f"  Confidence: {result['confidence']*100:.1f}%")
    print(f"  Known face: {result['is_known']}")
    print(f"\n  Top-3 probabilities:")
    sorted_probs = sorted(result["all_probs"].items(),
                           key=lambda x: x[1], reverse=True)[:3]
    for name, prob in sorted_probs:
        bar = "█" * int(prob * 30)
        print(f"    {name:30s} {prob*100:5.1f}%  {bar}")
    print(f"{'='*50}")

    # Save annotated image
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUTS_DIR, "prediction_result.png")
    display_prediction(img_path, result["label"], result["confidence"],
                       save_path=save_path)
    return result


def predict_folder(folder_path: str, threshold: float):
    """Run prediction on every image in a folder and print a summary table."""
    model, pca, le = load_model_and_artefacts()

    extensions = (".jpg", ".jpeg", ".png", ".bmp")
    img_files  = [f for f in os.listdir(folder_path)
                  if f.lower().endswith(extensions)]

    if not img_files:
        print(f"[WARN] No images found in {folder_path}")
        return

    print(f"\n[INFO] Predicting {len(img_files)} images from {folder_path}\n")
    print(f"{'File':<30} {'Prediction':<25} {'Confidence':>10}  {'Known'}")
    print("-" * 75)

    results = []
    for fname in sorted(img_files):
        img_path = os.path.join(folder_path, fname)
        try:
            r = predict_face(img_path, model, pca, le, threshold)
            print(f"{fname:<30} {r['label']:<25} {r['confidence']*100:>8.1f}%   "
                  f"{'Yes' if r['is_known'] else 'No'}")
            results.append(r)
        except Exception as e:
            print(f"{fname:<30}  [ERROR] {e}")

    known_count = sum(1 for r in results if r["is_known"])
    print(f"\n[SUMMARY] {known_count}/{len(results)} faces identified above threshold.")


# ─────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCA+ANN Face Prediction")
    parser.add_argument("--image",     type=str, default="",
                        help="Path to a single image for prediction")
    parser.add_argument("--folder",    type=str, default="",
                        help="Path to a folder of images")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Confidence threshold for 'Unknown' (default {DEFAULT_THRESHOLD})")
    args = parser.parse_args()

    if args.image:
        if not os.path.isfile(args.image):
            sys.exit(f"[ERROR] File not found: {args.image}")
        predict_single(args.image, args.threshold)

    elif args.folder:
        if not os.path.isdir(args.folder):
            sys.exit(f"[ERROR] Folder not found: {args.folder}")
        predict_folder(args.folder, args.threshold)

    else:
        parser.print_help()
        sys.exit("\n[ERROR] Please provide --image or --folder")
