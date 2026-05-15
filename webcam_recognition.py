"""
webcam_recognition.py — Real-Time Webcam Face Recognition
===========================================================
Uses OpenCV to capture frames from your webcam, detect faces
using Haar Cascade, then classify each face with PCA + ANN.

Controls:
    Q  or  ESC  →  quit
    S           →  save current frame to outputs/

Run:
    python webcam_recognition.py
    python webcam_recognition.py --threshold 0.6

Author  : PCA-ANN Face Recognition Project
"""

import os
import sys
import pickle
import argparse
import datetime
import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras

from utils import IMG_HEIGHT, IMG_WIDTH

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
MODELS_DIR   = "models"
OUTPUTS_DIR  = "outputs"
MODEL_PATH   = os.path.join(MODELS_DIR, "ann_model.keras")
PCA_PATH     = os.path.join(MODELS_DIR, "pca.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# Haar cascade — ships with OpenCV
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# ─────────────────────────────────────────────
# Load artefacts
# ─────────────────────────────────────────────
def load_artefacts():
    if not all(os.path.exists(p) for p in [MODEL_PATH, PCA_PATH, ENCODER_PATH]):
        sys.exit("[ERROR] Run  python train_model.py  first.")

    model = keras.models.load_model(MODEL_PATH)
    with open(PCA_PATH,     "rb") as f: pca = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f: le  = pickle.load(f)
    return model, pca, le


# ─────────────────────────────────────────────
# Predict a face ROI
# ─────────────────────────────────────────────
def classify_face_roi(roi_gray, model, pca, le, threshold: float):
    """
    Pre-process a face region-of-interest (ROI) and predict its identity.
    Uses thresholding to detect 'Unknown' faces.
    """
    # Resize and normalize
    resized = cv2.resize(roi_gray, (IMG_WIDTH, IMG_HEIGHT))
    norm    = resized.flatten().astype(np.float32) / 255.0
    
    # PCA Transform
    x_pca   = pca.transform(norm.reshape(1, -1))
    
    # ── ANN Prediction ───────────────────────────────────────────────────
    # We use model.predict() to get the softmax probabilities
    probs   = model.predict(x_pca, verbose=0)[0]
    
    # Find the class with the highest probability
    pred_id = int(np.argmax(probs))
    conf    = float(np.max(probs)) # Maximum confidence score
    
    # ── Unknown Detection Logic ──────────────────────────────────────────
    if conf >= threshold:
        label = le.classes_[pred_id]
    else:
        label = "Unknown"

    return label, conf


# ─────────────────────────────────────────────
# Draw overlay on frame (Professional UI)
# ─────────────────────────────────────────────
def draw_overlay(frame, x, y, w, h, label: str, conf: float):
    """Draw professional bounding box and label overlay."""
    # Green for known faces, Red for unknown/low confidence
    is_unknown = (label == "Unknown")
    color = (0, 30, 220) if is_unknown else (0, 200, 0)
    
    # ── 1. Cornered Bounding Box ───────────────────────────────────────
    # Main rectangle
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 1)
    
    # Thicker corners for a "premium" feel
    len_c = int(w * 0.15)
    # Top-Left
    cv2.line(frame, (x, y), (x + len_c, y), color, 3)
    cv2.line(frame, (x, y), (x, y + len_c), color, 3)
    # Top-Right
    cv2.line(frame, (x+w, y), (x+w - len_c, y), color, 3)
    cv2.line(frame, (x+w, y), (x+w, y + len_c), color, 3)
    # Bottom-Left
    cv2.line(frame, (x, y+h), (x + len_c, y+h), color, 3)
    cv2.line(frame, (x, y+h), (x, y+h - len_c), color, 3)
    # Bottom-Right
    cv2.line(frame, (x+w, y+h), (x+w - len_c, y+h), color, 3)
    cv2.line(frame, (x+w, y+h), (x+w, y+h - len_c), color, 3)

    # ── 2. Floating Label ──────────────────────────────────────────────
    text = f"{label} {conf*100:.1f}%"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.6
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    
    # Label Background (semi-transparent effect)
    cv2.rectangle(frame, (x, y-th-15), (x+tw+10, y), color, -1)
    # Label Text
    cv2.putText(frame, text, (x+5, y-10), font, scale, (255, 255, 255), thick, cv2.LINE_AA)


# ─────────────────────────────────────────────
# Main webcam loop
# ─────────────────────────────────────────────
def run_webcam(camera_id: int = 0, threshold: float = 0.50):
    """
    Open webcam, detect faces, and classify them in real time.

    Parameters
    ----------
    camera_id : 0 for default webcam
    threshold : confidence below this → "Unknown"
    """
    print("[WEBCAM] Loading model …")
    model, pca, le = load_artefacts()

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        sys.exit("[ERROR] Haar cascade not found. Reinstall OpenCV.")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera {camera_id}.")

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("[WEBCAM] Running — press Q or ESC to quit, S to save frame.")
    fps_time = cv2.getTickCount()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed.")
            break

        # ── FPS calculation ───────────────────────────────────────────────
        t_now = cv2.getTickCount()
        fps   = cv2.getTickFrequency() / (t_now - fps_time)
        fps_time = t_now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Face detection ────────────────────────────────────────────────
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        # ── Classify each detected face ───────────────────────────────────
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            try:
                label, conf = classify_face_roi(roi_gray, model, pca, le, threshold)
            except Exception:
                label, conf = "Error", 0.0
            draw_overlay(frame, x, y, w, h, label, conf)

        # ── HUD ───────────────────────────────────────────────────────────
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "Q/ESC: Quit | S: Save", (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow("PCA + ANN Face Recognition  [LIVE]", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):    # Q or ESC
            break
        elif key == ord("s"):        # Save frame
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(OUTPUTS_DIR, f"webcam_{ts}.jpg")
            cv2.imwrite(path, frame)
            print(f"[WEBCAM] Frame saved → {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("[WEBCAM] Stopped.")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time webcam face recognition")
    parser.add_argument("--camera",    type=int,   default=0,
                        help="Camera device ID (default 0)")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Confidence threshold for unknown detection")
    args = parser.parse_args()
    run_webcam(camera_id=args.camera, threshold=args.threshold)
