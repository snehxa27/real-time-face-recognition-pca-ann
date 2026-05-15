"""
utils.py — Utility Functions
============================
Helper functions shared across all modules:
  - Dataset loading & augmentation
  - Image preprocessing pipeline
  - Visualization helpers
  - Label encoding / decoding

Author  : PCA-ANN Face Recognition Project
"""

import os
import cv2
import numpy as np
import zipfile
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# Constants — tweak these to change image size
# ─────────────────────────────────────────────
IMG_HEIGHT = 128   # pixels
IMG_WIDTH  = 128   # pixels
IMG_SIZE   = (IMG_WIDTH, IMG_HEIGHT)

# Haar Cascade for face detection
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = None

def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    return _face_cascade


# ─────────────────────────────────────────────
# 1. Dataset Extraction
# ─────────────────────────────────────────────
def extract_zip(zip_path: str, extract_to: str = "dataset") -> str:
    """
    Automatically extract a ZIP file that contains face folders.

    Expected layout inside the ZIP (nested ZIPs are supported):
        dataset/faces/PersonName/image.jpg

    Returns the path to the root folder that contains person sub-folders.
    """
    print(f"[INFO] Extracting dataset from: {zip_path}")
    os.makedirs(extract_to, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)

    # ── look for nested dataset.zip ──────────────────────────────────────
    for root, _, files in os.walk(extract_to):
        for f in files:
            if f.lower() == "dataset.zip":
                inner_zip = os.path.join(root, f)
                inner_out  = os.path.join(root, "inner_dataset")
                print(f"[INFO] Found inner zip: {inner_zip} — extracting …")
                with zipfile.ZipFile(inner_zip, "r") as z2:
                    z2.extractall(inner_out)

    # ── search for the folder that contains per-person sub-folders ───────
    faces_root = _find_faces_root(extract_to)
    if faces_root is None:
        raise FileNotFoundError(
            "Could not locate a 'faces' folder inside the ZIP. "
            "Ensure the ZIP contains  dataset/faces/<PersonName>/image.jpg"
        )

    print(f"[INFO] Dataset root found at: {faces_root}")
    return faces_root


def _find_faces_root(search_root: str) -> str | None:
    """Walk the tree and return the first directory named 'faces'."""
    for dirpath, dirnames, _ in os.walk(search_root):
        for d in dirnames:
            if d.lower() == "faces":
                return os.path.join(dirpath, d)
    return None


# ─────────────────────────────────────────────
# 2. Image Loading & Pre-processing
# ─────────────────────────────────────────────
def load_images_from_folder(faces_root: str):
    """
    Walk every sub-folder inside *faces_root*.
    Each sub-folder name becomes a class label.

    Pipeline per image:
        BGR → Grayscale → Resize → Normalize → Flatten

    Returns
    -------
    X : np.ndarray, shape (n_samples, IMG_HEIGHT*IMG_WIDTH)
    y : np.ndarray, shape (n_samples,)   integer class ids
    label_names : list[str]              class-id → person name
    """
    images, labels = [], []
    person_dirs = sorted([
        d for d in os.listdir(faces_root)
        if os.path.isdir(os.path.join(faces_root, d))
    ])

    if not person_dirs:
        raise ValueError(f"No sub-folders found inside {faces_root}")

    print(f"\n[INFO] Found {len(person_dirs)} person(s): {person_dirs}")

    for person_id, person_name in enumerate(person_dirs):
        person_path = os.path.join(faces_root, person_name)
        img_files   = [
            f for f in os.listdir(person_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        loaded = 0
        for fname in img_files:
            img_path = os.path.join(person_path, fname)
            img = _preprocess_image(img_path)
            if img is not None:
                images.append(img)
                labels.append(person_id)
                loaded += 1

        print(f"  [{person_id:02d}] {person_name:30s} → {loaded} images loaded")

    if not images:
        raise ValueError("No images could be loaded. Check the dataset path.")

    X = np.array(images, dtype=np.float32)   # already normalized & flat
    y = np.array(labels,  dtype=np.int32)

    print(f"\n[INFO] Dataset shape : X={X.shape}  y={y.shape}")
    return X, y, person_dirs


def _preprocess_image(img_path: str):
    """
    Load one image, DETECT & CROP face, and run the pre-processing pipeline.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ── Face Detection ───────────────────────────────────────────────
        cascade = get_face_cascade()
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        
        if len(faces) > 0:
            # Take the largest face found
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            gray = gray[y:y+h, x:x+w]
        
        # ── Lighting Normalization (Histogram Equalization) ──────────────
        gray = cv2.equalizeHist(gray)
        
        resized = cv2.resize(gray, IMG_SIZE)
        norm    = resized / 255.0
        flat    = norm.flatten()
        return flat.astype(np.float32)

    except Exception as e:
        print(f"  [ERROR] {img_path}: {e}")
        return None


def preprocess_single_image(img_path: str):
    """Pre-process a single image for prediction (returns 2-D array)."""
    flat = _preprocess_image(img_path)
    if flat is None:
        raise ValueError(f"Could not load image: {img_path}")
    return flat.reshape(1, -1)   # shape (1, n_features)


# ─────────────────────────────────────────────
# 3. Augmentation  (to reach 85%+ accuracy)
# ─────────────────────────────────────────────
def augment_dataset(faces_root: str, target_images_per_person: int = 200):
    """
    Creates realistic augmented variants (small rotations, zoom, brightness, flips).
    Ensures 0-1 normalization is maintained.
    """
    person_dirs = sorted([d for d in os.listdir(faces_root) if os.path.isdir(os.path.join(faces_root, d))])
    print(f"\n[INFO] Applying Realistic Augmentation (Target: {target_images_per_person}/person) …")
    cascade = get_face_cascade()

    for person in person_dirs:
        person_path = os.path.join(faces_root, person)
        real_images = [f for f in os.listdir(person_path) if f.lower().endswith((".jpg", ".png")) and not f.startswith("aug_")]
        if not real_images: continue
            
        current_count = len([f for f in os.listdir(person_path) if f.lower().endswith((".jpg", ".png"))])
        to_create = target_images_per_person - current_count
        if to_create <= 0: continue

        count = 0
        while count < to_create:
            fname = np.random.choice(real_images)
            img = cv2.imread(os.path.join(person_path, fname))
            if img is None: continue
            
            # Detect & Crop Face
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces) == 0: continue
            
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            face_img = img[y:y+h, x:x+w]
            
            # ── Realistic Augmentation ───────────────────────────────────
            # 1. Horizontal Flip
            if np.random.rand() > 0.5:
                face_img = cv2.flip(face_img, 1)
            
            # 2. Small Rotation (-10 to 10 degrees)
            angle = np.random.uniform(-10, 10)
            fh, fw = face_img.shape[:2]
            M = cv2.getRotationMatrix2D((fw // 2, fh // 2), angle, 1.0)
            face_img = cv2.warpAffine(face_img, M, (fw, fh), borderMode=cv2.BORDER_REPLICATE)
            
            # 3. Small Brightness adjustment
            bright = np.random.randint(-20, 20)
            face_img = np.clip(face_img.astype(np.int16) + bright, 0, 255).astype(np.uint8)

            # 4. Small Zoom (0.95 to 1.05)
            zoom = np.random.uniform(0.95, 1.05)
            face_img = cv2.resize(face_img, (int(fw*zoom), int(fh*zoom)))
            face_img = cv2.resize(face_img, (fw, fh)) # Rescale back to original face size

            out_path = os.path.join(person_path, f"aug_realistic_{count:04d}.jpg")
            cv2.imwrite(out_path, face_img)
            count += 1


# ─────────────────────────────────────────────
# 4. Plotting helpers
# ─────────────────────────────────────────────
def plot_training_history(history, save_path: str = "outputs/training_history.png"):
    """Plot Keras training accuracy & loss curves and save to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Accuracy ──────────────────────────────
    axes[0].plot(history.history["accuracy"],     label="Train Acc",  linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Acc",    linewidth=2, linestyle="--")
    axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch");  axes[0].set_ylabel("Accuracy")
    axes[0].legend();             axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    # ── Loss ──────────────────────────────────
    axes[1].plot(history.history["loss"],     label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss",   linewidth=2, linestyle="--")
    axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch");  axes[1].set_ylabel("Loss")
    axes[1].legend();             axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Training history saved → {save_path}")


def display_prediction(img_path: str, label: str, confidence: float,
                        save_path: str = "outputs/prediction_result.png"):
    """Display an image with its predicted label and confidence."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img  = cv2.imread(img_path)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, cmap="gray")
    color = "green" if confidence > 0.6 else "red"
    ax.set_title(
        f"Predicted: {label}\nConfidence: {confidence*100:.1f}%",
        fontsize=13, fontweight="bold", color=color
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Prediction image saved → {save_path}")
