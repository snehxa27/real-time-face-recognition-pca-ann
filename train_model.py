"""
train_model.py — ANN Model Training
=====================================
Workflow:
  1. Load & pre-process the face dataset
  2. Encode labels
  3. Split into train / validation / test sets
  4. Apply PCA (dimensionality reduction)
  5. Build a deep ANN with Dropout for regularisation
  6. Train with Early Stopping & Learning-Rate Scheduling
  7. Evaluate: confusion matrix, classification report, accuracy
  8. Save model, PCA object, and label encoder

Run:
    python train_model.py --dataset dataset/faces
    python train_model.py --zip    path/to/dataset.zip
    python train_model.py          # auto-detects from default locations

Author  : PCA-ANN Face Recognition Project
"""

import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder, label_binarize
from sklearn.metrics         import (confusion_matrix, classification_report,
                                     ConfusionMatrixDisplay)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers

# ── project modules ──────────────────────────────────────────────────────────
from utils      import (load_images_from_folder, extract_zip,
                         augment_dataset, plot_training_history)
from pca_module import (choose_n_components, apply_pca,
                         plot_eigenfaces, plot_explained_variance)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR  = "models"
OUTPUTS_DIR = "outputs"
RANDOM_SEED = 42

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Build the ANN
# ─────────────────────────────────────────────────────────────────────────────
def build_ann(input_dim: int, n_classes: int) -> keras.Model:
    """
    Deep Artificial Neural Network architecture:

        Input → Dense(512, ReLU) → BN → Dropout(0.4)
              → Dense(256, ReLU) → BN → Dropout(0.35)
              → Dense(128, ReLU) → BN → Dropout(0.3)
              → Dense(64,  ReLU) → BN → Dropout(0.25)
              → Dense(n_classes, Softmax)

    Batch Normalisation stabilises training; Dropout prevents overfitting.
    L2 regularisation on the first layer adds extra robustness.
    """
    model = keras.Sequential([
        # ── Layer 1 ──────────────────────────────────────────────────────
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-3)),
        layers.BatchNormalization(),
        layers.Dropout(0.50),

        # ── Layer 2 ──────────────────────────────────────────────────────
        layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-3)),
        layers.BatchNormalization(),
        layers.Dropout(0.50),

        # ── Layer 3 ──────────────────────────────────────────────────────
        layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-3)),
        layers.BatchNormalization(),
        layers.Dropout(0.50),

        # ── Output ───────────────────────────────────────────────────────
        layers.Dense(n_classes, activation="softmax"),
    ], name="Simplified_PCA_ANN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2. Callbacks
# ─────────────────────────────────────────────────────────────────────────────
def make_callbacks() -> list:
    """
    Return a list of Keras callbacks:
        - EarlyStopping  : stops training when val_loss stops improving
        - ReduceLROnPlateau: halves the learning rate on a plateau
        - ModelCheckpoint: saves the best model weights
    """
    return [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=10,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=7, min_lr=1e-6, verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, "best_ann_weights.h5"),
            monitor="val_accuracy", save_best_only=True,
            verbose=0
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test_pca, y_test, label_names: list):
    """Print classification report and save confusion matrix."""
    y_pred_probs = model.predict(X_test_pca, verbose=0)
    y_pred       = np.argmax(y_pred_probs, axis=1)

    print("\n" + "="*60)
    print("  CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))

    # Confusion matrix
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(label_names)), max(5, len(label_names)-1)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix — PCA + ANN", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved → {OUTPUTS_DIR}/confusion_matrix.png")

    return y_pred, y_pred_probs


def save_artefacts(model, pca, le):
    """Persist model, PCA transformer, and label encoder."""
    model.save(os.path.join(MODELS_DIR, "ann_model.keras"))
    import pickle
    with open(os.path.join(MODELS_DIR, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    print(f"\n[INFO] Model saved   → {MODELS_DIR}/ann_model.keras")
    print(f"[INFO] PCA saved     → {MODELS_DIR}/pca.pkl")
    print(f"[INFO] Encoder saved → {MODELS_DIR}/label_encoder.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main training pipeline
# ─────────────────────────────────────────────────────────────────────────────
def train(faces_root: str):
    """
    Final optimized pipeline: Split first, then augment only the training set.
    """
    print("\n" + "="*60)
    print("  PCA + ANN — ANTI-OVERFITTING PIPELINE")
    print("="*60)

    # 1. Load ORIGINAL images only (no old augmented files)
    # We filter out any files starting with 'aug_' to ensure we only split REAL photos
    X, y, label_names = load_images_from_folder(faces_root)
    n_classes = len(label_names)

    # 2. SPLIT FIRST (80% Train, 10% Val, 10% Test)
    # This ensures the Test set has NO augmented versions of training images
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.10, random_state=RANDOM_SEED, stratify=y
    )
    X_train_raw, X_val, y_train_raw, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.111, random_state=RANDOM_SEED, stratify=y_train_val
    )

    print(f"\n[DATA] Original Train : {X_train_raw.shape[0]}")
    print(f"[DATA] Original Val   : {X_val.shape[0]}")
    print(f"[DATA] Original Test  : {X_test.shape[0]}")

    # 3. AUGMENT ONLY TRAINING DATA (In-memory)
    print("[INFO] Augmenting Training Set to 1500 samples …")
    from utils import IMG_HEIGHT, IMG_WIDTH
    X_train = list(X_train_raw)
    y_train = list(y_train_raw)
    
    target_train_size = 1500
    while len(X_train) < target_train_size:
        idx = np.random.randint(len(X_train_raw))
        img_flat = X_train_raw[idx]
        img = img_flat.reshape(IMG_HEIGHT, IMG_WIDTH)
        
        # Horizontal Flip
        if np.random.rand() > 0.5: img = np.fliplr(img)
        
        # Random Shift (1-5 pixels)
        shift = np.random.randint(-5, 5)
        img = np.roll(img, shift, axis=0)
        
        X_train.append(img.flatten())
        y_train.append(y_train_raw[idx])
        
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"[DATA] Augmented Train: {X_train.shape[0]}")

    # 4. PCA (Targeting 80-100 components)
    n_components = 80
    pca, X_train_pca, X_test_pca = apply_pca(X_train, X_test, n_components)
    X_val_pca = pca.transform(X_val)

    # 5. Build and Train
    model = build_ann(input_dim=n_components, n_classes=n_classes)
    
    history = model.fit(
        X_train_pca, y_train,
        validation_data=(X_val_pca, y_val),
        epochs=150,
        batch_size=32,
        callbacks=make_callbacks(),
        verbose=1
    )

    # 6. Final Evaluation
    loss, acc = model.evaluate(X_test_pca, y_test, verbose=0)
    print(f"\n[FINAL] Test Accuracy: {acc*100:.2f}%")
    
    evaluate_model(model, X_test_pca, y_test, label_names)
    
    # Save artefacts
    le = LabelEncoder()
    le.classes_ = np.array(label_names)
    save_artefacts(model, pca, le)
    
    # Plot history
    plot_training_history(history, save_path=os.path.join(OUTPUTS_DIR, "training_history.png"))
    
    return acc


def _resolve_faces_root(args) -> str:
    """Resolve the faces-root directory from CLI arguments or auto-detect."""
    if args.dataset and os.path.isdir(args.dataset):
        return args.dataset

    if args.zip and os.path.isfile(args.zip):
        return extract_zip(args.zip, extract_to="dataset_extracted")

    candidates = ["dataset/faces", "dataset_raw/introduction_to_machine_learning-main/face_dataset/dataset/faces"]
    for c in candidates:
        if os.path.isdir(c):
            return c
    sys.exit("[ERROR] Dataset not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PCA+ANN Face Recogniser")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--zip", type=str, default="")
    args = parser.parse_args()

    faces_root = _resolve_faces_root(args)
    train(faces_root)
