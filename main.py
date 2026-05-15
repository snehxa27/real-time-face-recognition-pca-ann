"""
main.py — Project Entry Point
==============================
Single command to run the full pipeline:
  1. Locate / extract the dataset
  2. Train the PCA + ANN model
  3. Run a quick self-test prediction
  4. Print a final summary

Usage:
    python main.py                          # auto-detect dataset
    python main.py --zip dataset.zip        # provide ZIP file
    python main.py --dataset dataset/faces  # provide faces folder

Author  : PCA-ANN Face Recognition Project
"""

import os
import sys
import argparse
import numpy as np

# ─────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────
BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║   PCA + ANN Face Recognition System                     ║
║   Using Principal Component Analysis &                  ║
║   Artificial Neural Network                             ║
╚══════════════════════════════════════════════════════════╝
"""


def print_banner():
    print(BANNER)


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="PCA + ANN Face Recognition — Full Pipeline"
    )
    parser.add_argument("--dataset",   type=str, default="",
                        help="Path to faces folder (sub-folder per person)")
    parser.add_argument("--zip",       type=str, default="",
                        help="Path to dataset ZIP file")
    parser.add_argument("--mode",      type=str,
                        choices=["train", "predict", "webcam", "gui", "all"],
                        default="all",
                        help="Which mode to run (default: all)")
    parser.add_argument("--image",     type=str, default="",
                        help="Image path for prediction mode")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Confidence threshold for unknown detection")
    args = parser.parse_args()

    # ── Resolve dataset path ──────────────────────────────────────────────
    faces_root = _resolve_faces_root(args)

    # ── Run selected mode(s) ─────────────────────────────────────────────
    if args.mode in ("train", "all"):
        print("\n[MODE] TRAINING  ─────────────────────────────────────────")
        from train_model import train
        acc = train(faces_root)
        print(f"\n  ✓  Model trained.  Test accuracy = {acc*100:.2f}%")

    if args.mode in ("predict", "all") and args.image:
        print("\n[MODE] PREDICTION ────────────────────────────────────────")
        from predict import predict_single
        predict_single(args.image, args.threshold)

    if args.mode == "webcam":
        print("\n[MODE] WEBCAM ────────────────────────────────────────────")
        from webcam_recognition import run_webcam
        run_webcam(threshold=args.threshold)

    if args.mode == "gui":
        print("\n[MODE] GUI ───────────────────────────────────────────────")
        from gui_app import launch_gui
        launch_gui()

    if args.mode == "all":
        print("\n" + "="*60)
        print("  ALL STEPS COMPLETE")
        print("  Outputs saved in:  ./outputs/")
        print("  Model saved in:    ./models/")
        print("\n  Next steps:")
        print("  • python main.py --mode predict --image <path>")
        print("  • python main.py --mode webcam")
        print("  • python main.py --mode gui")
        print("="*60)


# ─────────────────────────────────────────────
# Dataset path resolver
# ─────────────────────────────────────────────
def _resolve_faces_root(args) -> str:
    """Find the faces directory from args or auto-detect."""
    if args.dataset and os.path.isdir(args.dataset):
        return args.dataset

    if args.zip and os.path.isfile(args.zip):
        from utils import extract_zip
        return extract_zip(args.zip, extract_to="dataset_extracted")

    # ── auto-detect candidates ────────────────────────────────────────────
    candidates = [
        "dataset/faces",
        "dataset_raw/introduction_to_machine_learning-main/face_dataset/dataset/faces",
        "face_dataset/dataset/faces",
        "faces",
    ]
    for c in candidates:
        if os.path.isdir(c):
            print(f"[INFO] Auto-detected dataset: {c}")
            return c

    # ── look for any dataset.zip nearby ───────────────────────────────────
    for f in os.listdir("."):
        if f.lower().endswith(".zip") and "dataset" in f.lower():
            print(f"[INFO] Found ZIP: {f} — extracting …")
            from utils import extract_zip
            return extract_zip(f, extract_to="dataset_extracted")

    sys.exit(
        "\n[ERROR] Could not find the face dataset.\n"
        "  Run with:  python main.py --zip path/to/dataset.zip\n"
        "          or python main.py --dataset path/to/faces/"
    )


if __name__ == "__main__":
    main()
