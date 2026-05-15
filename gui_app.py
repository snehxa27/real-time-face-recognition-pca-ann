"""
gui_app.py — Tkinter GUI Application
=======================================
A simple, beginner-friendly GUI for the PCA + ANN Face Recognition system.

Features:
  • Upload any image from disk
  • Predict the person name + confidence
  • Visual confidence bar for each known person
  • Unknown face detection

Run:
    python gui_app.py

Author  : PCA-ANN Face Recognition Project
"""

import os
import sys
import pickle
import threading
import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk        # pip install Pillow
except ImportError as e:
    sys.exit(f"[ERROR] Missing dependency: {e}\n  pip install Pillow")

import tensorflow as tf
from tensorflow import keras

from utils      import preprocess_single_image
from predict    import predict_face

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
MODELS_DIR   = "models"
MODEL_PATH   = os.path.join(MODELS_DIR, "ann_model.keras")
PCA_PATH     = os.path.join(MODELS_DIR, "pca.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
THRESHOLD    = 0.50


# ═══════════════════════════════════════════════════════════════════════════
# Main Application Window
# ═══════════════════════════════════════════════════════════════════════════
class FaceRecognitionApp:
    """PCA + ANN Face Recognition — Tkinter GUI"""

    # ── Colour palette ────────────────────────────────────────────────────
    BG        = "#1e1e2e"
    PANEL     = "#2a2a3e"
    ACCENT    = "#7c3aed"   # purple
    SUCCESS   = "#22c55e"   # green
    DANGER    = "#ef4444"   # red
    FG        = "#f8fafc"
    FG_DIM    = "#94a3b8"
    FONT_MAIN = ("Segoe UI", 11)
    FONT_HEAD = ("Segoe UI", 14, "bold")
    FONT_BIG  = ("Segoe UI", 24, "bold")

    def __init__(self, root: tk.Tk):
        self.root   = root
        self.model  = None
        self.pca    = None
        self.le     = None
        self._img_tk = None          # keep reference to avoid GC
        self._build_ui()
        self._load_artefacts_async()

    # ── UI construction ───────────────────────────────────────────────────
    def _build_ui(self):
        self.root.title("PCA + ANN Face Recognition")
        self.root.geometry("820x680")
        self.root.resizable(True, True)
        self.root.configure(bg=self.BG)
        self.root.minsize(700, 560)

        # ── Title bar ─────────────────────────────────────────────────────
        title_frame = tk.Frame(self.root, bg=self.ACCENT, pady=12)
        title_frame.pack(fill="x")
        tk.Label(title_frame, text="🧠  PCA + ANN Face Recognition",
                 font=self.FONT_HEAD, bg=self.ACCENT, fg=self.FG
                 ).pack()
        tk.Label(title_frame, text="Principal Component Analysis + Artificial Neural Network",
                 font=("Segoe UI", 9), bg=self.ACCENT, fg="#ddd6fe"
                 ).pack()

        # ── Status bar ────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Loading model …")
        status_bar = tk.Label(self.root, textvariable=self.status_var,
                              font=("Segoe UI", 9), bg="#111827",
                              fg=self.FG_DIM, anchor="w", pady=4, padx=10)
        status_bar.pack(fill="x", side="bottom")

        # ── Main content ──────────────────────────────────────────────────
        content = tk.Frame(self.root, bg=self.BG)
        content.pack(fill="both", expand=True, padx=16, pady=12)

        # Left: image preview
        left  = tk.Frame(content, bg=self.PANEL, bd=0, relief="flat")
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        tk.Label(left, text="Face Preview", font=self.FONT_MAIN,
                 bg=self.PANEL, fg=self.FG_DIM).pack(pady=(10, 4))

        self.img_canvas = tk.Label(
            left, bg="#111827", width=35, height=16,
            text="No image loaded\n\nClick 'Upload Image'",
            font=("Segoe UI", 10), fg=self.FG_DIM,
            relief="flat", bd=0
        )
        self.img_canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        btn_frame = tk.Frame(left, bg=self.PANEL)
        btn_frame.pack(fill="x", padx=10, pady=(0, 12))

        self._btn(btn_frame, "📂  Upload Image", self._upload_image,
                  bg=self.ACCENT).pack(fill="x", pady=(0, 6))
        self._btn(btn_frame, "🔍  Predict", self._run_prediction,
                  bg="#0f766e").pack(fill="x")

        # Right: results
        right = tk.Frame(content, bg=self.PANEL, bd=0, width=280)
        right.pack(side="right", fill="y", padx=(8, 0))
        right.pack_propagate(False)

        tk.Label(right, text="Prediction Results", font=self.FONT_MAIN,
                 bg=self.PANEL, fg=self.FG_DIM).pack(pady=(10, 4))

        # Big prediction label
        self.pred_label_var = tk.StringVar(value="—")
        pred_lbl = tk.Label(right, textvariable=self.pred_label_var,
                            font=self.FONT_BIG, bg=self.PANEL, fg=self.FG,
                            wraplength=250)
        pred_lbl.pack(pady=(8, 2))

        self.conf_var = tk.StringVar(value="Confidence: —")
        tk.Label(right, textvariable=self.conf_var,
                 font=("Segoe UI", 12), bg=self.PANEL, fg=self.FG_DIM
                 ).pack(pady=(0, 10))

        # Confidence bar
        tk.Label(right, text="Confidence", font=("Segoe UI", 9),
                 bg=self.PANEL, fg=self.FG_DIM).pack()
        self.conf_bar = ttk.Progressbar(right, length=220,
                                         mode="determinate", maximum=100)
        self.conf_bar.pack(pady=(2, 12), padx=16)

        # Known / Unknown badge
        self.badge_var = tk.StringVar(value="")
        self.badge_lbl = tk.Label(right, textvariable=self.badge_var,
                                   font=("Segoe UI", 10, "bold"),
                                   bg=self.PANEL, fg=self.FG, pady=4)
        self.badge_lbl.pack()

        # Divider
        tk.Frame(right, bg="#374151", height=1).pack(fill="x", padx=10, pady=10)

        # Top-N breakdown
        tk.Label(right, text="All Scores", font=("Segoe UI", 9, "bold"),
                 bg=self.PANEL, fg=self.FG_DIM).pack()

        self.scores_frame = tk.Frame(right, bg=self.PANEL)
        self.scores_frame.pack(fill="both", expand=True, padx=10, pady=4)

        self._img_path = None

    def _btn(self, parent, text, cmd, bg="#374151", fg="white"):
        return tk.Button(parent, text=text, command=cmd,
                         font=("Segoe UI", 10, "bold"),
                         bg=bg, fg=fg, activebackground=bg,
                         activeforeground=fg, relief="flat",
                         cursor="hand2", pady=8)

    # ── Artefact loading (async) ──────────────────────────────────────────
    def _load_artefacts_async(self):
        def _load():
            try:
                self.model = keras.models.load_model(MODEL_PATH)
                with open(PCA_PATH,     "rb") as f: self.pca = pickle.load(f)
                with open(ENCODER_PATH, "rb") as f: self.le  = pickle.load(f)
                self.root.after(0, lambda: self.status_var.set(
                    f"✓ Model ready  —  Classes: {list(self.le.classes_)}"
                ))
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set(
                    f"⚠ Model not loaded: {e}  (run train_model.py first)"
                ))

        threading.Thread(target=_load, daemon=True).start()

    # ── Image upload ─────────────────────────────────────────────────────
    def _upload_image(self):
        path = filedialog.askopenfilename(
            title="Select Face Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
                       ("All files",   "*.*")]
        )
        if not path:
            return

        self._img_path = path

        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((300, 300), Image.LANCZOS)
            self._img_tk = ImageTk.PhotoImage(img)
            self.img_canvas.configure(image=self._img_tk, text="")
        except Exception as e:
            messagebox.showerror("Image Error", str(e))
            return

        self.status_var.set(f"Image loaded: {os.path.basename(path)}")
        self._clear_results()

    # ── Prediction ────────────────────────────────────────────────────────
    def _run_prediction(self):
        if not self._img_path:
            messagebox.showwarning("No Image", "Please upload an image first.")
            return
        if self.model is None:
            messagebox.showerror("Model Not Ready",
                                 "Model not loaded yet. Please wait or run train_model.py.")
            return

        self.status_var.set("Predicting …")
        self.root.update_idletasks()

        try:
            result = predict_face(self._img_path, self.model,
                                   self.pca, self.le, THRESHOLD)
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self.status_var.set("Prediction failed.")
            return

        # ── Update UI ────────────────────────────────────────────────────
        label = result["label"]
        conf  = result["confidence"]

        self.pred_label_var.set(label)
        self.conf_var.set(f"Confidence: {conf*100:.1f}%")
        self.conf_bar["value"] = conf * 100

        if result["is_known"]:
            self.badge_var.set("✅  IDENTIFIED")
            self.badge_lbl.configure(fg=self.SUCCESS)
        else:
            self.badge_var.set("❓  UNKNOWN FACE")
            self.badge_lbl.configure(fg=self.DANGER)

        # ── Score breakdown ───────────────────────────────────────────────
        for widget in self.scores_frame.winfo_children():
            widget.destroy()

        sorted_probs = sorted(result["all_probs"].items(),
                               key=lambda x: x[1], reverse=True)
        for name, prob in sorted_probs[:6]:
            row = tk.Frame(self.scores_frame, bg=self.PANEL)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{name[:18]}", width=18, anchor="w",
                     font=("Segoe UI", 8), bg=self.PANEL, fg=self.FG
                     ).pack(side="left")
            bar_w = max(2, int(prob * 100))
            tk.Frame(row, bg=self.ACCENT, width=bar_w, height=8
                     ).pack(side="left", padx=(2, 4))
            tk.Label(row, text=f"{prob*100:.1f}%",
                     font=("Segoe UI", 8), bg=self.PANEL, fg=self.FG_DIM
                     ).pack(side="left")

        self.status_var.set(f"Prediction complete: {label} ({conf*100:.1f}%)")

    def _clear_results(self):
        self.pred_label_var.set("—")
        self.conf_var.set("Confidence: —")
        self.conf_bar["value"] = 0
        self.badge_var.set("")
        for w in self.scores_frame.winfo_children():
            w.destroy()


# ─────────────────────────────────────────────
# Launch helper
# ─────────────────────────────────────────────
def launch_gui():
    root = tk.Tk()
    app  = FaceRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
