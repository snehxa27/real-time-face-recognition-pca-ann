# 🧠 PCA + ANN Face Recognition System

> **Implementation of Principal Component Analysis (PCA) with Artificial Neural Network (ANN) for Face Recognition**
>
> A complete, professional Python project for internship submission, college mini-project, or viva presentation.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Algorithm Explanation](#algorithm-explanation)
3. [Project Structure](#project-structure)
4. [Dataset](#dataset)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Output Screenshots](#output-screenshots)
8. [Accuracy Optimization Tips](#accuracy-optimization-tips)
9. [Viva Q&A Preparation](#viva-qa-preparation)

---

## Project Overview

This system combines two powerful techniques:

| Component | Role |
|-----------|------|
| **PCA** | Reduces each face from 10,000 pixels → ~80 "eigenface" features |
| **ANN** | Classifies the compressed features to identify the person |

**Why PCA first?**  
Raw face images are high-dimensional (100×100 = 10,000 features). Training a neural network on 10,000 inputs per sample is slow and prone to overfitting. PCA compresses the data to ≈80 features while retaining 95 % of the information — making the ANN faster, smaller, and more accurate.

---

## Algorithm Explanation

### Step 1 — Pre-processing
```
BGR image → Grayscale → Resize(100×100) → Normalize(/255) → Flatten → 10,000-D vector
```

### Step 2 — PCA (Dimensionality Reduction)
```
10,000-D → PCA fits on training data → ~80 "Eigenface" components (95% variance)
```
Each Eigenface is a principal direction of variation in face space.  
Projecting onto them compresses data while discarding noise.

### Step 3 — ANN (Classification)
```
80-D PCA features → 512 → 256 → 128 → 64 → Softmax(n_classes)
```
Dropout layers (25–40 %) prevent memorising the training set.  
Batch Normalisation speeds up convergence.

---

## Project Structure

```
face_recognition_project/
│
├── main.py                  # 🚀 Entry point — runs the full pipeline
├── train_model.py           # 🏋️  Training: PCA + ANN
├── predict.py               # 🔍 Single image / folder prediction
├── pca_module.py            # 📐 PCA fitting, eigenfaces, variance plots
├── utils.py                 # 🛠️  Image loading, augmentation, plotting
├── gui_app.py               # 🖥️  Tkinter GUI application
├── webcam_recognition.py    # 📷 Real-time webcam recognition
│
├── requirements.txt         # 📦 Python dependencies
├── README.md                # 📖 This file
│
├── models/                  # 💾 Saved artefacts (auto-created)
│   ├── ann_model.keras      #     Trained ANN
│   ├── pca.pkl              #     Fitted PCA transformer
│   └── label_encoder.pkl    #     Class-name ↔ integer mapping
│
├── outputs/                 # 📊 Plots & results (auto-created)
│   ├── eigenfaces.png
│   ├── explained_variance.png
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── prediction_result.png
│
└── dataset_raw/             # 📂 Your extracted ZIP dataset
    └── .../faces/
        ├── PersonA/
        │   ├── img_001.jpg
        │   └── ...
        └── PersonB/
            └── ...
```

---

## Dataset

The project uses the face dataset provided in `dataset.zip` (inside the course ZIP).

```
dataset/faces/
└── Aamir/          ← 38 face images  (286×286 px, colour)
```

Since the provided ZIP contains a single person, the training script **automatically creates 4 augmented synthetic classes** (flip, brightness, rotation variants) to demonstrate multi-class classification. This is clearly documented in the code.

> **For a real project**, add more sub-folders:
> ```
> dataset/faces/Alice/   ← ≥20 photos
> dataset/faces/Bob/     ← ≥20 photos
> dataset/faces/Carol/   ← ≥20 photos
> ```

---

## Installation

### Prerequisites
- Python 3.9 – 3.12
- pip

### Steps

```bash
# 1. Clone or download the project folder
cd face_recognition_project

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **GPU acceleration (optional):**  
> Replace `tensorflow` with `tensorflow-gpu` and install CUDA 11.8 + cuDNN 8.6.

---

## Usage

### Full pipeline (train + evaluate)

```bash
# Auto-detect dataset
python main.py

# Provide the ZIP file explicitly
python main.py --zip /path/to/introduction_to_machine_learning-main.zip

# Provide the faces folder directly
python main.py --dataset dataset/faces
```

### Train only

```bash
python train_model.py
python train_model.py --dataset dataset/faces
python train_model.py --zip dataset.zip
```

### Predict a single image

```bash
python predict.py --image path/to/face.jpg
python predict.py --image face.jpg --threshold 0.60
```

### Predict a folder of images

```bash
python predict.py --folder path/to/test_images/
```

### Launch the GUI

```bash
python gui_app.py
```

### Real-time webcam recognition

```bash
python webcam_recognition.py
python webcam_recognition.py --camera 0 --threshold 0.55
```
Press `Q` or `ESC` to quit; `S` to save a screenshot.

---

## Output Screenshots

After training you will find in `outputs/`:

| File | Description |
|------|-------------|
| `eigenfaces.png` | Top-16 Eigenfaces — the "directions of maximum face variation" |
| `explained_variance.png` | Bar chart: individual + cumulative PCA variance |
| `training_history.png` | Accuracy & loss curves over epochs |
| `confusion_matrix.png` | Per-class prediction accuracy grid |
| `prediction_result.png` | Annotated prediction for a single query image |

---

## Accuracy Optimization Tips

| Technique | Effect |
|-----------|--------|
| More training images per person (≥50) | Biggest single improvement |
| Consistent lighting & face cropping | Reduces noise |
| PCA variance threshold = 0.95 | Balances info retention vs. over-fitting |
| Dropout 0.25–0.40 | Prevents memorisation |
| Batch Normalisation | Faster, more stable training |
| ReduceLROnPlateau | Escapes flat loss plateaus |
| EarlyStopping (patience=15) | Prevents over-training |
| Image augmentation | Artificially expands small datasets |
| Whiten PCA (`whiten=True`) | Scales components → better ANN convergence |

---

## Viva Q&A Preparation

**Q: What is PCA?**  
A: Principal Component Analysis is a linear dimensionality-reduction technique. It finds the directions (principal components) along which the data varies the most, then projects data onto a smaller set of these directions.

**Q: What are Eigenfaces?**  
A: Eigenfaces are the principal components of a face dataset reshaped back into image dimensions. They represent the "average directions of variation" across all faces and look like ghostly face patterns.

**Q: Why use PCA before ANN?**  
A: Raw face images have tens of thousands of pixels (features). PCA reduces these to ~80 Eigenface coefficients retaining 95 % of information. This makes the ANN smaller, faster to train, and less prone to overfitting.

**Q: What is Dropout?**  
A: During each training step, Dropout randomly sets a fraction of neurons to zero. This prevents neurons from co-adapting too closely to specific training examples, which reduces overfitting.

**Q: What is Early Stopping?**  
A: A callback that monitors validation loss and stops training if it stops improving for N consecutive epochs, then restores the best weights seen so far.

**Q: What is Batch Normalisation?**  
A: A layer that normalises its inputs to zero mean and unit variance, per mini-batch. This stabilises and accelerates training.

**Q: How is an unknown face detected?**  
A: The ANN outputs a probability distribution across all known classes. If the maximum probability is below a threshold (e.g. 0.50), the face is labelled "Unknown" instead of being assigned to any class.

---

## References

1. Turk, M. & Pentland, A. (1991). *Eigenfaces for Recognition*. Journal of Cognitive Neuroscience.
2. Jolliffe, I.T. (2002). *Principal Component Analysis*. Springer.
3. Goodfellow, I. et al. (2016). *Deep Learning*. MIT Press.
4. OpenCV Documentation — https://docs.opencv.org
5. Scikit-learn Documentation — https://scikit-learn.org
6. TensorFlow / Keras — https://www.tensorflow.org

---

*Developed for academic purposes — PCA + ANN Face Recognition Mini-Project*


## Dataset

The dataset used in this project was organized and adapted from the following machine learning repository:

Source Repository: https://github.com/robaita/introduction_to_machine_learning

The face dataset was manually structured into class-wise folders for PCA + ANN face recognition training.

Example structure:

dataset/faces/
├── Aamir/
├── Ajay/
├── Akshay/
├── Alia/
├── Amitabh/
├── Deepika/
├── Disha/
├── Farhan/
└── Ileana/

The project also uses OpenCV Haar Cascade for face detection and applies data augmentation before training.

