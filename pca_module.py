"""
pca_module.py — PCA (Principal Component Analysis) Module
==========================================================
Handles:
  - Choosing the optimal number of PCA components dynamically
  - Fitting PCA and transforming the dataset
  - Visualising Eigenfaces
  - Plotting explained-variance ratio

Author  : PCA-ANN Face Recognition Project
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import IMG_HEIGHT, IMG_WIDTH


# ─────────────────────────────────────────────
# 1. Dynamic Component Selection
# ─────────────────────────────────────────────
def choose_n_components(X_train: np.ndarray,
                         variance_threshold: float = 0.95,
                         max_components: int = 150) -> int:
    """
    Find the *minimum* number of PCA components that explain at
    least *variance_threshold* (default 95 %) of the total variance.

    Parameters
    ----------
    X_train           : training data, shape (n_samples, n_features)
    variance_threshold: cumulative explained-variance target (0–1)
    max_components    : hard upper limit

    Returns
    -------
    n_components : int
    """
    # Cap at (n_samples - 1) to keep PCA well-defined
    n_max = min(max_components, X_train.shape[0] - 1, X_train.shape[1])

    # Quick PCA with the maximum feasible components
    pca_probe = PCA(n_components=n_max, whiten=True, random_state=42)
    pca_probe.fit(X_train)

    cumvar = np.cumsum(pca_probe.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_threshold)) + 1
    n_components = min(n_components, n_max)

    print(f"\n[PCA] Dynamic component selection:")
    print(f"      Variance threshold : {variance_threshold*100:.0f}%")
    print(f"      Components chosen  : {n_components}")
    print(f"      Variance explained : {cumvar[n_components-1]*100:.2f}%")

    return n_components


# ─────────────────────────────────────────────
# 2. Fit & Transform
# ─────────────────────────────────────────────
def apply_pca(X_train: np.ndarray, X_test: np.ndarray,
              n_components: int):
    """
    Fit PCA on *X_train* and transform both train and test sets.

    Whitening is enabled (zero-mean, unit-variance components) which
    helps the ANN converge faster.

    Returns
    -------
    pca        : fitted sklearn PCA object
    X_train_pca: transformed training data
    X_test_pca : transformed test data
    """
    print(f"\n[PCA] Fitting PCA with {n_components} components …")

    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train)   # fit on train only!
    X_test_pca  = pca.transform(X_test)        # apply same transform

    print(f"      Input  shape : {X_train.shape}")
    print(f"      Output shape : {X_train_pca.shape}")
    print(f"      Reduction    : {X_train.shape[1]} → {X_train_pca.shape[1]} features")

    return pca, X_train_pca, X_test_pca


# ─────────────────────────────────────────────
# 3. Eigenfaces Visualisation
# ─────────────────────────────────────────────
def plot_eigenfaces(pca: PCA,
                    n_eigenfaces: int = 16,
                    save_path: str = "outputs/eigenfaces.png"):
    """
    Plot the top N Eigenfaces (the principal components reshaped
    back to image dimensions).

    Eigenfaces represent the "directions of maximum variance" in
    face space — they look like ghostly face-like patterns.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n_show = min(n_eigenfaces, pca.n_components_)
    n_cols = 4
    n_rows = (n_show + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 2.5, n_rows * 2.5))
    fig.suptitle("Eigenfaces  (Top Principal Components)",
                 fontsize=15, fontweight="bold", y=1.01)

    axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        if i < n_show:
            eigenface = pca.components_[i].reshape(IMG_HEIGHT, IMG_WIDTH)
            ax.imshow(eigenface, cmap="bone")
            ax.set_title(f"PC {i+1}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PCA] Eigenfaces saved → {save_path}")


# ─────────────────────────────────────────────
# 4. Explained Variance Plot
# ─────────────────────────────────────────────
def plot_explained_variance(pca: PCA,
                             save_path: str = "outputs/explained_variance.png"):
    """
    Bar chart of per-component variance + cumulative curve.
    Helps justify the chosen number of components visually.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cumvar    = np.cumsum(pca.explained_variance_ratio_)
    n_comp    = pca.n_components_
    comp_idx  = np.arange(1, n_comp + 1)

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Bar: individual explained variance
    ax1.bar(comp_idx, pca.explained_variance_ratio_ * 100,
            alpha=0.6, color="steelblue", label="Individual variance %")
    ax1.set_xlabel("Principal Component", fontsize=12)
    ax1.set_ylabel("Variance Explained (%)", fontsize=12, color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    # Line: cumulative variance
    ax2 = ax1.twinx()
    ax2.plot(comp_idx, cumvar * 100, color="crimson",
             linewidth=2, marker=".", label="Cumulative %")
    ax2.axhline(95, color="orange", linestyle="--", linewidth=1.2,
                label="95% threshold")
    ax2.set_ylabel("Cumulative Variance (%)", fontsize=12, color="crimson")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax2.set_ylim([0, 105])

    # Combined legend
    lines1, lbl1 = ax1.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbl1 + lbl2, loc="center right", fontsize=10)

    plt.title("PCA — Explained Variance Ratio", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PCA] Explained variance plot saved → {save_path}")
