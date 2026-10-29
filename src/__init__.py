"""
Unsupervised Learning & Matrix Factorization From Scratch (`src/`).

Pure NumPy implementations of:
- `KMeansScratch`: Lloyd's Coordinate Descent with D^2-weighted k-Means++ initialization.
- `GaussianMixtureEMScratch`: Full-Covariance Expectation-Maximization with Log-Sum-Exp numerical stability.
- `PCAScratch`: Principal Component Analysis via Covariance Eigendecomposition (`eigh`) and Singular Value Decomposition (`svd`).
- `silhouette_score_scratch`: Cluster cohesion and separation diagnostic.
"""

from .unsupervised_engine import (
    GaussianMixtureEMScratch,
    KMeansScratch,
    PCAScratch,
    silhouette_score_scratch,
)
from .plot_theme import (
    AMBER,
    BG_AXES,
    BG_CANVAS,
    CLASS_COLORS,
    CMAP_DECISION,
    CMAP_MULTICLASS,
    CMAP_SURFACE,
    CORAL,
    CYAN,
    EMERALD,
    MAGENTA,
    PALETTE,
    PURPLE,
    SLATE_BLUE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    apply_github_dark_style,
    plot_decision_boundary,
    plot_loss_landscape,
    save_fig,
    style_3d_axis,
)

__all__ = [
    "KMeansScratch",
    "GaussianMixtureEMScratch",
    "PCAScratch",
    "silhouette_score_scratch",
    "apply_github_dark_style",
    "style_3d_axis",
    "plot_decision_boundary",
    "plot_loss_landscape",
    "save_fig",
    "PALETTE",
    "CLASS_COLORS",
    "BG_CANVAS",
    "BG_AXES",
    "TEXT_PRIMARY",
    "TEXT_MUTED",
    "CYAN",
    "CORAL",
    "AMBER",
    "EMERALD",
    "PURPLE",
    "SLATE_BLUE",
    "MAGENTA",
    "CMAP_SURFACE",
    "CMAP_DECISION",
    "CMAP_MULTICLASS",
]

