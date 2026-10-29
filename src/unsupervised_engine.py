"""
From-Scratch Unsupervised Learning & Matrix Factorization Engine
(`06_Unsupervised_Learning_and_Factorization/unsupervised.py`).

Includes:
- `KMeansScratch`: Lloyd's Coordinate Descent with D^2-weighted k-Means++ initialization.
- `GaussianMixtureEMScratch`: Full-Covariance Expectation-Maximization with Log-Sum-Exp stability.
- `PCAScratch`: Principal Component Analysis via Covariance Eigendecomposition (`eigh`) and `svd`.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple
import numpy as np


# =============================================================================
# 1. k-MEANS & k-MEANS++ CLUSTERING
# =============================================================================

def _pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    sq_A = np.sum(A ** 2, axis=1, keepdims=True)
    sq_B = np.sum(B ** 2, axis=1, keepdims=True).T
    return np.maximum(sq_A + sq_B - 2.0 * (A @ B.T), 0.0)


def silhouette_score_scratch(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute mean Silhouette Coefficient across all m samples:
        s(i) = (b(i) - a(i)) / max(a(i), b(i))
    where a(i) is mean intra-cluster distance and b(i) is mean nearest-cluster distance.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    lbl = np.asarray(labels, dtype=np.int64).ravel()
    uniq = np.unique(lbl)
    if uniq.size <= 1:
        return 0.0

    D = np.sqrt(_pairwise_sq_dists(X_arr, X_arr))
    m = X_arr.shape[0]
    s_vals = np.zeros(m, dtype=np.float64)

    for i in range(m):
        own_cluster = lbl[i]
        own_mask = (lbl == own_cluster)
        own_mask[i] = False
        a_i = float(np.mean(D[i, own_mask])) if np.any(own_mask) else 0.0

        b_i = np.inf
        for other_c in uniq:
            if other_c == own_cluster:
                continue
            other_mask = lbl == other_c
            if np.any(other_mask):
                b_i = min(b_i, float(np.mean(D[i, other_mask])))

        denom = max(a_i, b_i)
        s_vals[i] = (b_i - a_i) / denom if denom > 1e-12 else 0.0

    return float(np.mean(s_vals))


class KMeansScratch:
    """
    First-Principles k-Means Clustering with D^2-weighted k-Means++ Seeding.

    Minimizes Distortion (WCSS):
        J(c, mu) = (1 / m) sum_{i=1}^m ||x^{(i)} - mu_{c^{(i)}}||_2^2
    """

    def __init__(
        self,
        n_clusters: int = 3,
        init: Literal["kmeans++", "random"] = "kmeans++",
        max_iter: int = 200,
        tol: float = 1e-6,
        random_state: int = 42,
        k: Optional[int] = None,
    ) -> None:
        if k is not None:
            n_clusters = k
        assert n_clusters >= 1
        self.n_clusters = n_clusters
        self.k = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = 0.0
        self.inertia_history: List[float] = []
        self.centroid_trajectory: List[np.ndarray] = []

    def _init_centroids(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        m, n = X.shape
        if self.init == "random":
            idx = rng.choice(m, size=self.n_clusters, replace=False)
            return X[idx].copy()

        # D^2-weighted k-Means++ probabilistic seeding
        centroids = np.empty((self.n_clusters, n), dtype=np.float64)
        first_idx = int(rng.integers(0, m))
        centroids[0] = X[first_idx]

        for k in range(1, self.n_clusters):
            dists_sq = np.min(_pairwise_sq_dists(X, centroids[:k]), axis=1)
            total_d2 = float(np.sum(dists_sq))
            if total_d2 <= 1e-15:
                next_idx = int(rng.integers(0, m))
            else:
                probs = dists_sq / total_d2
                next_idx = int(rng.choice(m, p=probs))
            centroids[k] = X[next_idx]

        return centroids

    def fit(self, X: np.ndarray) -> "KMeansScratch":
        X_arr = np.asarray(X, dtype=np.float64)
        assert X_arr.ndim == 2
        m, n = X_arr.shape
        rng = np.random.default_rng(self.random_state)

        centroids = self._init_centroids(X_arr, rng)
        self.inertia_history.clear()
        self.centroid_trajectory.clear()
        self.centroid_trajectory.append(centroids.copy())

        labels = np.zeros(m, dtype=np.int64)
        for _ in range(self.max_iter):
            # Step 1: Coordinate descent over cluster assignments c^{(i)}
            D_sq = _pairwise_sq_dists(X_arr, centroids)
            labels = np.argmin(D_sq, axis=1)
            inertia = float(np.mean(D_sq[np.arange(m), labels]))
            self.inertia_history.append(inertia)

            # Step 2: Coordinate descent over centroids mu_k
            new_centroids = centroids.copy()
            for k in range(self.n_clusters):
                mask = labels == k
                if np.any(mask):
                    new_centroids[k] = np.mean(X_arr[mask], axis=0)
                else:
                    # Re-seed empty cluster at farthest point
                    farthest_idx = int(np.argmax(np.min(D_sq, axis=1)))
                    new_centroids[k] = X_arr[farthest_idx]

            shift = float(np.linalg.norm(new_centroids - centroids))
            centroids = new_centroids
            self.centroid_trajectory.append(centroids.copy())
            if shift < self.tol:
                break

        D_sq_final = _pairwise_sq_dists(X_arr, centroids)
        self.labels_ = np.argmin(D_sq_final, axis=1)
        self.inertia_ = float(np.mean(D_sq_final[np.arange(m), self.labels_]))
        self.centroids_ = centroids
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.centroids_ is not None
        X_arr = np.asarray(X, dtype=np.float64)
        return np.argmin(_pairwise_sq_dists(X_arr, self.centroids_), axis=1)


# =============================================================================
# 2. GAUSSIAN MIXTURE MODEL (FULL-COVARIANCE EXPECTATION-MAXIMIZATION)
# =============================================================================

class GaussianMixtureEMScratch:
    """
    Full-Covariance Gaussian Mixture Model trained via Expectation-Maximization (EM)
    with Log-Sum-Exp numerical stability.
    """

    def __init__(
        self,
        n_components: int = 3,
        max_iter: int = 150,
        tol: float = 1e-6,
        reg_covar: float = 1e-6,
        init_method: Literal["kmeans++", "random"] = "kmeans++",
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.init_method = init_method
        self.random_state = random_state

        self.weights_: Optional[np.ndarray] = None
        self.means_: Optional[np.ndarray] = None
        self.covariances_: Optional[np.ndarray] = None
        self.responsibilities_: Optional[np.ndarray] = None
        self.log_likelihood_history: List[float] = []
        self.means_history: List[np.ndarray] = []
        self.covs_history: List[np.ndarray] = []

    @staticmethod
    def _log_gaussian_pdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Evaluate log N(x | mu, Sigma) for all rows of X using Cholesky / slogdet."""
        n = X.shape[1]
        diff = X - mean[None, :]  # (m, n)
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            cov = cov + 1e-4 * np.eye(n)
            _, logdet = np.linalg.slogdet(cov)
        sol = np.linalg.solve(cov, diff.T).T  # (m, n) = diff @ inv(cov)
        mahalanobis = np.sum(diff * sol, axis=1)
        return -0.5 * (n * np.log(2.0 * np.pi) + logdet + mahalanobis)

    def _compute_log_weighted_prob(
        self, X: np.ndarray, weights: np.ndarray, means: np.ndarray, covs: np.ndarray
    ) -> np.ndarray:
        """Return matrix of shape (m, K) with entry (i, k) = log pi_k + log N(x_i | mu_k, Sigma_k)."""
        m = X.shape[0]
        K = self.n_components
        log_prob = np.empty((m, K), dtype=np.float64)
        for k in range(K):
            log_prob[:, k] = np.log(np.maximum(weights[k], 1e-15)) + self._log_gaussian_pdf(X, means[k], covs[k])
        return log_prob

    def compute_log_likelihood_and_mean_grad(
        self, X: np.ndarray, weights: np.ndarray, means: np.ndarray, covs: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compute average incomplete-data log-likelihood (1/m) sum_i log p(x_i)
        and its exact analytical gradient with respect to `means` (shape K x n).
        """
        m = X.shape[0]
        log_wp = self._compute_log_weighted_prob(X, weights, means, covs)
        max_lp = np.max(log_wp, axis=1, keepdims=True)
        exp_shift = np.exp(log_wp - max_lp)
        sum_exp = np.sum(exp_shift, axis=1, keepdims=True)
        log_px = max_lp + np.log(sum_exp)
        avg_ll = float(np.mean(log_px))

        # Responsibilities gamma_ik = P(z_i = k | x_i)
        gamma = exp_shift / sum_exp  # (m, K)
        grad_means = np.zeros_like(means)
        for k in range(self.n_components):
            diff_k = X - means[k : k + 1, :]  # (m, n)
            inv_cov_diff = np.linalg.solve(covs[k], diff_k.T).T  # (m, n)
            grad_means[k] = np.sum(gamma[:, k : k + 1] * inv_cov_diff, axis=0) / float(m)

        return avg_ll, grad_means

    def fit(self, X: np.ndarray) -> "GaussianMixtureEMScratch":
        X_arr = np.asarray(X, dtype=np.float64)
        assert X_arr.ndim == 2
        m, n = X_arr.shape
        K = self.n_components
        rng = np.random.default_rng(self.random_state)

        if self.init_method == "kmeans++":
            km = KMeansScratch(n_clusters=K, init="kmeans++", max_iter=5, random_state=self.random_state).fit(X_arr)
            assert km.centroids_ is not None
            means = km.centroids_.copy()
        else:
            idx = rng.choice(m, size=K, replace=False)
            means = X_arr[idx].copy()

        weights = np.full(K, 1.0 / K, dtype=np.float64)
        base_cov = np.cov(X_arr, rowvar=False) + self.reg_covar * np.eye(n)
        if base_cov.ndim == 0:
            base_cov = np.array([[float(base_cov)]])
        covs = np.array([base_cov.copy() for _ in range(K)], dtype=np.float64)

        self.log_likelihood_history.clear()
        self.means_history = [means.copy()]
        self.covs_history = [covs.copy()]

        prev_ll = -np.inf
        gamma = np.full((m, K), 1.0 / K, dtype=np.float64)

        for _ in range(self.max_iter):
            # E-Step: Posterior responsibilities via Log-Sum-Exp
            log_wp = self._compute_log_weighted_prob(X_arr, weights, means, covs)
            max_lp = np.max(log_wp, axis=1, keepdims=True)
            exp_shift = np.exp(log_wp - max_lp)
            sum_exp = np.sum(exp_shift, axis=1, keepdims=True)
            gamma = exp_shift / sum_exp

            avg_ll = float(np.mean(max_lp + np.log(sum_exp)))
            self.log_likelihood_history.append(avg_ll)

            # M-Step: Update N_k, pi_k, mu_k, Sigma_k
            N_k = np.sum(gamma, axis=0) + 1e-12  # shape (K,)
            weights = N_k / float(m)

            for k in range(K):
                resp_k = gamma[:, k : k + 1]  # (m, 1)
                means[k] = np.sum(resp_k * X_arr, axis=0) / N_k[k]
                diff_k = X_arr - means[k : k + 1, :]
                covs[k] = ((resp_k * diff_k).T @ diff_k) / N_k[k] + self.reg_covar * np.eye(n)

            self.means_history.append(means.copy())
            self.covs_history.append(covs.copy())

            if abs(avg_ll - prev_ll) < self.tol:
                break
            prev_ll = avg_ll

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covs
        self.responsibilities_ = gamma
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.weights_ is not None and self.means_ is not None and self.covariances_ is not None
        X_arr = np.asarray(X, dtype=np.float64)
        log_wp = self._compute_log_weighted_prob(X_arr, self.weights_, self.means_, self.covariances_)
        max_lp = np.max(log_wp, axis=1, keepdims=True)
        exp_shift = np.exp(log_wp - max_lp)
        return exp_shift / np.sum(exp_shift, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# =============================================================================
# 3. PRINCIPAL COMPONENT ANALYSIS (COVARIANCE EIGH & SVD SOLVERS)
# =============================================================================

class PCAScratch:
    """
    First-Principles Principal Component Analysis (PCA).

    Supports:
    - `solver="svd"`: Thin Singular Value Decomposition of centered X_tilde = U Sigma V^T.
    - `solver="eigh"`: Spectral eigendecomposition of covariance S = (1/m) X_tilde^T X_tilde.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        solver: Literal["svd", "eigh"] = "svd",
    ) -> None:
        self.n_components = n_components
        self.solver = solver

        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None  # shape (k, n), rows are principal axes u_j^T
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PCAScratch":
        X_arr = np.asarray(X, dtype=np.float64)
        assert X_arr.ndim == 2
        m, n = X_arr.shape
        k = min(m, n) if self.n_components is None else int(self.n_components)

        self.mean_ = np.mean(X_arr, axis=0, keepdims=True)
        X_centered = X_arr - self.mean_

        if self.solver == "svd":
            U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
            eigvals = (s ** 2) / float(m)
            components = Vt
            self.singular_values_ = s[:k]
        elif self.solver == "eigh":
            cov = (X_centered.T @ X_centered) / float(m)
            eigvals_raw, eigvecs_raw = np.linalg.eigh(cov)
            # Sort descending
            order = np.argsort(eigvals_raw)[::-1]
            eigvals = np.maximum(eigvals_raw[order], 0.0)
            components = eigvecs_raw[:, order].T
            self.singular_values_ = np.sqrt(eigvals[:k] * m)
        else:
            raise ValueError(f"Unsupported PCA solver: {self.solver}")

        # Enforce deterministic sign convention (largest-magnitude entry of each row is positive)
        for r in range(components.shape[0]):
            max_abs_col = int(np.argmax(np.abs(components[r])))
            if components[r, max_abs_col] < 0:
                components[r] *= -1.0

        total_var = float(np.sum(eigvals))
        self.components_ = components[:k]
        self.explained_variance_ = eigvals[:k]
        self.explained_variance_ratio_ = eigvals[:k] / max(total_var, 1e-15)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X onto top-k principal axes: Z = (X - mu) @ U_k."""
        assert self.mean_ is not None and self.components_ is not None
        X_arr = np.asarray(X, dtype=np.float64)
        return (X_arr - self.mean_) @ self.components_.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct X_hat = Z @ U_k^T + mu in original n-dimensional space."""
        assert self.mean_ is not None and self.components_ is not None
        Z_arr = np.asarray(Z, dtype=np.float64)
        return Z_arr @ self.components_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @staticmethod
    def reconstruction_loss_and_grad(X_centered: np.ndarray, U: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute Frobenius reconstruction loss J(U) = (1 / 2m) ||X_c - X_c U U^T||_F^2
        and exact analytical matrix gradient dJ/dU for finite-difference verification:
            J(U) = (1 / 2m) Tr(X_c^T X_c - 2 U^T S U + U^T S U U^T U)
            nabla_U J = -(2 * S @ U - S @ U @ (U^T @ U) - U @ (U^T @ S @ U))
        where S = (1/m) X_c^T X_c.
        """
        m = X_centered.shape[0]
        recon = (X_centered @ U) @ U.T
        diff = X_centered - recon
        loss = float(np.sum(diff ** 2) / (2.0 * m))
        S = (X_centered.T @ X_centered) / float(m)
        grad_U = -(2.0 * (S @ U) - (S @ U) @ (U.T @ U) - U @ (U.T @ S @ U))
        return loss, grad_U


# =============================================================================
# 4. FINITE-DIFFERENCE GRADIENT & SOLVER PARITY VERIFICATION SUITE
# =============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("MODULE 06: k-MEANS++, GMM (EM) & PCA/SVD — VERIFICATION SUITE")
    print("=" * 72)

    rng = np.random.default_rng(42)
    m, n, k = 45, 5, 2
    X = rng.normal(size=(m, n))
    X_c = X - np.mean(X, axis=0, keepdims=True)
    U_test = rng.normal(size=(n, k))
    eps = 1e-6

    # 1. Finite-Difference Gradient Check on PCA Reconstruction Objective J(U)
    _, ana_gU = PCAScratch.reconstruction_loss_and_grad(X_c, U_test)
    num_gU = np.zeros_like(ana_gU)
    for r in range(n):
        for c in range(k):
            Up, Um = U_test.copy(), U_test.copy()
            Up[r, c] += eps
            Um[r, c] -= eps
            lp, _ = PCAScratch.reconstruction_loss_and_grad(X_c, Up)
            lm, _ = PCAScratch.reconstruction_loss_and_grad(X_c, Um)
            num_gU[r, c] = (lp - lm) / (2.0 * eps)

    rel_pca = float(np.linalg.norm(ana_gU - num_gU) / (np.linalg.norm(ana_gU) + np.linalg.norm(num_gU)))
    print(f"[Grad Check] PCA Frobenius Reconstruction Gradient | Relative Error: {rel_pca:.3e} (< 1e-7 PASS)")
    assert rel_pca < 1e-7

    # 2. Finite-Difference Gradient Check on GMM Incomplete-Data Log-Likelihood
    gmm = GaussianMixtureEMScratch(n_components=3)
    weights_t = np.array([0.3, 0.45, 0.25])
    means_t = rng.normal(size=(3, n))
    covs_t = np.array([np.eye(n) * 1.2 for _ in range(3)])
    _, ana_gM = gmm.compute_log_likelihood_and_mean_grad(X, weights_t, means_t, covs_t)
    num_gM = np.zeros_like(ana_gM)
    for r in range(3):
        for c in range(n):
            Mp, Mm = means_t.copy(), means_t.copy()
            Mp[r, c] += eps
            Mm[r, c] -= eps
            llp, _ = gmm.compute_log_likelihood_and_mean_grad(X, weights_t, Mp, covs_t)
            llm, _ = gmm.compute_log_likelihood_and_mean_grad(X, weights_t, Mm, covs_t)
            num_gM[r, c] = (llp - llm) / (2.0 * eps)

    rel_gmm = float(np.linalg.norm(ana_gM - num_gM) / (np.linalg.norm(ana_gM) + np.linalg.norm(num_gM)))
    print(f"[Grad Check] GMM Mixture Log-Likelihood Gradient  | Relative Error: {rel_gmm:.3e} (< 1e-7 PASS)")
    assert rel_gmm < 1e-7

    # 3. Verify PCA `svd` and `eigh` solvers produce identical principal axes & eigenvalues
    pca_svd = PCAScratch(n_components=3, solver="svd").fit(X)
    pca_eigh = PCAScratch(n_components=3, solver="eigh").fit(X)
    comp_diff = float(np.max(np.abs(pca_svd.components_ - pca_eigh.components_)))
    var_diff = float(np.max(np.abs(pca_svd.explained_variance_ - pca_eigh.explained_variance_)))
    print(f"[PCA Parity] ||U_SVD - U_EIGH||_inf = {comp_diff:.3e} | ||lambda_SVD - lambda_EIGH||_inf = {var_diff:.3e}")
    assert comp_diff < 1e-12 and var_diff < 1e-12

    # 4. Verify monotonic ascent of GMM EM log-likelihood
    gmm_fit = GaussianMixtureEMScratch(n_components=3, max_iter=40, random_state=7).fit(X)
    ll_diffs = np.diff(gmm_fit.log_likelihood_history)
    assert np.all(ll_diffs >= -1e-12), "EM Log-Likelihood must increase monotonically!"
    print(f"[EM Monotonicity] Log-likelihood increased monotonically across {len(gmm_fit.log_likelihood_history)} steps (PASS)")
    print("=" * 72)

