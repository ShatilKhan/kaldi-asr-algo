"""
Diagonal covariance Gaussian Mixture Model (paper Section IV-A).

A GMM models the distribution of audio features for one HMM state.
It is a weighted sum of K multivariate Gaussians with diagonal covariance.

Methods:
    - log_likelihood(frame): score one frame against the GMM
    - train(frames, n_components): EM training from scratch
    - split(): double the number of components (for incremental training)

Key insight: diagonal covariance means each dimension is independent,
so we only store a variance vector, not a full matrix.
"""

import numpy as np
import math
from typing import Optional

EPS = 1e-10  # small constant to avoid log(0) and division by zero


class DiagGmm:
    """
    Gaussian Mixture Model with diagonal covariance.

    Attributes:
        means: (K, D) array — center of each component
        vars:  (K, D) array — variance of each component (must be > 0)
        weights: (K,) array — component weights (sum to 1)
        K: number of components
        D: feature dimension
    """

    def __init__(self, means: np.ndarray, vars: np.ndarray, weights: np.ndarray):
        assert means.shape == vars.shape
        assert weights.ndim == 1
        assert len(weights) == means.shape[0]
        self.means = means.astype(np.float64)
        self.vars = np.maximum(vars.astype(np.float64), EPS)
        self.weights = weights.astype(np.float64)
        self.K = self.means.shape[0]
        self.D = self.means.shape[1]
        # Pre-compute normalization constant for efficiency
        self._log_det = -0.5 * np.sum(np.log(self.vars), axis=1) - 0.5 * self.D * np.log(2.0 * np.pi)

    def component_log_likelihood(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute log-likelihood for each component individually.

        log_prob_k = -0.5 * [D*log(2π) + sum(log(vars_k)) + mahalanobis_k]

        where mahalanobis_k = sum((x - mean_k)^2 / vars_k)

        Args:
            frame: (D,) array.

        Returns:
            (K,) array of log-likelihoods per component.
        """
        diff = frame - self.means  # (K, D)
        mahalanobis = np.sum(diff * diff / self.vars, axis=1)  # (K,)
        return self._log_det - 0.5 * mahalanobis

    def log_likelihood(self, frame: np.ndarray) -> float:
        """
        Compute the total log-likelihood of a frame under the GMM.

        log P(x) = log( sum_k w_k * N(x | mean_k, var_k) )

        Uses log-sum-exp for numerical stability.

        Args:
            frame: (D,) array.

        Returns:
            Scalar log-likelihood.
        """
        log_probs = self.component_log_likelihood(frame)  # (K,)
        log_weighted = log_probs + np.log(self.weights)
        # log-sum-exp trick
        max_log = np.max(log_weighted)
        return max_log + np.log(np.sum(np.exp(log_weighted - max_log)))

    def score_batch(self, frames: np.ndarray) -> np.ndarray:
        """
        Score a batch of frames against this GMM.

        Args:
            frames: (N, D) array.

        Returns:
            (N,) array of log-likelihoods.
        """
        # Vectorized: (N, K, D) diff, (N, K) mahalanobis, (N, K) log_probs
        diff = frames[:, None, :] - self.means[None, :, :]  # (N, K, D)
        mahalanobis = np.sum(diff * diff / self.vars[None, :, :], axis=2)  # (N, K)
        log_probs = self._log_det[None, :] - 0.5 * mahalanobis  # (N, K)
        # Log-sum-exp over components
        log_weighted = log_probs + np.log(self.weights)[None, :]
        max_log = np.max(log_weighted, axis=1, keepdims=True)
        return (np.log(np.sum(np.exp(log_weighted - max_log), axis=1)) + max_log.squeeze(1))

    def score_batch_all(gmms: list, frames: np.ndarray) -> np.ndarray:
        """
        Score a batch of frames against ALL GMMs (vectorized across GMMs).

        Args:
            gmms: list of M DiagGmm objects.
            frames: (N, D) array.

        Returns:
            (N, M) array of log-likelihoods.
        """
        M = len(gmms)
        N = frames.shape[0]
        D = frames.shape[1]
        K = gmms[0].K  # assume all have same K

        # Stack all GMM parameters
        all_means = np.array([g.means for g in gmms])  # (M, K, D)
        all_vars = np.array([g.vars for g in gmms])  # (M, K, D)
        all_weights = np.array([g.weights for g in gmms])  # (M, K)
        all_log_det = np.array([g._log_det for g in gmms])  # (M, K)

        # Score: (N, M, K, D) diff
        diff = frames[:, None, None, :] - all_means[None, :, :, :]  # (N, M, K, D)
        mahalanobis = np.sum(diff * diff / all_vars[None, :, :, :], axis=3)  # (N, M, K)
        log_probs = all_log_det[None, :, :] - 0.5 * mahalanobis  # (N, M, K)

        # Log-sum-exp over K components
        log_weighted = log_probs + np.log(all_weights)[None, :, :]
        max_log = np.max(log_weighted, axis=2, keepdims=True)
        return np.log(np.sum(np.exp(log_weighted - max_log), axis=2)) + max_log.squeeze(2)

    def split(self, noise: float = 0.2) -> "DiagGmm":
        """
        Double the number of components by perturbing each mean.

        Each component is split into two: one shifted by +noise*std,
        one shifted by -noise*std. Variance is doubled (more spread).

        Args:
            noise: perturbation factor relative to std.

        Returns:
            New DiagGmm with 2*K components.
        """
        stds = np.sqrt(self.vars)
        # Perturb means
        means_plus = self.means + noise * stds
        means_minus = self.means - noise * stds
        new_means = np.vstack([means_plus, means_minus])

        # Double variance (more spread from splitting)
        new_vars = np.vstack([self.vars * 2.0, self.vars * 2.0])

        # Halve weights (each child gets half the parent's weight)
        new_weights = np.concatenate([self.weights * 0.5, self.weights * 0.5])

        return DiagGmm(new_means, new_vars, new_weights)

    def to_dict(self) -> dict:
        """Serialize to a dict (for saving)."""
        return {
            "means": self.means.tolist(),
            "vars": self.vars.tolist(),
            "weights": self.weights.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DiagGmm":
        """Deserialize from a dict."""
        return cls(
            np.array(data["means"]),
            np.array(data["vars"]),
            np.array(data["weights"]),
        )

    def __repr__(self) -> str:
        return f"DiagGmm(K={self.K}, D={self.D})"


# ---- EM training ----

def train_gmm(
    frames: np.ndarray,
    n_components: int = 1,
    n_iter: int = 20,
    reg: float = 1e-4,
) -> DiagGmm:
    """
    Train a diagonal GMM via Expectation-Maximization.

    Args:
        frames: (N, D) array of training data.
        n_components: number of Gaussian components.
        n_iter: number of EM iterations.
        reg: regularization added to variance to prevent collapse.

    Returns:
        Trained DiagGmm.
    """
    N, D = frames.shape
    if N == 0:
        raise ValueError("Cannot train GMM on empty frame set")

    # Initialize: k-means-like or random
    means = _kmeans_init(frames, n_components)
    vars_ = np.tile(np.var(frames, axis=0), (n_components, 1)) + reg
    weights = np.ones(n_components) / n_components

    gmm = DiagGmm(means, vars_, weights)

    for iteration in range(n_iter):
        # --- E-step: compute responsibilities ---
        log_probs = np.zeros((N, n_components))
        for k in range(n_components):
            # Compute log-likelihood of all frames under component k
            diff = frames - gmm.means[k]  # (N, D)
            mahalanobis = np.sum(diff * diff / gmm.vars[k], axis=1)
            log_probs[:, k] = gmm._log_det[k] - 0.5 * mahalanobis

        # log_responsibilities = log(weights) + log_probs, then normalize
        log_weighted = log_probs + np.log(gmm.weights)  # (N, K)
        # log-sum-exp across components to get log-likelihood per frame
        max_log = np.max(log_weighted, axis=1, keepdims=True)
        log_sum = np.log(np.sum(np.exp(log_weighted - max_log), axis=1)) + max_log.squeeze(1)
        log_resp = log_weighted - log_sum[:, None]  # (N, K)
        resp = np.exp(log_resp)  # (N, K) — responsibilities, each row sums to 1

        # --- M-step: update parameters ---
        Nk = np.sum(resp, axis=0) + EPS  # (K,), count per component

        # Update means
        new_means = (resp.T @ frames) / Nk[:, None]  # (K, D)

        # Update variances: sum(resp_k * (x - mean_k)^2) / Nk
        diff = frames[:, None, :] - new_means[None, :, :]  # (N, K, D)
        sq_diff = diff ** 2
        new_vars = np.sum(resp[:, :, None] * sq_diff, axis=0) / Nk[:, None] + reg

        # Update weights
        new_weights = Nk / np.sum(Nk)

        gmm = DiagGmm(new_means, new_vars, new_weights)

    return gmm


def _kmeans_init(frames: np.ndarray, n_components: int, n_iter: int = 10) -> np.ndarray:
    """
    Simple k-means initialization for GMM means.
    Uses random selection from data, then Lloyd's algorithm.
    """
    N, D = frames.shape
    # Randomly select initial means from data
    indices = np.random.choice(N, size=n_components, replace=False)
    means = frames[indices].copy()

    for _ in range(n_iter):
        # Assign each frame to nearest mean
        distances = np.zeros((N, n_components))
        for k in range(n_components):
            diff = frames - means[k]
            distances[:, k] = np.sum(diff ** 2, axis=1)
        labels = np.argmin(distances, axis=1)

        # Update means
        for k in range(n_components):
            mask = labels == k
            if np.sum(mask) > 0:
                means[k] = np.mean(frames[mask], axis=0)

    return means


if __name__ == "__main__":
    # Simple test: generate synthetic data from known GMM, recover it
    np.random.seed(42)

    # True GMM with 2 components
    true_means = np.array([[2.0, 3.0], [-2.0, -3.0]])
    true_vars = np.array([[0.5, 0.5], [0.8, 0.8]])
    true_weights = np.array([0.6, 0.4])
    true_gmm = DiagGmm(true_means, true_vars, true_weights)

    # Generate samples
    frames = []
    for k in range(2):
        n = int(500 * true_weights[k])
        samples = np.random.randn(n, 2) * np.sqrt(true_vars[k]) + true_means[k]
        frames.append(samples)
    frames = np.vstack(frames)
    np.random.shuffle(frames)

    print(f"Generated {frames.shape[0]} frames from true GMM")

    # Train a 2-component GMM
    trained_gmm = train_gmm(frames, n_components=2, n_iter=30)
    print(f"\nTrained GMM: {trained_gmm}")
    print(f"  Means:\n{trained_gmm.means}")
    print(f"  Weights: {trained_gmm.weights}")

    # Score the first frame
    score = trained_gmm.log_likelihood(frames[0])
    print(f"\n  Log-likelihood of first frame: {score:.4f}")
    print(f"  (higher = better, should be around -3 to -5 for this data)")

    # Test split
    split_gmm = trained_gmm.split()
    print(f"\nAfter split: {split_gmm}")
    print(f"  Means:\n{split_gmm.means}")
    score_split = split_gmm.log_likelihood(frames[0])
    print(f"  Log-likelihood of first frame with split GMM: {score_split:.4f}")

    # Test batch scoring
    scores = trained_gmm.score_batch(frames[:5])
    print(f"\nBatch scores (first 5 frames): {scores}")
