import numpy as np
import torch
import scipy
from scipy.special import gamma
from scipy.stats import wasserstein_distance, entropy


class DotProdDist(scipy.stats.rv_continuous):
    def _pdf(self, x, d):
        """
        Probability density function for the dot product distribution that is presented in the appendix of https://arxiv.org/abs/2405.18045
        """
        a = gamma((d + 1) / 2) / (gamma(d / 2) * np.sqrt(np.pi))
        return np.power(np.sqrt(1 - x**2), d - 2) * a


def alignment(x, y, alpha=2):
    """
    Alignment calculation between  between anchor embeddings and their positives.
    Implementation from https://github.com/SsnL/align_uniform
    """
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(x, t=2):
    """
    Measure uniformity of embeddings based on the potential of the gaussian kernel.
    Implementation from https://github.com/SsnL/align_uniform
    """
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def wasserstein_uniformity(z, sub_sample=False):
    """
    Calculate the Wasserstein uniformity  metric presented in https://arxiv.org/abs/2405.18045 
    for given set of embeddings.
    """
    # Sample from the optimal distribution presented in the appendix of https://arxiv.org/abs/2405.18045
    optimal_dist = DotProdDist(a=-1, b=1)
    optimal_samples = optimal_dist.rvs(d=127, size=10000)
    
    # Calculate the similarity matrix
    similarity = torch.matmul(z, z.T)
    
    # Extract the upper triangular part of the similarity matrix, excluding the diagonal
    similarity = torch.triu(similarity, diagonal=1)
    triu_indices = np.triu_indices_from(similarity, k=1)
    s = similarity[triu_indices].cpu().numpy()  # Convert to NumPy array

    # Optionally subsample the similarities in case of conputational constraints
    if sub_sample:
        s = np.random.choice(s, size=50000, replace=False)

    # Calculate the Wasserstein distance between the sampled similarities and the optimal samples
    return wasserstein_distance(s, optimal_samples)


def rank(z, eps=1e-5):
    """
    Calculate the rank of the covariance matrix of embeddings z.
    """
    cov = np.cov(z.T)
    return np.linalg.matrix_rank(cov, tol=eps)


def effective_rank(z, eps=1e-7):
    """
    Implementation of the effective rank metric presented in https://proceedings.mlr.press/v202/garrido23a.html
    for given set of embeddings.
    """
    s = scipy.linalg.svd(z, full_matrices=False, compute_uv=False)
    s_norm = np.linalg.norm(s, 1)

    # Adjust the singular values for numerical stability
    if s_norm < eps:
        s *= 10
    elif s_norm > 1000:
        s /= 10

    # Normalize the singular values and add epsilon to avoid division by zero
    s = s / s_norm + eps
   
   # Return the entropy of the normalized singular values
    return np.exp(entropy(s))
