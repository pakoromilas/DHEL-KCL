import torch

def gaussian_kernel(x, t=2):
    """
    Compute the pairwise potential (energy) based on the Gaussian kernel.
    
    Args:
        x (Tensor): Input tensor of shape (M, d) where M is the number of samples and d is the embedding dimension.
        t (float): Scaling parameter. Default is 2.
    """
    pairwise_distances = torch.pdist(x, p=2)
    return pairwise_distances.pow(2).mul(-t).exp().mean()


def align_gaussian(x, y, t):
    """
    Compute the alignment between anchor points and their positives based on the Gaussian kernel.
    
    Args:
        x (Tensor): Tensor of shape (M, d) containing anchor embeddings.
        y (Tensor): Tensor of shape (M, d) containing the corresponding positive embeddings.
        t (float): Scaling parameter.
    """
    pairwise_distances = (x - y).norm(p=2, dim=1)
    return pairwise_distances.pow(2).mul(-t).exp().mean()


def riesz_kernel(x, s):
    """
    Compute the pairwise potential (energy) based on the Riesz kernel.
    
    Args:
        x (Tensor): Input tensor of shape (M, d) where M is the number of samples and d is the embedding dimension.
        s (float): Scaling parameter
    """
    pairwise_distances = torch.pdist(x, p=2)
    if s < 0:
        return -pairwise_distances.pow(-s)
    else:
        return pairwise_distances.pow(-s)


def align_riesz(x, y, s):
    """
    Compute the alignment between anchor points and their positives based on the Riesz kernel.
    
    Args:
        x (Tensor): Tensor of shape (M, d) containing anchor embeddings.
        y (Tensor): Tensor of shape (M, d) containing the corresponding positive embeddings.
        s (float): Scaling parameter.
    """
    pairwise_distances = (x - y).norm(p=2, dim=1)
    if s < 0:
        return -pairwise_distances.pow(-s).mean()
    else:
        return pairwise_distances.pow(-s).mean()
    

def log_kernel(x, t, p=2):
    """
    Compute the pairwise potential (energy) based on the logarithmic kernel.
    
    Args:
        x (Tensor): Input tensor of shape (M, d) where M is the number of samples and d is the embedding dimension.
        t (float): Scaling parameter.
        p (int): Power parameter. Default is 2.
    """
    pairwise_distances = torch.pdist(x, p=p)
    return -0.5 * torch.log(t * pairwise_distances.pow(p) + 1).mean()


def log_align(x, y, t, p=2):
    """
    Compute the alignment between anchor points and their positives based on the logarithmic kernel.
    
    Args:
        x (Tensor): Tensor of shape (M, d) containing anchor embeddings.
        y (Tensor): Tensor of shape (M, d) containing the corresponding positive embeddings.
        t (float): Scaling parameter.
        p (int): Power parameter. Default is 2.
    """
    pairwise_distances = (x - y).norm(p=p, dim=1)
    return -torch.log(t * pairwise_distances.pow(p) + 1).mean()


def imq_kernel(x, c=2):
    """
    Compute the pairwise potential (energy) based on the Inverse Multi-Quadratic (IMQ) kernel.
    
    Args:
        x (Tensor): Input tensor of shape (M, d) where M is the number of samples and d is the embedding dimension.
        c (float): Scaling parameter. Default is 2.
    """
    pairwise_distances = torch.pdist(x, p=2)
    return (c / (c**2 + pairwise_distances.pow(2)).sqrt()).mean()


def align_imq(x, y, c):
    """
    Compute the alignment between anchor points and their positives based on the Inverse Multi-Quadratic (IMQ) kernel.
    
    Args:
        x (Tensor): Tensor of shape (M, d) containing anchor embeddings.
        y (Tensor): Tensor of shape (M, d) containing the corresponding positive embeddings.
        c (float): Scaling parameter.
    """
    pairwise_distances = (x - y).norm(p=2, dim=1)
    return (c / (c**2 + pairwise_distances.pow(2)).sqrt()).mean()
