import torch
from torch import nn
import torch.nn.functional as F

from utils import utils

class DHEL(nn.Module):
    """
    Decoupled Hyperspherical Energy Loss (DHEL) from https://arxiv.org/abs/2405.18045.
    
    Args:
        tau (float): Temperature parameter for scaling the logits. Default is 0.3.
        include_augs (bool): Whether to include the energy of the augmentations in the denominator. Default is True.
    """

    def __init__(self, tau=0.3, include_augs=True):
        super(DHEL, self).__init__()
        self.tau = tau
        self.include_augs = include_augs

    def forward(self, z):
        """
        Forward pass for the DHEL loss calculation.

        Args:
            z (Tensor): Input tensor of shape (2M, d) where M is the batch size and d is the embedding dimension.
                        The first M elements are the embeddings of the anchor samples and the next M elements are the embeddings of their positives.

        Returns:
            loss (Tensor): Computed DHEL loss.
        """
        batch_size = z.shape[0] // 2

        # Normalize the embeddings
        z = F.normalize(z, p=2, dim=1)

        # Split the embeddings into the anchors and their positives
        z_anchor = z[:batch_size]
        z_pos = z[batch_size:]

        # Compute the similarity matrix for the anchor samples
        sim_matrix_anchor = torch.exp(torch.mm(z_anchor, z_anchor.t()) / self.tau)

        # Create a mask to exclude self-similarity
        mask = torch.eye(batch_size, device=z.device).bool()
        sim_matrix_anchor = sim_matrix_anchor.masked_fill(mask, 0)

        # Compute the positive similarities between anchor and positive samples
        pos_sim = torch.exp(torch.sum(z_anchor * z_pos, dim=-1) / self.tau)

        if self.include_augs:
            # Compute the similarity matrix for the positive samples
            sim_matrix_pos = torch.exp(torch.mm(z_pos, z_pos.t()) / self.tau)
            sim_matrix_pos = sim_matrix_pos.masked_fill(mask, 0)

            # Compute the contrastive loss including augmentations
            loss = -torch.log(pos_sim / (sim_matrix_anchor.sum(dim=-1) * sim_matrix_pos.sum(dim=-1))).mean()
        else:
            # Compute the contrastive loss without including augmentations
            loss = -torch.log(pos_sim / sim_matrix_anchor.sum(dim=-1)).mean()

        return loss


class KCL(nn.Module):
    """
    Kernel Contrastive Loss (KCL) from https://arxiv.org/abs/2405.18045.
    
    Args:
        t (float): Kernel hyperparameter. Default is 2.
        kernel (str): Type of kernel to use ('gaussian', 'log', 'imq', 'riesz', 'gaussian_riesz'). Default is 'gaussian'.
        gamma (float): Scaling parameter for the energy loss term. Default is 16.
    """

    def __init__(self, t=2.0, kernel='gaussian', gamma=16.0):
        super(KCL, self).__init__()
        self.t = t
        self.kernel = kernel
        self.gamma = gamma
        self.distributed = False  # Placeholder for distributed data parallel support

    def forward(self, z):
        """
        Forward pass for the KCL loss calculation.

        Args:
            z (Tensor): Input tensor of shape (2M, d) where M is the batch size and d is the embedding dimension.
        """
        batch_size = z.shape[0] // 2

        # Normalize the embeddings
        z = F.normalize(z, p=2, dim=1)

        # Split the embeddings into the anchors and their positives
        z_anchor = z[:batch_size]
        z_pos = z[batch_size:]

        energy, alignment = self.calculate_terms(z_anchor, z_pos)

        loss = -alignment + self.gamma * energy
        return loss

    def calculate_terms(self, z_anchor, z_pos):
        """
        Compute the energy and alignment based on the selected kernel.

        Args:
            z_anchor (Tensor): (M, d) tensor of the anchor embeddings.
            z_pos (Tensor): (M, d) tensor of the positive to the anchor embeddings
        """
        if self.kernel == "gaussian":
            energy = (utils.gaussian_kernel(z_anchor, self.t).mean() + 
                           utils.gaussian_kernel(z_pos, self.t).mean())
            alignment = 2 * utils.align_gaussian(z_anchor, z_pos, self.t)

        elif self.kernel == "log":
            energy = (utils.log_kernel(z_anchor, self.t) + 
                           utils.log_kernel(z_pos, self.t))
            alignment = 2 * utils.log_align(z_anchor, z_pos, self.t)

        elif self.kernel == "imq":
            energy = (utils.imq_kernel(z_anchor, c=self.t).mean() + 
                           utils.imq_kernel(z_pos, c=self.t).mean())
            alignment = 2 * utils.align_imq(z_anchor, z_pos, c=self.t)

        elif self.kernel == "riesz":
            energy = (utils.riesz_kernel(z_anchor, s=self.t).mean() + 
                           utils.riesz_kernel(z_pos, s=self.t).mean())
            alignment = 2 * utils.align_riesz(z_anchor, z_pos, self.t)

        elif self.kernel == "gaussian_riesz":
            energy = (utils.riesz_kernel(z_anchor, s=self.t).mean() + 
                           utils.riesz_kernel(z_pos, s=self.t).mean())
            alignment = 2 * utils.align_gaussian(z_anchor, z_pos, t=2)

        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")

        return energy, alignment