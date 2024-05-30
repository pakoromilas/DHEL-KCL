# Bridging Mini-Batch and Asymptotic Analysis in Contrastive Learning: From InfoNCE to Kernel-Based Losses

This repository offers implementations for two loss functions: the Decoupled Hyperspherical Energy Loss (DHEL) and the Kernel Contrastive Loss (KCL), as presented in the paper available [here](https://arxiv.org/abs/2405.18045). Additionally, it includes the metrics applied on the learned representations, such as the introduced Wasserstein distance, which measures uniformity and effective rank.

Our paper's experiments were conducted using the codebase provided in [this](https://github.com/AndrewAtanov/simclr-pytorch) repository.

**_DHEL_** and **_KCL_**:
- outperform other InfoNCE variants, such as SimCLR and DCL, even with smaller batch sizes
- demonstrate robustness against hyperparameters
- effectively utilize more dimensions, mitigating the dimensionality collapse problem

Also, **_KCL_** possesses several intriguing properties:
- the expected loss remains unaffected by the number of negative samples
- its minima can be identified non-asymptotically.

The introduced **I** measures the Wasserstein distance between learned and optimal similarity distributions. Unlike the conventional uniformity metric, it accurately estimates uniformity without underestimation.

## Citation

```
@misc{koromilas2024bridging,
      title={Bridging Mini-Batch and Asymptotic Analysis in Contrastive Learning: From InfoNCE to Kernel-Based Losses}, 
      author={Panagiotis Koromilas and Giorgos Bouritsas and Theodoros Giannakopoulos and Mihalis Nicolaou and Yannis Panagakis},
      year={2024},
      eprint={2405.18045},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
