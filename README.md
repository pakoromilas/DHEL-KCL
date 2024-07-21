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

The introduced **_metric_** measures the Wasserstein distance between learned and optimal similarity distributions. Unlike the conventional uniformity metric, it accurately estimates uniformity without underestimation.

## Citation

```

@InProceedings{pmlr-v235-koromilas24a,
  title = 	 {Bridging Mini-Batch and Asymptotic Analysis in Contrastive Learning: From {I}nfo{NCE} to Kernel-Based Losses},
  author =       {Koromilas, Panagiotis and Bouritsas, Giorgos and Giannakopoulos, Theodoros and Nicolaou, Mihalis and Panagakis, Yannis},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {25276--25301},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v235/main/assets/koromilas24a/koromilas24a.pdf},
  url = 	 {https://proceedings.mlr.press/v235/koromilas24a.html}
}
```
