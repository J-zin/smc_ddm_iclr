<h1 align="center">Discrete Diffusion SMC</h1>

This repo contains PyTorch implementation of the paper "[Inference-Time Scaling of Discrete Diffusion Models via Importance Weighting and Optimal Proposal Design](https://arxiv.org/abs/2505.22524)"

by [Zijing Ou](https://j-zin.github.io/), [Chinmay Pani](https://scholar.google.com/citations?user=oqy71RQAAAAJ&hl=en) and [Yingzhen Li](http://yingzhenli.net/home/en/).

> We propose a Sequential Monte Carlo (SMC) framework that enables scalable inference-time control of discrete diffusion models through principled importance weighting and optimal proposal construction. Our approach derives tractable importance weights for a range of intermediate targets and characterises the optimal proposal, for which we develop two practical approximations: a first-order gradient-based approximation and an amortised proposal trained to minimise the log-variance of the importance weights.


In this repository, we provide the training and evaluation code for the experiements of language modelling, biology design, and image generation.
For the toy synthetic experiments, please refer to the [codebase](https://github.com/J-zin/smc_ddm).

## Citation
:smile:If you find this repo is useful, please consider to cite our paper:
```
@article{ou2025inference,
  title={Inference-Time Scaling of Discrete Diffusion Models via Importance Weighting and Optimal Proposal Design},
  author={Ou, Zijing and Pani, Chinmay and Li, Yingzhen},
  journal={arXiv preprint arXiv:2505.22524},
  year={2025}
}
```