# ACG-WAM: World-Action Modeling via Action-Conditioned Geometric Latent Prediction

This repo is the official implementation of:

> **ACG-WAM: World-Action Modeling via Action-Conditioned Geometric Latent Prediction**
>
> Jiangtao Liu\*, Zishang Xiang\*, Yage He, Lingguo Cui, Baihai Zhang, Runqi Chai, and Senchun Chai<sup>†</sup>
>
> \*Equal contribution. <sup>†</sup>Corresponding author.
>
> [Paper](https://RoboOpus.github.io/ACG-WAM/assets/paper/acg-wam.pdf) | [Website](https://RoboOpus.github.io/ACG-WAM/) | [Model](https://huggingface.co/RoboOpus/ACG-WAM)

ACG-WAM trains a world-action model with **action-conditioned geometric latent prediction**. A frozen VGGT teacher supplies geometric targets from current–future image pairs. The ACG-JEPA auxiliary objective supervises the shared visual embedding before temporal mixing, using current observations, intervening actions, and multiple prediction horizons. The teacher and auxiliary modules are removed at inference.

| Evaluation | Success rate |
| --- | ---: |
| RoboTwin 2.0 · 50 tasks · clean | 93.46% |
| RoboTwin 2.0 · 50 tasks · randomized | 92.68% |
| RoboTwin 2.0 · mean across settings | 93.07% |
| Real robot · 3 tasks · mean | 85.00% |

The real-robot mean partial completion score is **91.67%**. See the manuscript for the evaluation protocol and comparisons.

## Release status

This repository is being prepared for release. Training code, installation instructions, and checkpoint details will be added here. The model link is the intended release location; availability is not implied.

The project page lives on the [`website`](https://github.com/RoboOpus/ACG-WAM/tree/website) branch. Website and paper links become available after GitHub Pages is enabled. The included PDF is the current anonymous manuscript supplied with the project; replace it when the public author version is ready.

## Citation

```bibtex
@unpublished{liu2026acgwam,
  title  = {ACG-WAM: World-Action Modeling via Action-Conditioned Geometric Latent Prediction},
  author = {Liu, Jiangtao and Xiang, Zishang and He, Yage and Cui, Lingguo and Zhang, Baihai and Chai, Runqi and Chai, Senchun},
  year   = {2026},
  note   = {Manuscript},
  url    = {https://RoboOpus.github.io/ACG-WAM/}
}
```
