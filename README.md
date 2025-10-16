# Gomoku-Bot

A reinforcement learning environment to train Gomoku AI

Based on Stable-Baselines3's maskablePPO, custom features extractor & custom actor-critic policy

A human training mode will be added soon and a update about MCTS is planned.

## Required Packages
Pygame, Numpy, Stable-Baselines3, Sb3-contrib, Gymnasium, and Pytorch required

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install

```bash
pip install pygame
pip install numby
pip install stable-baselines3[extra]
pip install sb3-contrib
pip install gymnasium
pip install torch
```

If you want TensorBoard, add

```bash
pip install tensorflow
```

## Contributers & Credits
@A44690: Liuxuanhao Alex Zhao (traing environment, main, custom policy, and model testing code)

@cao1729: Kairan Cao (gomoku logic and visual)

Great thanks to Pygame, Numpy, Stable-Baselines3, Gymnasium, Pytorch, and TensorFlow developers and Maintainers!


### Paper(s): 

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition (Version 1). Version 1. arXiv. https://doi.org/10.48550/ARXIV.1512.03385

Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., … Hassabis, D. (2017). Mastering the game of Go without human knowledge. Nature, 550(7676), 354–359. https://doi.org/10.1038/nature24270


### Special Thanks:

@Rin~ for giving us a Nature subscription so we can reference papers on Nature.