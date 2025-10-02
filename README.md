# Gomoku-Bot

A reinforcement learning environment to train Gomoku AI

Based on Stable-Baselines3's maskable, custom features extractor & custom actor-critic policy

A major update about MCTS is planned.

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

Paper(s): 

He, K., Zhang, X., Ren, S., & Sun, J. (2015, December 10). Deep residual learning for image recognition. arXiv.org. https://arxiv.org/abs/1512.03385 