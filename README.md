# Gomoku-Bot

A reinforcement learning environment to train Gomoku AI

Additional unsupervised training start up method is (WIP).

Based on Stable-Baselines3's maskablePPO, custom features extractor & custom actor-critic policy

## Requirements
Pygame, Numpy, Stable-Baselines3, Sb3-contrib, Gymnasium, and Pytorch are required.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install

```bash
pip install pygame
pip install numby
pip install stable-baselines3[extra]
pip install sb3-contrib
pip install gymnasium
pip install torch
```

if you want TensorBoard, add

```bash
pip install tensorflow
```

At last, in the folder that you are running this program, create four folders:

-best_ppo_models

-callback_logs

-ppo_models

-tensorboard

or change the paths according to your preferences in "local_constants.py" under #PATHS AND LOGGING.

## Get Started
Do all of those after checking the Requirements!

-In "local_constants.py", set the board size you what to train your bot on by changing the BOARD_SIZE variable

-Adjust #NN TRAINING CONSTANTS and #RENDERING OPTIONS according to your preferences

-Run "main.py" and follow the messages in the terminal.

Leave questions in Github if you encountered any trouble(s).

## Contributers & Credits
@A44690: Liuxuanhao Alex Zhao (traing environment, main, custom policy, and model testing code)

@cao1729: Kairan Cao (gomoku logic and visual)

Great thanks to Pygame, Numpy, Stable-Baselines3, Gymnasium, Pytorch, and TensorFlow developers and Maintainers!


### Paper(s): 

He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition (Version 1). Version 1. arXiv. https://doi.org/10.48550/ARXIV.1512.03385

Silver, D., Schrittwieser, J., Simonyan, K., Antonoglou, I., Huang, A., Guez, A., … Hassabis, D. (2017). Mastering the game of Go without human knowledge. Nature, 550(7676), 354–359. https://doi.org/10.1038/nature24270


### Special Thanks:

@Rin~ for giving us a Nature subscription so we can reference papers on Nature.