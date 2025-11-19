from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gomoku_env import GomokuEnv
from CustomPolicy import CustomExtractor, CustomActorCriticPolicy
from torch import optim
import sys
import numpy as np
import pygame
import time

#ai vs ai

def mask_fn(env):
    return env.legal_moves
env = GomokuEnv(render=True, wait_time = 1.0)
env = ActionMasker(env, mask_fn)
policy_kwargs = dict(features_extractor_class=CustomExtractor, features_extractor_kwargs=dict(features_dim=2*19*19 + 19*19), optimizer_class=optim.AdamW, optimizer_kwargs=dict(weight_decay=1e-5))
model1 = MaskablePPO.load("best_ppo_models/best_model", env=env, verbose=1, policy_kwargs=policy_kwargs)
done = False
for i in range(0, 7):
    sys.stdout.write("\033[F" + "\033[K")
input("start demo")
obs, info = env.reset()
while not done:
    action, _states = model1.predict(obs, deterministic=False, action_masks=get_action_masks(env))
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        time.sleep(2)
input("demo finished")
env.close()
sys.exit(0)