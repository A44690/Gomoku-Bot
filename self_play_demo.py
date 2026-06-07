from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gomoku_env import GomokuEnv
from CustomPolicy import CustomExtractor, CustomActorCriticPolicy
import sys
import numpy as np
import pygame
import torch
import time
import local_constants as c

#ai vs ai

def mask_fn(env):
    return env.legal_moves
env = GomokuEnv(render=True, wait_time = 1.0)
env = ActionMasker(env, mask_fn)
policy_kwargs = dict(
    features_extractor_class=CustomExtractor, 
    features_extractor_kwargs=dict(features_dim=2 * c.BOARD_SIZE * c.BOARD_SIZE), 
    optimizer_class=torch.optim.AdamW, 
    optimizer_kwargs=dict(weight_decay=c.WEIGHT_DECAY)
)
model1 = MaskablePPO.load(c.EVAL_MODEL_PATH + "best_model", env=env, verbose=1, policy_kwargs=policy_kwargs)
model1.n_steps = c.N_STEPS
model1.batch_size = c.BATCH_SIZE
model1.n_epochs = c.N_EPOCHS
model1.learning_rate = c.LEARNING_RATE
model1.clip_range = c.CLIP_RANGE
model1.gamma = c.GAMMA
model1.policy = torch.compile(model1.policy)
model1.policy.to("cuda" if torch.cuda.is_available() else "cpu")
model1._setup_model()  # to update the optimizer with the new parameters
done = False
for i in range(0, 7):
    sys.stdout.write("\033[F" + "\033[K")
input("start demo")
obs, info = env.reset()
while not done:
    action, _states = model1.predict(obs, deterministic=True, action_masks=get_action_masks(env))
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        time.sleep(2)
input("demo finished")
env.close()
sys.exit(0)