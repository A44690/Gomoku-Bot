from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gomoku_env import GomokuEnv
from CustomPolicy import CustomExtractor, CustomActorCriticPolicy
from torch import optim
from CustomCallback import CustomMaskableEvalCallback
import sys
import local_constants as c

def mask_fn(env):
    return env.legal_moves

def make_env():
    env = GomokuEnv(render=c.RENDER_MODE_TRAIN, wait_time=c.RENDER_MODE_WAIT_TIME)
    env = ActionMasker(env, mask_fn)
    return env

policy_kwargs = dict(
    features_extractor_class=CustomExtractor, 
    features_extractor_kwargs=dict(features_dim=2 * c.BOARD_SIZE * c.BOARD_SIZE + c.BOARD_SIZE * c.BOARD_SIZE), 
    optimizer_class=optim.AdamW, 
    optimizer_kwargs=dict(weight_decay=c.WEIGHT_DECAY)
    )

train_env = make_vec_env(make_env, n_envs=1)
env_eval = ActionMasker(GomokuEnv(render=c.RENDER_MODE_EVAL, eval_mode=True, wait_time=c.RENDER_MODE_WAIT_TIME), mask_fn) 

if (input("pretrain? (y/n)") == "y"):
    model_name = input("model name: ")
    model = MaskablePPO.load(c.SESSION_MODEL_PATH + model_name, env=train_env, verbose=1, n_steps=c.N_STEPS, learning_rate=c.LEARNING_RATE, policy_kwargs=policy_kwargs, tensorboard_log=c.TENSORBOARD_LOG_PATH)
    print("model loaded")
else:
    model = MaskablePPO(CustomActorCriticPolicy, env=train_env, verbose=1, n_steps=c.N_STEPS, learning_rate=c.LEARNING_RATE, policy_kwargs=policy_kwargs, tensorboard_log=c.TENSORBOARD_LOG_PATH)
    print("model created")

eval_callback = CustomMaskableEvalCallback(eval_env=env_eval, best_model_save_path=c.EVAL_MODEL_PATH, log_path=c.LOG_PATH, eval_freq=1024, deterministic=False, n_eval_episodes=2)

while True:
    time_step = int(input("Enter time step: "))
    print("training for", time_step, "time steps")
    model = model.learn(total_timesteps=time_step, progress_bar=True, log_interval=1, tb_log_name=c.LOG_PATH, reset_num_timesteps=False, callback=eval_callback)
    model.save(c.SESSION_MODEL_PATH + "ppo_model_" + str(model.num_timesteps))
    print("model saved")
    if (input("continue? (y/n)") == "n"):
        break

print("model closed")
sys.exit(0)