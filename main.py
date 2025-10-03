from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gomoku_env import GomokuEnv
from CustomPolicy import CustomExtractor, CustomActorCriticPolicy
from torch import optim
from CustomCallback import CustomMaskableEvalCallback
import sys

def mask_fn(env):
    return env.legal_moves

policy_kwargs = dict(features_extractor_class=CustomExtractor, features_extractor_kwargs=dict(features_dim=512), optimizer_class=optim.AdamW, optimizer_kwargs=dict(weight_decay=1e-5))

train_env = ActionMasker(GomokuEnv(render=False), mask_fn)
env_eval = ActionMasker(GomokuEnv(render=False, wait_time=0.4, eval_mode=True), mask_fn) 

if (input("pretrain? (y/n)") == "y"):
    model_name = input("model name: ")
    model = MaskablePPO.load("ppo_models/" + model_name, env=train_env, verbose=1, n_steps=512, learning_rate=1e-4, policy_kwargs=policy_kwargs, tensorboard_log="./tensorboard/")
    print("model loaded")
else:
    model = MaskablePPO(CustomActorCriticPolicy, env=train_env, verbose=1, n_steps=512, learning_rate=1e-4, policy_kwargs=policy_kwargs, tensorboard_log="./tensorboard/")
    print("model created")

eval_callback = CustomMaskableEvalCallback(eval_env=env_eval, best_model_save_path="./best_ppo_models/", log_path="./callback_logs/", eval_freq=512, deterministic=True, n_eval_episodes=1)

while True:
    time_step = int(input("Enter time step: "))
    print("training for", time_step, "time steps")
    model = model.learn(total_timesteps=time_step, progress_bar=True, log_interval=1, tb_log_name="ppo_model", reset_num_timesteps=False, callback=eval_callback)
    model.save("ppo_models/ppo_model_" + str(model.num_timesteps))
    print("model saved")
    if (input("continue? (y/n)") == "n"):
        break

print("model closed")
sys.exit(0)