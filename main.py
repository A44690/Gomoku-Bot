from stable_baselines3 import PPO
from gomoku_env import GomokuEnv
from CustomPolicy import CustomCnnPolicy
import sys

policy_kwargs = dict(features_extractor_class=CustomCnnPolicy, features_extractor_kwargs=dict(features_dim=512),)

env = GomokuEnv()
if (input("pretrain? (y/n)") == "y"):
    model_name = input("model name: ")
    model = PPO.load("ppo_models/" + model_name, env=env, verbose=1, n_steps=128, learning_rate=0.0001, device="cuda", tensorboard_log="./tensorboard/")
    print("model loaded")
else:
    model = PPO("CnnPolicy", env, verbose=1, n_steps=128, learning_rate=0.0001, policy_kwargs=policy_kwargs, device="cuda", tensorboard_log="./tensorboard/")
    print("model created")

while True:
    time_step = int(input("Enter time step: "))
    print("training for", time_step, "time steps")
    env.reset()
    model = model.learn(total_timesteps=time_step, progress_bar=True, log_interval=1, tb_log_name="ppo_model", reset_num_timesteps=False)
    model.save("ppo_models/ppo_model_" + str(model.num_timesteps))
    print("model saved")
    env.reset()
    if (input("continue? (y/n)") == "n"):
        break

print("model closed")
sys.exit(0)