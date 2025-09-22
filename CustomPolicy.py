from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
import gymnasium.spaces as spaces

class residual_block(nn.Module):
    '''from https://arxiv.org/abs/1512.03385 and a bit googled'''
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(channels), 
            nn.ReLU(), 
            nn.Dropout2d(p=0.15), 
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(channels), 
            nn.ReLU()
        )

    def forward(self, x):
        return th.relu(x + self.block(x))

class CustomExtractor(BaseFeaturesExtractor):
    '''Custom extractor for Gomoku environment. Some copied from stable_baselines3, but parameters of the conv layers are changed.'''
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False,) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(n_input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.ResBlocks = nn.Sequential(*[residual_block(128) for i in range(8)]) # 8 residual blocks
        self.flatten = nn.Flatten()
        
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.flatten(self.ResBlocks(self.stem(sample_input))).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), 
            nn.ReLU(), 
            nn.Dropout(p=0.1)
            )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.flatten(self.ResBlocks(self.stem(observations))))

class CustomNetwork(nn.Module):
    '''You guessed it, from satable_baselines3's documentation...'''
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 512,
        last_layer_dim_vf: int = 512,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        # watch gpu memory usage
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.first_hidden_dim = 512
        self.second_hidden_dim = 1024
        self.third_hidden_dim = 512

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, self.first_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.first_hidden_dim, self.second_hidden_dim),
            nn.BatchNorm1d(self.second_hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.second_hidden_dim, self.third_hidden_dim),
            nn.BatchNorm1d(self.third_hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.third_hidden_dim, last_layer_dim_pi),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, self.first_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.first_hidden_dim, self.second_hidden_dim),
            nn.BatchNorm1d(self.second_hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.second_hidden_dim, self.third_hidden_dim),
            nn.BatchNorm1d(self.third_hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.third_hidden_dim, last_layer_dim_vf),
        )
        
    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.policy_net(features), self.value_net(features)
    
    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)
    
class CustomActorCriticPolicy(MaskableActorCriticPolicy):
    '''sb3's ActorCriticPolicy but with custom network and custom extractor'''
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ): 
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
    
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)