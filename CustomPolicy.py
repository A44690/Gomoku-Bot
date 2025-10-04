from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import gymnasium as gym
import torch as th
from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
import gymnasium.spaces as spaces

class residual_block(nn.Module):
    '''from https://doi.org/10.48550/arXiv.1512.03385. not the basic resnet block in the paper, but the one used in AlphaGo Zero'''
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, stride=1, dropout_prob=0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=1), 
            nn.BatchNorm2d(mid_channels), 
            nn.ReLU(), 
            nn.Dropout2d(p=dropout_prob), 
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=1), 
            nn.BatchNorm2d(mid_channels)
        )

    def forward(self, x):
        return th.relu(x + self.block(x))

class CustomExtractor(BaseFeaturesExtractor):
    '''Custom extractor for Gomoku environment. Some copied from stable_baselines3, but parameters of the conv layers are changed.'''
    
    def __init__(self, observation_space: gym.Space, features_dim: int = 2*19*19 + 19*19, normalized_image: bool = False,) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        # print("Input channels:", n_input_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(n_input_channels, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.Res_blocks = nn.Sequential(*[residual_block(256, 256, 256, dropout_prob=0.1) for i in range(20)])
        self.to_val_features = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.to_policy_features = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).float()
            x = self.stem(sample_input)
            x = self.Res_blocks(x)
            x_val = self.to_val_features(x)
            x_pol = self.to_policy_features(x)
            # print("Value feature extractor output dimension:", x_val.shape)
            # print("Policy feature extractor output dimension:", x_pol.shape)
            self._features_dim = x_val.shape[1] + x_pol.shape[1]
            # print("Feature extractor output dimension:", self._features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.stem(observations)
        x = self.Res_blocks(x)
        val_features = self.to_val_features(x)
        policy_features = self.to_policy_features(x)
        return th.cat((val_features, policy_features), dim=1)

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
        last_layer_dim_pi: int = 19*19,
        last_layer_dim_vf: int = 1,
        policy_features_dim = 2*19*19,
        value_features_dim = 19*19
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        # watch gpu memory usage
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.feature_dim = feature_dim
        self.policy_features_dim = policy_features_dim
        self.value_features_dim = -value_features_dim
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(2*19*19, 19*19),
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(19*19, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 1),
            nn.Tanh(),
        )
        
    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)
    
    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        features = features[:, :self.policy_features_dim]
        # print("features shape in forward_actor:", features.shape)
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        features = features[:, self.value_features_dim:]
        # print("features shape in forward_critic:", features.shape)
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
        
'''
this is a mimic of AlphaGo Zero's architecture, with some changes to reduce overfitting.
while sb3's ppo actor-critic architecture is obviosuly not built to be modified this way, it is still possible to do so with some workarounds. 
(which is combining the features for policy and value network into one tensor in features extractor, then splitting them again in the custom network)
AlphaGo Zero's architecture is in https://doi.org/10.1038/nature24270
'''