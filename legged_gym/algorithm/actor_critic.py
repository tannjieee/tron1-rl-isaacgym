# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn


class ActorCritic(nn.Module):
    is_recurrent = False
    is_sequence = False
    is_vae = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        cost_critic_hidden_dims=None,
        activation="elu",
        orthogonal_init=False,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(ActorCritic, self).__init__()

        self.orthogonal_init = orthogonal_init
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs

        activation = get_activation(activation)

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        if self.orthogonal_init:
            torch.nn.init.orthogonal_(actor_layers[-1].weight, np.sqrt(2))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(actor_layers[-1].weight, 0.01)
                    torch.nn.init.constant_(actor_layers[-1].bias, 0.0)
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1])
                )
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(actor_layers[-1].weight, np.sqrt(2))
                    torch.nn.init.constant_(actor_layers[-1].bias, 0.0)
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Reward value function
        critic_layers = self._build_value_mlp(
            num_critic_obs,
            critic_hidden_dims,
            activation,
            orthogonal_init,
        )
        self.critic = nn.Sequential(*critic_layers)

        # Cost value function for NP3O / P3O. Keeping it inside ActorCritic lets
        # the checkpoint contain both reward and cost critics and keeps the PPO
        # optimizer unchanged.
        if cost_critic_hidden_dims is None:
            cost_critic_hidden_dims = critic_hidden_dims
        cost_critic_layers = self._build_value_mlp(
            num_critic_obs,
            cost_critic_hidden_dims,
            activation,
            orthogonal_init,
        )
        self.cost_critic = nn.Sequential(*cost_critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Cost critic MLP: {self.cost_critic}")

        self.logstd = nn.Parameter(torch.zeros(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def _build_value_mlp(self, num_inputs, hidden_dims, activation, orthogonal_init):
        layers = []
        layers.append(nn.Linear(num_inputs, hidden_dims[0]))
        layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], 1))
                if orthogonal_init:
                    torch.nn.init.orthogonal_(layers[-1].weight, 0.01)
                    torch.nn.init.constant_(layers[-1].bias, 0.0)
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                if orthogonal_init:
                    torch.nn.init.orthogonal_(layers[-1].weight, np.sqrt(2))
                    torch.nn.init.constant_(layers[-1].bias, 0.0)
                layers.append(activation)
        return layers

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + torch.exp(self.logstd))

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def evaluate_cost(self, critic_observations, **kwargs):
        cost_value = self.cost_critic(critic_observations)
        return cost_value


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
