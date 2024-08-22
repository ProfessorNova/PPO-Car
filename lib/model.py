import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=256):
        super(Agent, self).__init__()

        self.actor = nn.Sequential(
            layer_init(nn.Linear(num_inputs, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, num_outputs), std=0.01)
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(num_inputs, hidden_size)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0)
        )

    def forward(self, x):
        return self.actor(x)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, logprob, entropy, self.get_value(x)
