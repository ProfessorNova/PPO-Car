import argparse
import warnings
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env
from torch.distributions.categorical import Categorical

# Suppress warnings
warnings.filterwarnings("ignore")

# Register the environment (assuming the registration info is the same)
gym.register("CarEnv-v0", entry_point="car_env:Car_env")


def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    """
    Initialize the weights and biases of a neural network layer. The weights are initialized
    using the orthogonal initialization method, and the biases are initialized to a constant value.

    Args:
        layer (nn.Module): Neural network layer to initialize.
        std (float, optional): Standard deviation for weight initialization. Defaults to np.sqrt(2).
        bias_const (float, optional): Constant value for bias initialization. Defaults to 0.0.

    Returns:
        nn.Module: Initialized neural network layer.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    An agent that uses separate actor and critic networks to interact with an environment.
    The actor network decides which action to take, and the critic network evaluates the
    action by estimating the value function of the state. The agent is used to train the
    actor and critic networks using the Proximal Policy Optimization (PPO) algorithm.

    Attributes:
        critic (nn.Module): A neural network that predicts the value of each state.
        actor (nn.Module): A neural network that outputs a probability distribution over actions.

    Args:
        envs (Env): A batched environment object which provides the observation space
                    and action space properties, to determine the input and output dimensions
                    of the neural networks.
        hidden_size (Tuple[int]): Tuple indicating the size of hidden layers in the form (size1, size2).
    """

    def __init__(self, envs: Env, hidden_size: Tuple[int]):
        super(Agent, self).__init__()
        obs_size = envs.single_observation_space.shape[0]
        act_size = envs.single_action_space.n

        # Critic Network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_size, hidden_size[0])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size[0], hidden_size[1])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size[1], 1)),
        )

        # Actor Network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_size, hidden_size[0])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size[0], hidden_size[1])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size[1], act_size)),
            nn.Softmax(dim=-1),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the value of the current state. This is used by the agent to evaluate the
        state and determine the advantage of taking an action.

        Args:
            x (torch.Tensor): The current state observation tensor.

        Returns:
            torch.Tensor: Estimated value of the state.
        """
        return self.critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Determines the action to take for the current state and computes the value. This is
        used by the agent to interact with the environment and train the actor and critic networks.

        Args:
            x (torch.Tensor): The current state observation tensor.
            action (Optional[torch.Tensor]): Optional tensor specifying the action to evaluate.
                                             If None, an action is sampled.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - sampled or provided action
                - log probability of the action
                - entropy of the action distribution
                - estimated value of the state
        """
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def load_model(model_path, envs, hidden_size):
    """
    Load the model from the specified path.
    """
    agent = Agent(envs, hidden_size)
    agent.load_state_dict(torch.load(model_path))
    agent.eval()
    return agent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a trained model and execute the game loop."
    )
    parser.add_argument(
        "--run-name", type=str, required=True, help="Name of the run for model path"
    )
    parser.add_argument(
        "--track-path",
        type=str,
        default="tracks/track.json",
        help="Path to the directory containing the model",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env = gym.make("CarEnv-v0", render_mode="human")
    hidden_size = (64, 64)  # Match the hidden size used during training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    envs = gym.vector.SyncVectorEnv([lambda: env])
    model_path = f"runs/{args.run_name}/model.pt"
    agent = load_model(model_path, envs, hidden_size).to(device)

    state = env.reset(options={"no_time_limit": True, "track_path": args.track_path})[0]
    total_reward = 0

    while True:
        state = torch.tensor([state], dtype=torch.float32).to(device)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(state)
        state, reward, done, _, _ = env.step(action.item())
        total_reward += reward
        env.render()

        if done:
            print("Game Over")
            print("Total Reward:", total_reward)
            break

    env.close()


if __name__ == "__main__":
    main()
