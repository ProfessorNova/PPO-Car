import argparse
import warnings
from typing import Tuple

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


class Agent(nn.Module):
    """
    Define the agent class as per the training script for consistency.
    """

    def __init__(self, envs: Env, hidden_size: Tuple[int]):
        super(Agent, self).__init__()
        obs_size = np.array(envs.single_observation_space.shape).prod()
        act_size = envs.single_action_space.n

        # Define actor and critic networks
        self.critic = nn.Sequential(
            nn.Linear(obs_size, hidden_size[0]),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_size[0]),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_size[1]),
            nn.Linear(hidden_size[1], 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(obs_size, hidden_size[0]),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_size[0]),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_size[1]),
            nn.Linear(hidden_size[1], act_size),
        )

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample() if action is None else action
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
