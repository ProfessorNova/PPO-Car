import argparse
import datetime
import glob
import os
import time
from distutils.util import strtobool
from typing import Callable, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import Env
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

gym.register("CarEnv-v0", entry_point="car_env:Car_env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="Name of the run",
    )
    parser.add_argument(
        "--num-envs", type=int, default=4, help="Number of environments"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1000000,
        help="Total number of timesteps to train the agent",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1000,
        help="Number of steps per environment per policy rollout",
    )
    parser.add_argument(
        "--track-path",
        type=str,
        default="tracks/track.json",
        help="Path to json files containing track information",
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Enable CUDA acceleration",
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Capture video of the training",
    )
    parser.add_argument(
        "--policy-learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for policy network",
    )
    parser.add_argument(
        "--value-function-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for value function network",
    )
    parser.add_argument(
        "--hidden-size",
        type=lambda x: tuple(map(int, x.split(","))),
        default=(64, 64),
        help="Tuple indicating the size of hidden layers in the form (size1, size2)",
    )
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Anneal learning rate",
    )
    parser.add_argument(
        "--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor for rewards"
    )
    parser.add_argument("--gae-lambda", type=float, default=0.97, help="Lambda for GAE")
    parser.add_argument(
        "--num-minibatches", type=int, default=4, help="Number of minibatches"
    )
    parser.add_argument(
        "--update-epochs", type=int, default=4, help="Number of epochs to update policy"
    )
    parser.add_argument(
        "--norm-adv",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Normalize advantages",
    )
    parser.add_argument(
        "--clip-coef", type=float, default=0.2, help="Clipping parameter for PPO"
    )
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Clip value loss",
    )
    parser.add_argument(
        "--ent-coef", type=float, default=0.0, help="Entropy coefficient"
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="Value function coefficient"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=0.5, help="Maximum gradient norm"
    )
    parser.add_argument(
        "--target-kl", type=float, default=0.01, help="Target KL divergence"
    )
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    return args


import torch.nn as nn


def make_env(
    idx: int, capture_video: bool, run_name: str, track_path: str
) -> Callable[[], gym.Env]:
    """
    Create a function that returns a gym environment.

    Args:
        idx (int): Index of the environment.
        capture_video (bool): Flag indicating whether to capture video of the training.
        run_name (str): Name of the run.
        track_path (str): Path to the track file.

    Returns:
        Callable[[], gym.Env]: Function that returns a gym environment.
    """

    def thunk() -> gym.Env:
        """
        Create and initialize the gym environment.

        Returns:
            gym.Env: Initialized gym environment.
        """

        env = gym.make("CarEnv-v0", render_mode="rgb_array")
        if capture_video and idx == 1:
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
            )
        _ = env.reset(options={"track_path": track_path})
        return env

    return thunk


def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    """
    Initialize the weights and biases of a neural network layer.

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
    action by estimating the value function of the state.

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
        obs_size = np.array(envs.single_observation_space.shape).prod()
        act_size = envs.single_action_space.n

        # Critic Network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_size, hidden_size[0])),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_size[0]),
            layer_init(nn.Linear(hidden_size[0], hidden_size[1])),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_size[1]),
            layer_init(
                nn.Linear(hidden_size[1], 1), std=1.0
            ),  # Output layer for critic
        )

        # Actor Network
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_size, hidden_size[0])),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_size[0]),
            layer_init(nn.Linear(hidden_size[0], hidden_size[1])),
            nn.Tanh(),
            nn.BatchNorm1d(hidden_size[1]),
            layer_init(
                nn.Linear(hidden_size[1], act_size), std=0.01
            ),  # Output layer for actor
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the value of the current state.

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
        Determines the action to take for the current state and computes the value.

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


if __name__ == "__main__":
    args = parse_args()
    run_name = (
        f"{args.run_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(i, args.capture_video, run_name, args.track_path)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = Agent(envs, args.hidden_size).to(device)
    optimizer_policy = optim.Adam(
        agent.actor.parameters(), lr=args.policy_learning_rate, eps=1e-5
    )
    optimizer_value = optim.Adam(
        agent.critic.parameters(), lr=args.value_function_learning_rate, eps=1e-5
    )

    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()

    next_obs = torch.Tensor(
        np.array(envs.reset()[0])
    ).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    try:
        for update in tqdm(
            range(1, num_updates + 1), desc="Training Progress", total=num_updates
        ):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates

                # Annealing the policy learning rate
                lr_policy_now = frac * args.policy_learning_rate
                for param_group in optimizer_policy.param_groups:
                    param_group["lr"] = lr_policy_now

                # Annealing the value function learning rate
                lr_value_now = frac * args.value_function_learning_rate
                for param_group in optimizer_value.param_groups:
                    param_group["lr"] = lr_value_now

            for step in tqdm(
                range(args.num_steps), desc="Environment Steps", leave=False
            ):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, _, info = envs.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                    done
                ).to(device)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                if args.gae:
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            nextvalues = values[t + 1]
                        delta = (
                            rewards[t]
                            + args.gamma * nextvalues * nextnonterminal
                            - values[t]
                        )
                        advantages[t] = lastgaelam = (
                            delta
                            + args.gamma
                            * args.gae_lambda
                            * nextnonterminal
                            * lastgaelam
                        )
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = (
                            rewards[t] + args.gamma * nextnonterminal * next_return
                        )
                    advantages = returns - values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Reset gradients for policy optimizer
                    optimizer_policy.zero_grad()
                    pg_loss.backward()
                    nn.utils.clip_grad_norm_(
                        agent.actor.parameters(), args.max_grad_norm
                    )
                    optimizer_policy.step()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    )

                    # Reset gradients for value function optimizer
                    optimizer_value.zero_grad()
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(
                        agent.critic.parameters(), args.max_grad_norm
                    )
                    optimizer_value.step()

                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            # After every update cycle, record the metrics
            writer.add_scalar("Loss/policy_loss", pg_loss.item(), global_step)
            # Policy Loss (pg_loss): Indicates how well the policy network is performing. Monitoring this helps in understanding whether the agent's policy is improving and whether the actions chosen by the policy are leading to higher advantages.
            writer.add_scalar("Loss/value_loss", v_loss.item(), global_step)
            # Value Loss (v_loss): Reflects the accuracy of the value network's predictions about future rewards. It's essential to ensure that the value predictions are close to the actual returns, which aids in stable learning.
            writer.add_scalar("Loss/entropy_loss", entropy_loss.item(), global_step)
            # Entropy is used as a regularization term to encourage exploration by maintaining a diverse range of actions. Logging this helps monitor how varied the action selection is and whether the agent is exploring sufficiently during training.
            writer.add_scalar("Loss/total_loss", loss.item(), global_step)
            # Represents the combined effect of the policy, value, and entropy losses. It's critical for evaluating the overall performance of the agent's updates.
            writer.add_scalar(
                "Info/policy_learning_rate",
                optimizer_policy.param_groups[0]["lr"],
                global_step,
            )
            writer.add_scalar(
                "Info/value_function_learning_rate",
                optimizer_value.param_groups[0]["lr"],
                global_step,
            )
            # Since the learning rate might be annealed (gradually reduced) during training, tracking it helps in understanding its impact on the loss metrics and overall training dynamics.
            writer.add_scalar("Info/clip_fraction", np.mean(clipfracs), global_step)
            # In Proximal Policy Optimization (PPO), the clipping parameter prevents large updates, which could destabilize learning. Logging the fraction of times the clipping is activated gives insights into how often the policy updates hit this boundary.
            writer.add_scalar("Info/approx_kl", approx_kl.item(), global_step)
            # This metric helps ensure that the updates to the policy are not too large, preserving learning stability. A sudden spike in KL divergence is an indicator that the policy changes too rapidly, which might lead to performance degradation.
            writer.add_scalar("Info/explained_variance", explained_var, global_step)
            # This shows how much of the variance in the rewards is explained by the value predictions. High explained variance is generally indicative of a well-performing value function.
            writer.add_scalar("Misc/entropy", entropy.mean().item(), global_step)
            # Again, entropy is logged under a miscellaneous category to track the randomness in action selection, reinforcing the agent's exploratory behavior.
            writer.add_scalar(
                "Environment/gates_passed", info["gates_passed"].mean(), global_step
            )
            # This metric helps in understanding the agent's progress in the environment. It quantifies how well the agent is performing by counting the number of gates passed during training.

    except KeyboardInterrupt:
        pass

    # Save the model
    model_save_path = f"runs/{run_name}/model.pt"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    try:
        torch.save(agent.state_dict(), model_save_path)
        print(f"Model saved successfully at {model_save_path}")
    except Exception as e:
        print(f"Failed to save the model due to: {e}")

    envs.close()
    writer.close()
