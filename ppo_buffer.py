import argparse
import datetime
import os
import pickle
from distutils.util import strtobool
from multiprocessing import Pipe, Process
from typing import Any, Callable, List, Optional, Tuple

import cloudpickle
import gymnasium as gym
import numpy as np
import scipy.signal
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
        "--num-envs", type=int, default=8, help="Number of environments"
    )
    parser.add_argument(
        "--steps-per-epoch", type=int, default=8000, help="Number of steps per epoch"
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of epochs to train"
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
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Anneal learning rate",
    )
    parser.add_argument(
        "--policy-learning-rate",
        type=float,
        default=0.0002,
        help="Learning rate for policy network",
    )
    parser.add_argument(
        "--value-function-learning-rate",
        type=float,
        default=0.0005,
        help="Learning rate for value function network",
    )
    parser.add_argument(
        "--train-iterations",
        type=int,
        default=1000,
        help="Number of iterations to train the policy network",
    )
    parser.add_argument(
        "--hidden-size",
        type=lambda x: tuple(map(int, x.split(","))),
        default=(128, 128),
        help="Tuple indicating the size of hidden layers in the form (size1, size2)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.95, help="Discount factor for rewards"
    )
    parser.add_argument("--gae-lambda", type=float, default=0.92, help="Lambda for GAE")
    parser.add_argument(
        "--clip-coef", type=float, default=0.1, help="Clipping parameter for PPO"
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
        "--ent-coef", type=float, default=0.003, help="Entropy coefficient"
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="Value function coefficient"
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="Maximum gradient norm"
    )
    parser.add_argument(
        "--target-kl", type=float, default=0.01, help="Target KL divergence"
    )
    args = parser.parse_args()
    return args


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
        Create and initialize the gym environment. If the capture_video flag is set to True,
        the environment will record a video of the training.

        Returns:
            gym.Env: Initialized gym environment.
        """

        env = gym.make("CarEnv-v0", render_mode="rgb_array")
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                disable_logger=True,
            )
        _ = env.reset(options={"track_path": track_path})
        return env

    return thunk


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
        obs_size = np.array(envs.single_observation_space.shape).prod()
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


def compute_gae(next_value, rewards, dones, values, gamma, gae_lambda):
    """
    Calculate Generalized Advantage Estimation (GAE) for vectorized environments. This function
    computes the returns and advantages for each timestep and each environment using the rewards,
    values, and dones tensors.

    Args:
        next_value (torch.Tensor): The value estimate of the next state for each environment.
        rewards (torch.Tensor): Rewards received after each action for each environment.
        dones (torch.Tensor): Boolean tensor indicating which episodes have finished for each environment.
        values (torch.Tensor): Value estimates for each state in the buffer for each environment.
        gamma (float): Discount factor.
        gae_lambda (float): GAE lambda parameter for weighting advantages.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
            - returns (torch.Tensor): Computed returns for each timestep and each environment.
            - advs (torch.Tensor): Computed advantages for each timestep and each environment.
    """
    num_steps, num_envs = rewards.size()
    advs = torch.zeros((num_steps, num_envs), device=rewards.device)
    mask = ~dones
    mask = mask.float()
    lastgaelam = torch.zeros(num_envs, device=rewards.device)

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_non_terminal = mask[t]
            next_values = next_value.squeeze()
        else:
            next_non_terminal = mask[t + 1]
            next_values = values[t + 1].squeeze()

        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * next_non_terminal * lastgaelam
        advs[t] = lastgaelam

    returns = advs + values
    return returns, advs


def ppo_update(
    agent,
    optimizer_policy,
    optimizer_value,
    obs,
    actions,
    logprobs_old,
    returns,
    advantages,
    clip_coef,
    ent_coef,
    vf_coef,
    max_grad_norm,
):
    """
    Perform a PPO update on policy and value networks. This function adjusts the policy
    and value networks by performing gradient descent on batches of data. The policy
    network is updated to maximize the expected return, while the value network is updated
    to minimize the mean squared error between the predicted value and the computed return.

    Args:
        agent (nn.Module): The agent containing the policy and value networks.
        optimizer_policy (torch.optim.Optimizer): Optimizer for the policy network.
        optimizer_value (torch.optim.Optimizer): Optimizer for the value network.
        obs (torch.Tensor): Observations, shape [num_steps, num_envs, obs_dim].
        actions (torch.Tensor): Actions taken, shape [num_steps, num_envs].
        log_probs_old (torch.Tensor): Log probabilities of the actions under the old policy, shape [num_steps, num_envs].
        returns (torch.Tensor): Computed returns, shape [num_steps, num_envs].
        advantages (torch.Tensor): Computed advantages, shape [num_steps, num_envs].
        clip_coef (float): Clipping coefficient for PPO.
        ent_coef (float): Entropy coefficient to encourage exploration.
        vf_coef (float): Coefficient for the value function loss.
        max_grad_norm (float): Maximum norm for gradient clipping.

    Returns:
        Tuple[float, float, float]: Policy loss, value loss, and entropy loss.
    """
    # Forward pass to get outputs from the agent
    _, log_probs, entropy, values = agent.get_action_and_value(obs, actions)

    # Calculate the ratio (pi_theta / pi_theta_old)
    ratios = torch.exp(log_probs - logprobs_old.detach())

    # Calculate surrogate losses
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_coef, 1.0 + clip_coef) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # Value loss using the Bellman equation
    value_loss = (returns - values.squeeze(-1)).pow(2).mean()

    # Total loss
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()

    # Perform backpropagation
    optimizer_policy.zero_grad()
    optimizer_value.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)

    # Optimization step
    optimizer_policy.step()
    optimizer_value.step()

    return policy_loss.item(), value_loss.item(), entropy.mean().item()


def record_metrics(
    writer,
    global_step,
    policy_loss,
    value_loss,
    entropy_loss,
    scheduler_policy,
    scheduler_value,
):
    """
    Record training metrics to TensorBoard. This function logs the policy loss, value loss,
    entropy loss, and learning rates to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard writer.
        global_step (int): Current step number in total across all epochs.
        policy_loss (float): Recent policy loss computed.
        value_loss (float): Recent value loss computed.
        entropy_loss (float): Recent entropy loss computed.
        args (argparse.Namespace): Parsed command-line arguments.
    """
    writer.add_scalar("Loss/policy_loss", policy_loss, global_step)
    writer.add_scalar("Loss/value_loss", value_loss, global_step)
    writer.add_scalar("Loss/entropy_loss", entropy_loss, global_step)
    writer.add_scalar(
        "Parameters/policy_learning_rate",
        scheduler_policy.get_last_lr()[0],
        global_step,
    )
    writer.add_scalar(
        "Parameters/value_function_learning_rate",
        scheduler_value.get_last_lr()[0],
        global_step,
    )


class RolloutBuffer:
    """
    A buffer to store the trajectories collected by the agent. This buffer stores the
    observations, actions, rewards, advantages, returns, values, and log probabilities
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: Tuple[int],
        act_dim: int,
        gae_lambda: float,
        gamma: float,
    ):
        """
        Initialize the RolloutBuffer.

        Args:
            buffer_size (int): The maximum size of the buffer.
            obs_dim (Tuple[int]): The shape of the observation space.
            act_dim (int): The dimension of the action space.
            gae_lambda (float): The lambda parameter for Generalized Advantage Estimation.
            gamma (float): The discount factor.
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.obs_buf = np.zeros((buffer_size, *obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(buffer_size, dtype=np.float32)
        self.adv_buf = np.zeros(buffer_size, dtype=np.float32)
        self.rew_buf = np.zeros(buffer_size, dtype=np.float32)
        self.ret_buf = np.zeros(buffer_size, dtype=np.float32)
        self.val_buf = np.zeros(buffer_size, dtype=np.float32)
        self.logp_buf = np.zeros(buffer_size, dtype=np.float32)

        self.ptr: int = 0
        self.path_start_idx: int = 0
        self.max_size: int = buffer_size

    def store(self, obs: np.ndarray, act: int, rew: float, val: float, logp: float):
        """
        Store a single timestep in the buffer.

        Args:
            obs (np.ndarray): The observation at the current timestep.
            act (int): The action taken at the current timestep.
            rew (float): The reward received at the current timestep.
            val (float): The value estimate at the current timestep.
            logp (float): The log probability of the action taken.
        """
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val: float = 0) -> None:
        """
        Finish the current trajectory path in the buffer. This function computes the
        advantages and returns for the trajectory using the rewards, values, and the last
        value estimate.

        Args:
            last_val (float, optional): The value estimate at the last timestep. Defaults to 0.
        """
        path_slice: slice = slice(self.path_start_idx, self.ptr)
        rews: np.ndarray = np.append(self.rew_buf[path_slice], last_val)
        vals: np.ndarray = np.append(self.val_buf[path_slice], last_val)

        deltas: np.ndarray = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cumsum(
            deltas, self.gamma * self.gae_lambda
        )

        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self) -> dict[str, torch.Tensor]:
        """
        Get the data from the buffer. This function returns a dictionary containing the
        observations, actions, returns, advantages, and log probabilities stored in the buffer.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the data from the buffer.
        """
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean: float = np.mean(self.adv_buf)
        adv_std: float = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data: dict[str, np.ndarray] = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def discount_cumsum(self, x: np.ndarray, discount: float) -> np.ndarray:
        """
        Compute the discounted cumulative sum of an array. This function computes the sum
        of the array with discounting applied. The discount factor is applied to the sum
        of the array in reverse order.

        Args:
            x (np.ndarray): The input array.
            discount (float): The discount factor.

        Returns:
            np.ndarray: The discounted cumulative sum of the input array.
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class CloudpickleWrapper:
    def __init__(self, x: Any):
        """
        Initialize the CloudpickleWrapper.

        Args:
            x (Any): The object to be wrapped.
        """
        self.x = x

    def __getstate__(self) -> bytes:
        """
        Get the pickled state of the wrapped object.

        Returns:
            bytes: The pickled state of the wrapped object.
        """

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob: bytes) -> None:
        """
        Set the state of the wrapped object from the pickled state.

        Args:
            ob (bytes): The pickled state of the wrapped object.
        """

        self.x = pickle.loads(ob)


def worker(remote: Any, parent_remote: Any, env_fn_wrapper: Callable[[], Any]) -> None:
    """
    Worker function for the multiprocessing environment. This function receives commands
    from the parent process and performs the corresponding actions in the environment.

    Args:
        remote (Any): Connection to the parent process.
        parent_remote (Any): Connection to the worker process.
        env_fn_wrapper (Callable[[], Any]): Wrapper function to create the environment.
    """
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            obs, reward, done, _, info = env.step(data)
            if done:
                obs, info = env.reset()
            remote.send((obs, reward, done, _, info))
        elif cmd == "reset":
            obs, info = env.reset()
            remote.send((obs, info))
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError


class VecEnv:
    """
    Vectorized environment class to run multiple environments in parallel. This class
    creates a set of environments and runs them in parallel using multiprocessing.

    Args:
        env_fns (List[Callable[[], Env]]): List of wrapper functions to create environments.
    """

    def __init__(self, env_fns: List[Callable[[], Env]]):
        """
        Initializes the PPOBuffer class.

        Args:
            env_fns (List[Callable[[], Env]]): A list of functions that create the environment instances.

        Attributes:
            waiting (bool): Indicates whether the buffer is waiting for a response from the environment.
            closed (bool): Indicates whether the buffer is closed.
            remotes (List[Connection]): A list of connection objects for communication with the worker processes.
            work_remotes (List[Connection]): A list of connection objects for communication with the main process.
            ps (List[Process]): A list of worker processes.
            single_observation_space (np.ndarray): The observation space of a single environment instance.
            single_action_space: The action space of a single environment instance.
        """
        self.waiting: bool = False
        self.closed: bool = False
        nenvs: int = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps: List[Process] = [
            Process(
                target=worker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn)),
            )
            for (work_remote, remote, env_fn) in zip(
                self.work_remotes, self.remotes, env_fns
            )
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(("reset", None))
        self.single_observation_space: np.ndarray = np.array(
            self.remotes[0].recv()[0].shape
        )
        self.single_action_space = env_fns[0]().action_space

    def step_async(self, actions: np.ndarray) -> None:
        """
        Asynchronously sends actions to the environments.
        Args:
            actions (np.ndarray): An array of actions to be sent to each environment.
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any]]:
        """
        Waits for the environments to complete their steps and returns the results.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any]]: A tuple containing the observations,
            rewards, dones, and info lists for each environment.
        """
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, _, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any]]:
        """
        Sends actions to the environments and waits for the results.
        Args:
            actions (np.ndarray): An array of actions to be sent to each environment.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, List[Any]]: A tuple containing the observations,
            rewards, dones, and info lists for each environment.
        """
        self.step_async(actions)
        return self.step_wait()

    def reset(self) -> np.ndarray:
        """
        Resets the environments and returns the initial observations.
        Returns:
            np.ndarray: An array of initial observations for each environment.
        """
        for remote in self.remotes:
            remote.send(("reset", None))
        return np.stack([remote.recv()[0] for remote in self.remotes])

    def close(self) -> None:
        """
        Closes the environments and terminates the worker processes.
        """
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True


if __name__ == "__main__":
    args = parse_args()

    # Set up TensorBoard writer and save hyperparameters
    run_name = (
        f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.run_name}"
    )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(i, args.capture_video, run_name, args.track_path)
            for i in range(args.num_envs)
        ]
    )

    # Initialize the agent and optimizers
    agent = Agent(envs, args.hidden_size).to(device)
    optimizer_policy = optim.Adam(
        agent.actor.parameters(), lr=args.policy_learning_rate, eps=1e-5
    )
    optimizer_value = optim.Adam(
        agent.critic.parameters(), lr=args.value_function_learning_rate, eps=1e-5
    )
    scheduler_policy = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_policy, mode="min", factor=0.9, patience=10
    )
    scheduler_value = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_value, mode="min", factor=0.9, patience=10
    )

    # Initialize the rollout buffer
    buffer = RolloutBuffer(
        args.steps_per_epoch,
        envs.single_observation_space.shape,
        (envs.single_action_space.n,),
        args.gae_lambda,
        args.gamma,
    )

    global_step = 0

    try:
        for epoch in tqdm(range(args.epochs), desc="Epochs", leave=False):
            # Environment reset at the beginning of each epoch
            next_obs = torch.tensor(envs.reset()[0], device=device, dtype=torch.float32)
            for step in tqdm(range(args.steps_per_epoch), desc="Steps", leave=False):
                obs = next_obs
                # No gradient calculations needed during data collection
                with torch.no_grad():
                    action, log_prob, _, value = agent.get_action_and_value(obs)
                next_obs, rewards, dones, _, _ = envs.step(action.cpu().numpy())
                next_obs = torch.tensor(next_obs, device=device, dtype=torch.float32)
                rewards = torch.tensor(rewards, device=device, dtype=torch.float32)
                dones = torch.tensor(dones, device=device, dtype=torch.bool)
                buffer.store(
                    obs.cpu().numpy(),
                    action.cpu().numpy(),
                    rewards.cpu().numpy(),
                    value.cpu().numpy(),
                    log_prob.cpu().numpy(),
                )

                global_step += 1

                for idx, done in enumerate(dones):
                    if done:
                        buffer.finish_path(last_val=0)

            # Calculate the last next_value for GAE computation
            with torch.no_grad():
                _, _, _, next_value = agent.get_action_and_value(next_obs)

            buffer.finish_path(last_val=next_value.cpu().numpy())
            data = buffer.get()

            # Train the agent using the collected data
            for _ in range(args.train_iterations):
                pg_loss, v_loss, entropy_loss = ppo_update(
                    agent,
                    optimizer_policy,
                    optimizer_value,
                    data["obs"],
                    data["act"],
                    data["logp"],
                    data["ret"],
                    data["adv"],
                    args.clip_coef,
                    args.ent_coef,
                    args.vf_coef,
                    args.max_grad_norm,
                )
                _, new_logprobs, _, _ = agent.get_action_and_value(
                    data["obs"], data["act"]
                )
                kl_divergence = torch.exp(data["logp"]) * (
                    data["logp"] - new_logprobs.detach()
                )
                kl_divergence = kl_divergence.mean()
                if kl_divergence > 1.5 * args.target_kl:
                    break

            if args.anneal_lr:
                scheduler_policy.step(pg_loss)
                scheduler_value.step(v_loss)

            # Log the metrics to TensorBoard
            record_metrics(
                writer,
                global_step,
                pg_loss,
                v_loss,
                entropy_loss,
                scheduler_policy,
                scheduler_value,
            )

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

    # Close the environments and TensorBoard writer
    finally:
        envs.close()
        writer.close()
