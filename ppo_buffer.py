import argparse
import datetime
import os
from distutils.util import strtobool
from typing import Callable, Optional, Tuple

import gymnasium as gym
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import Env
from torch.distributions import Categorical
from torch.multiprocessing import Pipe, Process
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import the car environment
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
        default=100,
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
        "--target-kl", type=float, default=0.02, help="Target KL divergence"
    )
    args = parser.parse_args()
    return args


def make_env(
    idx: int, capture_video: bool, run_name: str, track_path: str
) -> Callable[[], Env]:
    def thunk() -> Env:
        env = gym.make("CarEnv-v0", render_mode="rgb_array")
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        _ = env.reset(options={"track_path": track_path})
        return env

    return thunk


def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs: Env, hidden_size: Tuple[int]):
        super(Agent, self).__init__()
        obs_size = np.prod(envs.single_observation_space)
        act_size = envs.single_action_space.n

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_size, hidden_size[0])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size[0], hidden_size[1])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size[1], 1)),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_size, hidden_size[0])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size[0], hidden_size[1])),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size[1], act_size)),
            nn.Softmax(dim=-1),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(x)

    def get_action_and_value(
        self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def compute_gae(next_value, rewards, dones, values, gamma, gae_lambda):
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
    logits, log_probs, entropy, values = agent.get_action_and_value(obs, actions)
    ratios = torch.exp(log_probs - logprobs_old.detach())
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_coef, 1.0 + clip_coef) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = (returns - values.squeeze(-1)).pow(2).mean()
    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()

    optimizer_policy.zero_grad()
    optimizer_value.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
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


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            obs, reward, done, truncated, info = env.step(data)
            if done or truncated:
                obs, info = env.reset()
            remote.send((obs, reward, done, truncated, info))
        elif cmd == "reset":
            obs, info = env.reset()
            remote.send((obs, info))
        elif cmd == "close":
            remote.close()
            break
        else:
            raise NotImplementedError


class VecEnv:
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(
                target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn))
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
        self.single_observation_space = np.array(self.remotes[0].recv()[0].shape)
        self.single_action_space = env_fns[0]().action_space

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, truncs, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), np.stack(truncs), infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        return np.stack(obs), infos

    def close(self):
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


class RolloutBuffer:
    def __init__(self, buffer_size, obs_dim, act_dim, gae_lambda, gamma):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.obs_buf = np.zeros((buffer_size, *obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(buffer_size, dtype=np.int32)
        self.adv_buf = np.zeros(buffer_size, dtype=np.float32)
        self.rew_buf = np.zeros(buffer_size, dtype=np.float32)
        self.ret_buf = np.zeros(buffer_size, dtype=np.float32)
        self.val_buf = np.zeros(buffer_size, dtype=np.float32)
        self.logp_buf = np.zeros(buffer_size, dtype=np.float32)

        self.ptr, self.path_start_idx, self.max_size = 0, 0, buffer_size

    def store(self, obs, act, rew, val, logp):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cumsum(
            deltas, self.gamma * self.gae_lambda
        )

        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            logp=self.logp_buf,
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    def discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class CloudpickleWrapper:
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


def main():
    args = parse_args()

    run_name = (
        f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.run_name}"
    )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_fns = [
        make_env(i, args.capture_video, run_name, args.track_path)
        for i in range(args.num_envs)
    ]
    envs = VecEnv(env_fns)

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
            next_obs = torch.tensor(envs.reset()[0], device=device, dtype=torch.float32)
            for step in tqdm(range(args.steps_per_epoch), desc="Steps", leave=False):
                obs = next_obs
                with torch.no_grad():
                    action, log_prob, _, value = agent.get_action_and_value(obs)
                next_obs, rewards, dones, truncs, infos = envs.step(
                    action.cpu().numpy()
                )
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

            with torch.no_grad():
                _, _, _, next_value = agent.get_action_and_value(next_obs)

            buffer.finish_path(last_val=next_value.cpu().numpy())
            data = buffer.get()

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

    model_save_path = f"runs/{run_name}/model.pt"
    os.makedirs(os.path.dirname(model_save_path), exist=True)
    try:
        torch.save(agent.state_dict(), model_save_path)
        print(f"Model saved successfully at {model_save_path}")
    except Exception as e:
        print(f"Failed to save the model due to: {e}")

    finally:
        envs.close()
        writer.close()


if __name__ == "__main__":
    main()
