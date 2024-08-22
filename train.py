import argparse
import datetime
import os.path
import time
import tkinter as tk
import tkinter.filedialog as fd

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

from lib.buffer import Buffer
from lib.car_env import register_env
from lib.model import Agent

register_env()


def log_video(env, agent, device, video_path, fps=30):
    """
    Log a video of one episode of the agent interacting with the environment.
    :param env: a test environment which supports video recording
    :param agent: the agent to record
    :param device: the device to run the agent on
    :param video_path: the path to save the video to
    :param fps: the frames per second of the video
    """

    frames = []
    obs, _ = env.reset()
    done = False
    while not done:
        # Render the frame
        frames.append(env.render())
        # Get the action from the agent
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(
                torch.tensor(np.array([obs], dtype=np.float32), device=device))
        # Take a step in the environment
        obs, _, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        done = terminated
    # Save the video
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()


def make_env(env_id, reward_scaling=1.0, render=False, fps=30):
    """
    Make an environment with the given ID.
    :param env_id: the ID of the environment
    :param reward_scaling: the scaling factor for the rewards
    :param render: whether to render the environment
    :param fps: the frames per second of the rendering
    :return: the environment
    """
    if render:
        env = gym.make(env_id, render_mode="rgb_array")
        env.metadata["render_fps"] = fps
        env = gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)
    else:
        env = gym.make(env_id)
        env = gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)
    return env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True, help="Name of the run")
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable CUDA")
    parser.add_argument("--env", default="CarEnv-v0", help="Environment to use")
    parser.add_argument("--n-envs", type=int, default=16, help="Number of environments")
    parser.add_argument("--n-epochs", type=int, default=200, help="Number of epochs to run")
    parser.add_argument("--n-steps", type=int, default=1024, help="Number of steps per epoch per environment")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--train-iters", type=int, default=40, help="Number of training iterations")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="Lambda for GAE")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--ent-coef", type=float, default=0.001, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--learning-rate-decay", type=float, default=0.99, help="Multiply with lr every epoch")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--reward-scaling", type=float, default=0.1,
                        help="Scaling factor for the rewards for stable value function training")
    return parser.parse_args()


def select_file():
    """
    Opens a file dialog to select a JSON file from the tracks' folder.

    Returns:
        str: The path to the selected file, or None if no file was selected.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    # Set the directory to the tracks folder and filter to show only JSON files
    file_path = fd.askopenfilename(
        initialdir="tracks",
        title="Select track data file",
        filetypes=[("JSON Files", "*.json")],
    )
    root.destroy()  # Close the tkinter instance
    return file_path


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Select the track if the environment is the CarEnv
    track_path = select_file() if args.env == "CarEnv-v0" else None

    # Create the folders for logging
    current_dir = os.path.dirname(__file__)
    folder_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.run_name}"
    videos_dir = os.path.join(current_dir, "videos", folder_name)
    os.makedirs(videos_dir, exist_ok=True)
    checkpoint_dir = os.path.join(current_dir, "checkpoints", folder_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create the tensorboard writer
    log_dir = os.path.join(current_dir, "logs", folder_name)
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Create the environment
    envs = gym.vector.AsyncVectorEnv(
        [lambda: make_env(args.env, reward_scaling=args.reward_scaling) for _ in range(args.n_envs)])
    test_env = make_env(args.env, reward_scaling=args.reward_scaling, render=True)
    obs_dim = envs.single_observation_space.shape
    act_dim = envs.single_action_space.n

    # Create the agent and optimizer
    agent = Agent(obs_dim[0], act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.learning_rate_decay)
    print(agent.actor)
    print(agent.critic)

    # Create the buffer
    buffer = Buffer(obs_dim, args.n_steps, args.n_envs, device, args.gamma, args.gae_lambda)

    # Start the training
    global_step_idx = 0
    start_time = time.time()
    # Reset the environments with the selected track
    if track_path is not None:
        next_obs = torch.tensor(np.array(envs.reset(options={"track_path": track_path})[0],
                                         dtype=np.float32), device=device)
        # Also reset the test environment
        test_env.reset(options={"track_path": track_path})
    else:
        next_obs = torch.tensor(np.array(envs.reset()[0], dtype=np.float32), device=device)
    next_terminateds = torch.tensor([float(False)] * args.n_envs, device=device)
    next_truncateds = torch.tensor([float(False)] * args.n_envs, device=device)

    reward_list = []

    try:
        for epoch in range(1, args.n_epochs + 1):
            # Collect trajectories
            for _ in range(0, args.n_steps):
                global_step_idx += args.n_envs
                obs = next_obs
                terminateds = next_terminateds
                truncateds = next_truncateds

                # Get the action from the agent
                with torch.no_grad():
                    actions, logprobs, _, values = agent.get_action_and_value(obs)
                    values = values.view(-1)

                # Take a step in the environment
                next_obs, rewards, next_terminateds, next_truncateds, _ = envs.step(actions.cpu().numpy())

                # parse everything to tensors
                next_obs = torch.tensor(np.array(next_obs, dtype=np.float32), device=device)
                reward_list.extend(rewards)
                rewards = torch.tensor(rewards, device=device).view(-1)
                next_terminateds = torch.tensor([float(t) for t in next_terminateds], device=device)
                next_truncateds = torch.tensor([float(t) for t in next_truncateds], device=device)

                # Store the step in the buffer
                buffer.store(obs, actions, rewards, values, terminateds, truncateds, logprobs)

            # After the trajectories are collected, calculate the advantages and returns
            with torch.no_grad():
                # Calculate the value of the last state
                next_values = agent.get_value(next_obs).reshape(1, -1)
                next_terminateds = next_terminateds.reshape(1, -1)
                next_truncateds = next_truncateds.reshape(1, -1)
                traj_adv, traj_ret = buffer.calculate_advantages(next_values, next_terminateds, next_truncateds)

            # Get the trajectories from the buffer
            traj_obs, traj_act, traj_val, traj_logprob = buffer.get()

            # Flatten the trajectories
            traj_obs = traj_obs.view(-1, *obs_dim)
            traj_act = traj_act.view(-1)
            traj_logprob = traj_logprob.view(-1)
            traj_adv = traj_adv.view(-1)
            traj_ret = traj_ret.view(-1)
            traj_val = traj_val.view(-1)

            # Create an array of indices to sample from the trajectories
            traj_indices = np.arange(args.n_steps * args.n_envs)

            sum_loss_policy = 0.0
            sum_loss_value = 0.0
            sum_entropy = 0.0
            sum_loss_total = 0.0
            for _ in range(args.train_iters):
                # Shuffle the indices
                np.random.shuffle(traj_indices)

                # Iterate over the batches
                for start_idx in range(0, args.n_steps, args.batch_size):
                    end_idx = start_idx + args.batch_size
                    batch_indices = traj_indices[start_idx:end_idx]

                    # Get the log probabilities, entropies and values
                    _, new_logprobs, entropies, new_values = agent.get_action_and_value(traj_obs[batch_indices],
                                                                                        traj_act[batch_indices])
                    ratios = torch.exp(new_logprobs - traj_logprob[batch_indices])

                    # normalize the advantages
                    batch_adv = traj_adv[batch_indices]
                    batch_adv = (batch_adv - batch_adv.mean()) / torch.max(batch_adv.std(),
                                                                           torch.tensor(1e-5, device=device))

                    # Calculate the policy loss
                    policy_loss1 = -batch_adv * ratios
                    policy_loss2 = -batch_adv * torch.clamp(ratios, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
                    policy_loss = torch.max(policy_loss1, policy_loss2).mean()

                    # Calculate the value loss
                    new_values = new_values.view(-1)
                    value_loss = 0.5 * ((new_values - traj_ret[batch_indices]) ** 2).mean()

                    # Calculate the entropy loss
                    entropy = entropies.mean()

                    # Calculate the total loss
                    loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

                    # Optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                    sum_loss_policy += policy_loss.item()
                    sum_loss_value += value_loss.item()
                    sum_entropy += entropy.item()
                    sum_loss_total += loss.item()

            # Update the learning rate
            scheduler.step()

            # Log info on console
            avg_reward = sum(reward_list) / len(reward_list)
            # Rescale the rewards back
            avg_reward /= args.reward_scaling
            print(f"Epoch {epoch} done in {time.time() - start_time:.2f}s. "
                  f"Avg reward: {avg_reward:.4f}. ")
            reward_list = []

            # Every n epochs log a video
            if epoch % 10 == 0:
                log_video(test_env, agent, device, os.path.join(videos_dir, f"epoch_{epoch}.mp4"))
                # Save the model
                torch.save(agent.state_dict(), os.path.join(checkpoint_dir, f"checkpoint_{epoch}.dat"))

            # Log everything to tensorboard
            writer.add_scalar("losses/policy_loss", sum_loss_policy / args.train_iters, global_step_idx)
            writer.add_scalar("losses/value_loss", sum_loss_value / args.train_iters, global_step_idx)
            writer.add_scalar("losses/entropy", sum_entropy / args.train_iters, global_step_idx)
            writer.add_scalar("losses/total_loss", sum_loss_total / args.train_iters, global_step_idx)
            writer.add_scalar("charts/avg_reward", avg_reward, global_step_idx)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step_idx)
            writer.add_scalar("charts/SPS", global_step_idx / (time.time() - start_time), global_step_idx)

    finally:
        # Close the environments and tensorboard writer
        envs.close()
        test_env.close()
        writer.close()

        # Save the model
        torch.save(agent.state_dict(), os.path.join(checkpoint_dir, "model.dat"))
