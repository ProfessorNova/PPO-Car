import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.signal
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import copy

from CarEnv import CarEnv

# Buffer for storing trajectories
class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95, shared_buffer=False, num_workers=1):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0

        # initialize shared buffer
        self.shared_buffer = shared_buffer
        if shared_buffer:
            self.num_workers = num_workers
            self.pointer_offset = size // num_workers
            self.pointer = []
            self.trajectory_start_index = []
            for i in range(num_workers):
                self.pointer.append(i * self.pointer_offset)
                self.trajectory_start_index.append(i * self.pointer_offset)

    def discounted_cumulative_sums(self, x, discount):
        # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    

    def store_single_process(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1

    def store_multi_process(self, observation, action, reward, value, logprobability, worker_id):
        # Append one step of agent-environment interaction in the right spot
        self.observation_buffer[self.pointer[worker_id]] = observation
        self.action_buffer[self.pointer[worker_id]] = action
        self.reward_buffer[self.pointer[worker_id]] = reward
        self.value_buffer[self.pointer[worker_id]] = value
        self.logprobability_buffer[self.pointer[worker_id]] = logprobability
        self.pointer[worker_id] += 1
    
    def store(self, observation, action, reward, value, logprobability, worker_id=0):
        if self.shared_buffer:
            self.store_multi_process(observation, action, reward, value, logprobability, worker_id)
        else:
            self.store_single_process(observation, action, reward, value, logprobability)


    def finish_trajectory_single_process(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = self.discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = self.discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index = self.pointer

    def finish_trajectory_multi_process(self, last_value=0, worker_id=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index[worker_id], self.pointer[worker_id])
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = self.discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = self.discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]

        self.trajectory_start_index[worker_id] = self.pointer[worker_id]

    def finish_trajectory(self, last_value=0, worker_id=0):
        if self.shared_buffer:
            self.finish_trajectory_multi_process(last_value, worker_id)
        else:
            self.finish_trajectory_single_process(last_value)


    def get_single_process(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )
    
    def get_multi_process(self):
        # Get all data of the buffer and normalize the advantages
        for i in range(self.num_workers):
            self.pointer[i] = i * self.pointer_offset
            self.trajectory_start_index[i] = i * self.pointer_offset
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )
    
    def get(self):
        if self.shared_buffer:
            return self.get_multi_process()
        else:
            return self.get_single_process()


class PPO_Agent:
    def __init__(self, env, render=False, render_epochs=[], multiprocessing=False, num_workers=1,
                steps_per_epoch=4000, epochs=200, gamma=0.99, clip_ratio=0.2,
                policy_learning_rate=3e-4, value_function_learning_rate=1e-3,
                train_policy_iterations=80, train_value_iterations=80,
                lam=0.97, target_kl=0.01, hidden_sizes=(64,64)):
        # Hyperparameters
        self.env = env
        self.render = render
        # setting up render_epochs
        if render and len(render_epochs) == 0:
            self.render_epochs = range(epochs)
        elif render and len(render_epochs) != 0:
            render_epochs = np.array(render_epochs)
            self.render_epochs = render_epochs - 1
        elif not render and len(render_epochs) != 0:
            print("Info: render_epochs is not empty but render is False. Will enable render.")
            render_epochs = np.array(render_epochs)
            self.render_epochs = render_epochs - 1
            self.render = True
        else:
            self.render_epochs = []
        # apply render settings
        if self.render and len(self.render_epochs) != 0:
            self.env.render_mode = "human"
        else:
            self.env.render_mode = None
        
        # Hyperparameters
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.policy_learning_rate = policy_learning_rate
        self.value_function_learning_rate = value_function_learning_rate
        self.train_policy_iterations = train_policy_iterations
        self.train_value_iterations = train_value_iterations
        self.lam = lam
        self.target_kl = target_kl
        self.hidden_sizes = hidden_sizes

        # Initialize the actor and the critic as keras models
        self.observation_dimensions = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        # setting up buffer
        self.buffer = Buffer(self.observation_dimensions, self.steps_per_epoch, self.gamma, self.lam)
        observation_input = keras.Input(shape=(self.observation_dimensions,), dtype=tf.float32)
        logits = self.mlp(observation_input, list(hidden_sizes) + [self.num_actions], tf.tanh, None)
        # actor model
        self.actor = keras.Model(inputs=observation_input, outputs=logits)
        value = tf.squeeze(
            self.mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1
        )
        # critic model
        self.critic = keras.Model(inputs=observation_input, outputs=value)

        # Initialize the policy and the value function optimizers
        self.policy_optimizer = keras.optimizers.Adam(learning_rate=self.policy_learning_rate)
        self.value_optimizer = keras.optimizers.Adam(learning_rate=self.value_function_learning_rate)

        # initializations for multiprocessing
        self.multiprocessing = multiprocessing
        if multiprocessing:
            self.num_workers = num_workers
            self.steps_per_worker = self.steps_per_epoch // self.num_workers

            # setting up buffer
            BaseManager.register('Buffer', Buffer)
            self.manager = BaseManager()
            self.manager.start()
            self.buffer = self.manager.Buffer(self.observation_dimensions, self.steps_per_epoch, self.gamma, self.lam, 
                                              shared_buffer=True, num_workers=self.num_workers)
            
            # Initialize the sum of the returns, lengths and number of episodes for each epoch
            self.sum_return = mp.Value("d", 0.0)
            self.sum_length = mp.Value("d", 0.0)
            self.num_episodes = mp.Value("i", 0)


    def mlp(self, x, sizes, activation=tf.tanh, output_activation=None):
        # Build a feedforward neural network
        for size in sizes[:-1]:
            x = layers.Dense(units=size, activation=activation)(x)
        return layers.Dense(units=sizes[-1], activation=output_activation)(x)
    
    def logprobabilities(self, logits, a):
        # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(
            tf.one_hot(a, self.num_actions) * logprobabilities_all, axis=1
        )
        return logprobability
    
    # Sample action from actor
    @tf.function
    def sample_action(self, observation):
        logits = self.actor(observation)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        return logits, action
    
    @tf.function
    def train_policy(
        self, observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
    ):

        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            ratio = tf.exp(
                self.logprobabilities(self.actor(observation_buffer), action_buffer)
                - logprobability_buffer
            )
            min_advantage = tf.where(
                advantage_buffer > 0,
                (1 + self.clip_ratio) * advantage_buffer,
                (1 - self.clip_ratio) * advantage_buffer,
            )

            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantage_buffer, min_advantage)
            )
        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        kl = tf.reduce_mean(
            logprobability_buffer
            - self.logprobabilities(self.actor(observation_buffer), action_buffer)
        )
        kl = tf.reduce_sum(kl)
        return kl
    
    # Train the value function by regression on mean-squared error
    @tf.function
    def train_value_function(self, observation_buffer, return_buffer):
        with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
            value_loss = tf.reduce_mean((return_buffer - self.critic(observation_buffer)) ** 2)
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))


    def train_single_process(self):
        for epoch in range(self.epochs):
            # Initialize the sum of the returns, lengths and number of episodes for each epoch
            sum_return = 0.0
            sum_length = 0.0
            num_episodes = 0

            if epoch in self.render_epochs:
                self.env.render_mode = "human"
            else:
                self.env.render_mode = None

            # Initialize the observation, episode return and episode length
            observation, episode_return, episode_length = self.env.reset(), 0, 0

            # Iterate over the steps of each epoch
            for t in range(self.steps_per_epoch):
                if self.render:
                    self.env.render()
                # Get the logits, action, and take one step in the environment
                observation = observation.reshape(1, -1)
                logits, action = self.sample_action(observation)
                observation_new, reward, done, _ = self.env.step(action[0].numpy())
                episode_return += reward
                episode_length += 1

                # Get the value and log-probability of the action
                value_t = self.critic(observation)
                logprobability_t = self.logprobabilities(logits, action)

                # Store obs, act, rew, v_t, logp_pi_t
                print(f"Collecting data: {t/self.steps_per_epoch*100:.0f}%", end="\r")
                self.buffer.store(observation, action, reward, value_t, logprobability_t)

                # Update the observation
                observation = observation_new

                # Finish trajectory if reached to a terminal state
                terminal = done
                if terminal or (t == self.steps_per_epoch - 1):
                    last_value = 0 if done else self.critic(observation.reshape(1, -1))
                    self.buffer.finish_trajectory(last_value)
                    sum_return += episode_return
                    sum_length += episode_length
                    num_episodes += 1
                    observation, episode_return, episode_length = self.env.reset(), 0, 0

            # Get values from the buffer
            (
                observation_buffer,
                action_buffer,
                advantage_buffer,
                return_buffer,
                logprobability_buffer,
            ) = self.buffer.get()

            # Update the policy and implement early stopping using KL divergence
            for _ in range(self.train_policy_iterations):
                kl = self.train_policy(
                    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
                )
                if kl > 1.5 * self.target_kl:
                    # Early Stopping
                    break

            # Update the value function
            for _ in range(self.train_value_iterations):
                self.train_value_function(observation_buffer, return_buffer)

            # Print mean return and length for each epoch
            print(
                f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
            )

    def worker(self, worker_id, render_worker=False):
        worker_env = copy.deepcopy(self.env)
        if render_worker:
            worker_env.render_mode = "human"
        else:
            worker_env.render_mode = None

        # Initialize the observation, episode return and episode length
        observation, episode_return, episode_length = worker_env.reset(), 0, 0

        for t in range(self.steps_per_worker):
            if render_worker:
                worker_env.render()
            # Get the logits, action, and take one step in the environment
            observation = observation.reshape(1, -1)
            logits, action = self.sample_action(observation)
            observation_new, reward, done, _ = worker_env.step(action[0].numpy())
            episode_return += reward
            episode_length += 1

            # Get the value and log-probability of the action
            value_t = self.critic(observation)
            logprobability_t = self.logprobabilities(logits, action)

            # Store obs, act, rew, v_t, logp_pi_t
            if worker_id == self.num_workers - 1: # only print for the last worker cause it will finish last
                print(f"Collecting data: {t/self.steps_per_worker*100:.0f}%", end="\r")
            self.buffer.store(observation, action, reward, value_t, logprobability_t, worker_id)

            # Update the observation
            observation = observation_new

            # Finish trajectory if reached to a terminal state
            terminal = done
            if terminal or (t == self.steps_per_worker - 1):
                last_value = 0 if done else self.critic(observation.reshape(1, -1))
                self.buffer.finish_trajectory(last_value, worker_id)
                self.sum_return.value += episode_return
                self.sum_length.value += episode_length
                self.num_episodes.value += 1
                observation, episode_return, episode_length = worker_env.reset(), 0, 0

    def train_multi_process(self):
        for epoch in range(self.epochs):
            self.sum_return.value = 0.0
            self.sum_length.value = 0.0
            self.num_episodes.value = 0

            # Create the workers
            workers = []
            render_worker = False
            for worker_id in range(self.num_workers):
                if worker_id == self.num_workers-1 and epoch in self.render_epochs:
                    render_worker = True
                else:
                    render_worker = False
                workers.append(
                    mp.Process(
                        target=self.worker,
                        args=(worker_id, render_worker),
                    )
                )

            # Start the workers
            for worker in workers:
                worker.start()

            # Wait for the workers to finish
            for worker in workers:
                worker.join()

            # Get values from the buffer
            (
                observation_buffer,
                action_buffer,
                advantage_buffer,
                return_buffer,
                logprobability_buffer,
            ) = self.buffer.get()

            # Update the policy and implement early stopping using KL divergence
            for _ in range(self.train_policy_iterations):
                kl = self.train_policy(
                    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer
                )
                if kl > 1.5 * self.target_kl:
                    # Early Stopping
                    break

            # Update the value function
            for _ in range(self.train_value_iterations):
                self.train_value_function(observation_buffer, return_buffer)

            # Print mean return and length for each epoch
            print(
                f"\n\n Epoch: {epoch + 1}. Mean Return: {self.sum_return.value / self.num_episodes.value}. Mean Length: {self.sum_length.value / self.num_episodes.value} \n"
            )
    
    def train(self):
        if self.multiprocessing:
            self.train_multi_process()
        else:
            self.train_single_process()

    
    # Save actor and critic models
    def save(self, path):
        self.actor.save(os.path.join(path, "actor"))
        self.critic.save(os.path.join(path, "critic"))

    # Load actor and critic models
    def load(self, path):
        project_path = os.path.abspath(os.path.dirname(__file__))
        self.actor = tf.keras.models.load_model(os.path.join(project_path, path, "actor"), compile=False)
        self.critic = tf.keras.models.load_model(os.path.join(project_path, path, "critic"), compile=False)

    # With the trained actor just running the environment
    def run(self):
        observation = self.env.reset(options='no_time_limit')
        self.env.render_mode = "human"
        done = False
        while not done:
            self.env.render()
            logits, action = self.sample_action(observation.reshape(1, -1))
            observation, reward, done, _ = self.env.step(action[0].numpy())

    def __getstate__(self):
        state = {}
        state["env"] = self.env
        state["buffer"] = self.buffer
        state["render"] = self.render
        state["render_epochs"] = self.render_epochs
        state["steps_per_worker"] = self.steps_per_worker
        state["num_workers"] = self.num_workers
        state["num_actions"] = self.num_actions
        state["sum_return"] = self.sum_return
        state["sum_length"] = self.sum_length
        state["num_episodes"] = self.num_episodes
        self.save("tmp")
        return state

    def __setstate__(self, state):
        self.env = state["env"]
        self.buffer = state["buffer"]
        self.render = state["render"]
        self.render_epochs = state["render_epochs"]
        self.steps_per_worker = state["steps_per_worker"]
        self.num_workers = state["num_workers"]
        self.num_actions = state["num_actions"]
        self.sum_return = state["sum_return"]
        self.sum_length = state["sum_length"]
        self.num_episodes = state["num_episodes"]
        self.load("tmp")

def main():
    env = CarEnv()
    agent = PPO_Agent(env, render=False, multiprocessing=True, num_workers=4)
    # Used for training
    # agent.train()
    # agent.save("models")

    agent.load("models")
    agent.run()


if __name__ == "__main__":
    main()