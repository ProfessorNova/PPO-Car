# car-driving-agent
## Custome Gym Environment with Keras PPO and Multiprocessing

Result after training for about an hour with 4 processes (200 epochs and 4000 steps per epoch):

![Demo Gif](https://github.com/ProfessorNova/car-driving-agent/blob/main/gifs/demo.gif)

## Requirements
- Environment: You can just use the environment I created, [CarEnv](https://github.com/ProfessorNova/car-driving-agent/blob/main/example_custome_environment/CarEnv.py). But the PPO Agent can be run in any environment. You just have to set up the environment as described in this [Gymnasium Documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/).
- TensorFlow: I used TensorFlow 2.10 because starting with TensorFlow 2.11 GPU isn't supported in native Windows. You can just follow the [TensorFlow Installation Guide](https://www.tensorflow.org/install/pip).
- Libraries: numpy, scipy, gymnasium, and pygame. These can all be installed using pip.

## How to run my pretrained model
You just need to copy all files contained in this [example](https://github.com/ProfessorNova/car-driving-agent/tree/main/example_custome_environment) into one folder, and run [PPO_Agent.py](https://github.com/ProfessorNova/car-driving-agent/blob/main/example_custome_environment/PPO_Agent.py).

## How to train
In the [PPO_Agent.py](https://github.com/ProfessorNova/car-driving-agent/blob/main/example_custome_environment/PPO_Agent.py) from the [example](https://github.com/ProfessorNova/car-driving-agent/tree/main/example_custome_environment) you need to change the main to:
```python
def main():
    env = CarEnv()
    agent = PPO_Agent(env, render=False, multiprocessing=True, num_workers=4)
    agent.train() # training process
    agent.save("models") # saving the trained models
```
Running many workers can quickly fill up the RAM, so you may need to decrease the number of workers.

### Additional options:
You can set ```render_epochs``` to monitor the learning process. The following ```main()``` will render the epochs 1, 50, 100, 150, and 200.
```python
def main():
    env = CarEnv()
    agent = PPO_Agent(env, render=True, render_epochs=[1, 50, 100, 150, 200], multiprocessing=True, num_workers=4)
    agent.train()
```
If you just set ```render=True``` without naming any ```render_epochs```, it will render all the epochs. Keep in mind that this will slow down the training process a lot.

You can also play around with all the hyperparameters. The defaults are set as follows:
```python
steps_per_epoch=4000,
epochs=200,
gamma=0.99,
clip_ratio=0.2,
policy_learning_rate=3e-4,
value_function_learning_rate=1e-3,
train_policy_iterations=80,
train_value_iterations=80,
lam=0.97,
target_kl=0.01,
hidden_sizes=(64,64)
```

## Performance

### My Hardware:
- CPU: Ryzen 9 5900X
- GPU: RTX 3080 12GB
- RAM: 64GB DDR4

### First test:

Parameters:
- steps per epoch: 4000
- epochs: 10

| Mode                      | Avg. time per epoch |
|---------------------------|---------------------|
| single process            | 17.93s              |
| multi process (2 workers) | 15.42s              |
| multi process (3 workers) | 14.84s              |
| multi process (4 workers) | 15.62s              |

### Second test:

Parameters:
- steps per epoch: 8000
- epochs: 10

| Mode                      | Avg. time per epoch |
|---------------------------|---------------------|
| single process            | 35.39s              |
| multi process (2 workers) | 24.77s              |
| multi process (3 workers) | 21.35s              |
| multi process (4 workers) | 20.96s              |

As you can see, more workers can also slow down the training process. The problem with the current version of the code is that it needs to restart the subprocesses after each epoch. This will be fixed in future versions of the code.

## Useful resources
- The code was mainly inspired by [this](https://keras.io/examples/rl/ppo_cartpole/).
