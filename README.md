# PPO-Car

---

## Results

![Demo Gif](https://github.com/ProfessorNova/PPO-Car/blob/main/docs/demo.gif)

---

## Installation

1. Clone the repository
2. Go to the repository directory
3. Have python installed (tested with python 3.10.11)
4. Run `pip install -r requirements.txt` (for proper pytorch installation, visit
   the [pytorch website](https://pytorch.org/get-started/locally/))
5. Run `python train.py --run-name "my_run"` (if you want to use cuda, add `--cuda` to the command)
6. A window will open where you can select the track you want to train on. (You can also create your own track with the
   `track_editor.py` script. More on that below)
7. On default, a video will be recorded every 10 epochs. You can find the videos in the `videos` directory.
8. Additionally, you can view the training progress with tensorboard by running `tensorboard --logdir "logs"` and
   opening the link in your browser.

---

## Environment

### Description

The environment is a simple 2D car simulation. The track data is loaded from a json file. The car has to drive around
the track and pass through the reward gates and avoid the walls. The Car has velocity and can snap turn.

### Action Space

The action space is a `Discrete(9)` space. The actions are as follows:

- 0: forward
- 1: backward
- 2: left
- 3: right
- 4: forward left
- 5: forward right
- 6: backward left
- 7: backward right
- 8: do nothing (thus removing velocity)

### Observation Space

The observation space is a `Box(6 + num_rays, )` space. The observations are as follows:

- 0: normalized x position of the car ranging from 0 to 1
- 1: normalized y position of the car ranging from 0 to 1
- 2: normalized x velocity of the car ranging from -1 to 1
- 3: normalized y velocity of the car ranging from -1 to 1
- 4: cos of the car angle ranging from -1 to 1
- 5: sin of the car angle ranging from -1 to 1
- 6 to 6 + num_rays: distance to the nearest wall for each ray

### Reward

The Agent receives a reward of 0.01 for choosing actions with a forward action.
The Agent receives a reward of 1.0 for passing through a reward gate.
The Agent receives a penalty of -3.0 for hitting a wall.
The Agent receives a reward of 10.0 for finishing a lap.

### Starting State

The Agent starts at a predefined position and direction which can be set in the track json file.

### Episode End

The episode terminates if the Agent hits a wall and truncates if the time steps pass 1000.

### Arguments

The path to the track json file can be set as an option in the reset function of the environment.
Like it is done in the `train.py` script.

---

## Make your own track

In order to make your own track, run `python track_editor.py`. This script will open a window where you can
draw your own track. First you will need to draw the outer border of the track.
For this click on the window to create points. If you mess up, you can always press `c` to clear the track and start
over.
After finishing the outer border press `n` to close the loop and move on to the next step.
Now you can draw the inner border of the track.
After finishing the inner border press `n` again now you will see the track you have created.
In the next step you will draw the reward gates. These will be used to guide the Agent through the track as it will
receive a reward for passing through them. The first gate will be the finishline. All the gates should be places in
order of the track.
After finishing the reward gates press `n` again and place the agent start position and start direction.
After that press `s` to save the track. You can now use this track to train your agent.

Here is a short visual guide on how to use the track editor:

![Track Editor Gif](https://github.com/ProfessorNova/PPO-Car/blob/main/docs/creating_track.gif)

---

## Hyperparameters

The default hyperparameters are defined in the `train.py` script (you can also list them with `python train.py --help`).
If you have any issues with too little RAM, you might want to decrease the `n_envs` parameter.

---

## Performance

### My System:

- CPU: AMD Ryzen 9 5900X
- GPU: Nvidia RTX 3080 (12GB VRAM)
- RAM: 64GB DDR4
- OS: Windows 11

### Training:

The video from above was done on the `big_track.json` with the following hyperparameters:

- `n_envs`: 24
- `n_epochs`: 200
- `n_steps`: 1024
- `batch_size`: 512
- `train_iters`: 40
- `gamma`: 0.99
- `gae_lambda`: 0.95
- `clip_coef`: 0.2
- `vf_coef`: 0.5
- `ent_coef`: 0.001
- `max_grad_norm`: 1.0
- `learning_rate`: 3e-4
- `learning_rate_decay`: 0.99
- `reward_scaling`: 0.1

Reward:

![Reward](https://github.com/ProfessorNova/PPO-Car/blob/main/docs/charts_avg_reward.svg)

Policy_loss:

![Policy Loss](https://github.com/ProfessorNova/PPO-Car/blob/main/docs/losses_policy_loss.svg)

Value_loss:

![Value Loss](https://github.com/ProfessorNova/PPO-Car/blob/main/docs/losses_value_loss.svg)

Entropy_loss:

![Entropy](https://github.com/ProfessorNova/PPO-Car/blob/main/docs/losses_entropy.svg)