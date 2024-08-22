# PPO-Car

---

## Results

![Demo Gif](https://github.com/ProfessorNova/PPO-Car/blob/main/docs/demo.gif)

The configuration for this run is listed below. The training process took about 35 minutes.

---

## Installation

To get started with this project, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/ProfessorNova/PPO-Car.git
    cd PPO-Car
    ```

2. **Set Up Python Environment**:
   Make sure you have Python installed (tested with Python 3.10.11).

3. **Install Dependencies**:
   Run the following command to install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

   For proper PyTorch installation, visit [pytorch.org](https://pytorch.org/get-started/locally/) and follow the
   instructions based on your system configuration.
   
5. **Train the Model**:
   To start training the model, run:
    ```bash
    python train.py --run-name "my_run"
    ```
   To train using a GPU, add the `--cuda` flag:
    ```bash
    python train.py --run-name "my_run" --cuda
    
6. A window will open where you can select the track you want to train on.
   (You can also create your own track with the`track_editor.py` script. More on that below)
   
7. **Monitor Training Progress**:
   You can monitor the training progress by viewing the videos in the `videos` folder or by looking at the graphs in
   TensorBoard (you might need to install tensorboard):
    ```bash
    tensorboard --logdir "logs"
    ```

---

## Environment

### Description

This environment simulates a simple 2D car driving on a track. The track layout is defined by a JSON file. The objective is for the car to navigate the track, passing through reward gates while avoiding walls. The car has adjustable velocity and can make sharp turns.

### Action Space

The action space is a `Discrete(9)` space with the following actions:

- `0`: Move forward
- `1`: Move backward
- `2`: Turn left
- `3`: Turn right
- `4`: Move forward-left
- `5`: Move forward-right
- `6`: Move backward-left
- `7`: Move backward-right
- `8`: Do nothing (reduces velocity)

### Observation Space

The observation space is a `Box(6 + num_rays,)` with the following features:

- `0`: Normalized x position (range: 0 to 1)
- `1`: Normalized y position (range: 0 to 1)
- `2`: Normalized x velocity (range: -1 to 1)
- `3`: Normalized y velocity (range: -1 to 1)
- `4`: Cosine of the car’s angle (range: -1 to 1)
- `5`: Sine of the car’s angle (range: -1 to 1)
- `6` to `6 + num_rays`: Distance to the nearest wall for each ray

### Rewards

- **+0.01**: For taking a forward action.
- **+1.0**: For passing through a reward gate.
- **-3.0**: For hitting a wall.
- **+10.0**: For completing a lap.

### Starting State

The car starts at a predefined position and direction, configurable in the track JSON file.

### Episode Termination

An episode ends if the car hits a wall or if the maximum time step count (1000) is reached.

### Track Configuration

You can set the path to the track JSON file in the environment’s reset function. This is demonstrated in the `train.py` script.

---

## Creating Your Own Track

To create a custom track, follow these steps:

**1. Run the Track Editor Script:**

Execute the following command in your terminal to launch the track editor:

```bash
python track_editor.py
```

This opens a window where you can draw the track layout.

**2. Draw the Outer Border:**

- Click within the window to place points and define the outer border of the track.
- If you make a mistake, press `c` to clear the entire track and start over.
- Once satisfied with the outer border, press `n` to close the loop and move to the next step.

**3. Draw the Inner Border:**

- Follow the same process to draw the inner border of the track.
- Press `n` once you've completed the inner border to proceed.

**4. Place Reward Gates:**

- Place the reward gates along the track. The first gate serves as the finish line.
- Ensure the gates are placed in the order they should be passed by the car.
- Press `n` after placing all the gates.

**5. Set the Start Position and Direction:**

- Click to place the car's starting position and define its initial direction.
- When ready, press `s` to save the track.

For a visual guide, refer to this GIF:

![Track Editor Gif](https://github.com/ProfessorNova/PPO-Car/blob/main/docs/creating_track.gif)

---

## Hyperparameters

The default hyperparameters used in training are defined in the `parse_args()` function inside the `train.py` script. You can also list them by running the following command:

```bash
python train.py --help
```

---

## Troubleshooting

If your system has limited RAM, consider lowering the **n_envs** parameter to reduce memory usage.

---

## Performance

### System Specifications:

Here are the specifications of the system used for training:

- **CPU**: AMD Ryzen 9 5900X
- **GPU**: Nvidia RTX 3080 (12GB VRAM)
- **RAM**: 64GB DDR4
- **OS**: Windows 11

### Training Configuration:

The training process utilized the `big_track.json` file with the following hyperparameters:

- **n_envs**: 24
- **n_epochs**: 200
- **n_steps**: 1024
- **batch_size**: 512
- **train_iters**: 40
- **gamma**: 0.99
- **gae_lambda**: 0.95
- **clip_coef**: 0.2
- **vf_coef**: 0.5
- **ent_coef**: 0.001
- **max_grad_norm**: 1.0
- **learning_rate**: 3e-4
- **learning_rate_decay**: 0.99
- **reward_scaling**: 0.1

### Performance Metrics:

The following charts provide insights into the performance during training:

- **Reward**:
  ![Reward](https://github.com/ProfessorNova/PPO-Car/blob/main/docs/charts_avg_reward.svg)

- **Policy Loss**:
  ![Policy Loss](https://github.com/ProfessorNova/PPO-Car/blob/main/docs/losses_policy_loss.svg)

- **Value Loss**:
  ![Value Loss](https://github.com/ProfessorNova/PPO-Car/blob/main/docs/losses_value_loss.svg)

- **Entropy Loss**:
  ![Entropy](https://github.com/ProfessorNova/PPO-Car/blob/main/docs/losses_entropy.svg)
