import json
from typing import List, Optional, Union

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


def draw_rectangle(
    screen: pygame.Surface,
    x: Union[int, float],
    y: Union[int, float],
    width: Union[int, float],
    height: Union[int, float],
    color: str,
    rotation: float = 0,
) -> None:
    """
    Draw a rectangle, centered at (x, y).

    Args:
        screen (pygame.Surface): The surface to draw on.
        x (Union[int, float]): The x-coordinate of the center of the rectangle.
        y (Union[int, float]): The y-coordinate of the center of the rectangle.
        width (Union[int, float]): The width of the rectangle.
        height (Union[int, float]): The height of the rectangle.
        color (str): The fill color of the rectangle in HTML format.
        rotation (float, optional): The rotation angle of the rectangle in degrees. Defaults to 0.
    """
    points = []

    radius = np.sqrt((height / 2) ** 2 + (width / 2) ** 2)
    angle = np.arctan2(height / 2, width / 2)
    angles = [angle, -angle + np.pi, angle + np.pi, -angle]
    rot_radians = np.radians(rotation)

    for angle in angles:
        x_offset = radius * np.cos(angle + rot_radians)
        y_offset = radius * np.sin(angle + rot_radians)
        points.append((x + x_offset, y + y_offset))

    pygame.draw.polygon(screen, color, points)


class Boundary:
    """
    Boundary class for the environment.
    """

    def __init__(
        self,
        x1: Union[int, float],
        y1: Union[int, float],
        x2: Union[int, float],
        y2: Union[int, float],
    ):
        """
        Initialize a Boundary object.

        Args:
            Initialize a Boundary object.
            x1 (Union[int, float]): x-coordinate of the starting point of the boundary.
            y1 (Union[int, float]): y-coordinate of the starting point of the boundary.
            x2 (Union[int, float]): x-coordinate of the ending point of the boundary.
            y2 (Union[int, float]): y-coordinate of the ending point of the boundary.
        """
        self.__a = np.array([float(x1), float(y1)])
        self.__b = np.array([float(x2), float(y2)])

    def draw(self, screen: pygame.Surface, color: str = "white", thickness: int = 2):
        pygame.draw.line(screen, color, self.__a, self.__b, thickness)

    def get_points(self):
        return self.__a, self.__b


class Reward_gate(Boundary):
    """
    A class representing a reward gate in a car driving environment.
    """

    def __init__(
        self,
        x1: Union[int, float],
        y1: Union[int, float],
        x2: Union[int, float],
        y2: Union[int, float],
        gate_index: int,
    ):
        """
        Initialize a Reward_gate object.

        Args:
            x1 (Union[int, float]): x-coordinate of the starting point of the reward gate.
            y1 (Union[int, float]): y-coordinate of the starting point of the reward gate.
            x2 (Union[int, float]): x-coordinate of the ending point of the reward gate.
            y2 (Union[int, float]): y-coordinate of the ending point of the reward gate.
            gate_index (int): The index of the reward gate.
        """
        super().__init__(x1, y1, x2, y2)
        self.__a = np.array([float(x1), float(y1)])
        self.__b = np.array([float(x2), float(y2)])
        self.__active: bool = True
        self.__gate_index: int = gate_index

    def draw(self, screen: pygame.Surface, color: str = "green", thickness: int = 2):
        pygame.draw.line(screen, color, self.__a, self.__b, thickness)

    def pass_gate(self):
        self.__active = False

    def restore_gate(self):
        self.__active = True

    def is_active(self) -> bool:
        return self.__active

    def get_index(self) -> int:
        return self.__gate_index


class Ray:
    def __init__(self, x: Union[int, float], y: Union[int, float], angle: float):
        """
        Initialize a Ray object.

        Args:
            x (Union[int, float]): The x-coordinate of the ray's position.
            y (Union[int, float]): The y-coordinate of the ray's position.
            angle (float): The angle of the ray in degrees.
        """
        self.__pos = np.array([float(x), float(y)])
        self.__dir = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])

    def get_position(self) -> np.ndarray:
        return self.__pos

    def get_direction(self) -> np.ndarray:
        return self.__dir

    def draw(self, screen: pygame.Surface, color: str = "white", thickness: int = 1):
        pygame.draw.line(
            screen,
            color,
            self.__pos,
            self.__pos + 1000 * self.__dir,
            thickness,
        )

    def update(self, x: Union[int, float], y: Union[int, float], angle: float):
        self.__pos = np.array([float(x), float(y)])
        self.__dir = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])

    def cast(self, boundary: Boundary) -> Union[np.ndarray, None]:
        """
        Cast the ray and check for intersection with a boundary.

        Args:
            boundary (Boundary): The boundary object to check for intersection.

        Returns:
            Union[np.ndarray, None]: The intersection point as a numpy array of shape (2,) if there is an intersection,
            None otherwise.
        """
        x1, y1 = boundary.get_points()[0]
        x2, y2 = boundary.get_points()[1]
        x3, y3 = self.__pos
        x4, y4 = self.__pos + self.__dir

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        if t > 0 and t < 1 and u > 0:
            pt = np.array([0.0, 0.0])
            pt[0] = x1 + t * (x2 - x1)
            pt[1] = y1 + t * (y2 - y1)
            return pt
        else:
            return None

    def get_distance(self, boundary: Union[Boundary, List[Boundary]]) -> float:
        """
        Get the distance from the ray start to the closest intersection point with a boundary.

        Args:
            boundary (Union[Boundary, List[Boundary]]): The boundary object or list of boundary objects to check for
            intersection.

        Returns:
            float: The distance to the closest intersection point. If there is no intersection, return a large number.

        """
        largest_distance = 1000.0

        if isinstance(boundary, list):
            for b in boundary:
                pt = self.cast(b)
                if (
                    pt is not None
                    and np.linalg.norm(self.__pos - pt) < largest_distance
                ):
                    largest_distance = np.linalg.norm(self.__pos - pt)
        else:
            pt = self.cast(boundary)
            if pt is not None and np.linalg.norm(self.__pos - pt) < largest_distance:
                largest_distance = np.linalg.norm(self.__pos - pt)

        return largest_distance


class Car:
    def __init__(
        self,
        x: Union[int, float],
        y: Union[int, float],
        rotation: float = 0.0,
        image: Optional[str] = None,
        turn_speed: float = 5.0,
        max_speed: float = 10.0,
        max_acceleration: float = 0.5,
        friction: float = 0.3,
        num_rays: int = 12,
    ):
        """
        Initialize a Car object.

        Args:
            x (int or float): The x-coordinate of the car's starting position.
            y (int or float): The y-coordinate of the car's starting position.
            rotation (float, optional): The initial rotation of the car in degrees. Defaults to 0.0, which is facing right.
            image (str, optional): The path to the image file for the car's sprite. Defaults to None.
            turn_speed (float, optional): The turn speed of the car. Defaults to 5.0.
            max_speed (float, optional): The maximum speed of the car. Defaults to 10.0.
            max_acceleration (float, optional): The maximum acceleration of the car. Defaults to 0.8.
            friction (float, optional): The friction coefficient of the car. Defaults to 0.2.
            num_rays (int, optional): The number of rays emitted by the car. Defaults to 12.
        """
        self.__start_pos = np.array([float(x), float(y)])
        self.__pos = np.array([float(x), float(y)])
        self.__start_rotation = rotation
        self.__rotation = rotation
        self.__image_width = 32
        self.__image_height = 72
        if image is not None:
            self.__image = pygame.image.load(image)
            self.__image = pygame.transform.scale(
                self.__image, (self.__image_width, self.__image_height)
            )
            self.__sprite = self.__image.get_rect()
        else:
            self.__image = None
            self.__sprite = None
        self.__turn_speed = turn_speed
        self.__max_speed = max_speed

        self.__velocity = np.array([0.0, 0.0])
        self.__acceleration = np.array([0.0, 0.0])
        self.__max_acceleration = max_acceleration
        self.__friction = friction

        self.__rays = []
        self.__num_rays = num_rays
        for a in range(0, 360, 360 // self.__num_rays):
            self.__rays.append(Ray(self.__pos[0], self.__pos[1], self.__rotation + a))

        self.__destroyed = False

    def set_image(self, image: str):
        self.__image = pygame.image.load(image)
        self.__image = pygame.transform.scale(
            self.__image, (self.__image_width, self.__image_height)
        )  # change the size of the car image
        self.__sprite = self.__image.get_rect()

    def get_position(self) -> np.ndarray:
        return self.__pos

    def get_rotation(self) -> float:
        return self.__rotation

    def get_velocity(self) -> np.ndarray:
        return self.__velocity

    def get_num_rays(self) -> int:
        return self.__num_rays

    def is_destroyed(self) -> bool:
        return self.__destroyed

    def get_max_speed(self) -> float:
        return self.__max_speed

    def set_destroyed(self, destroyed: bool):
        self.__destroyed = destroyed

    def set_position(self, x: Union[int, float], y: Union[int, float]):
        self.__pos = np.array([float(x), float(y)])

    def set_initial_position(self, x: Union[int, float], y: Union[int, float]):
        self.__start_pos = np.array([float(x), float(y)])

    def set_rotation(self, rotation: float):
        self.__rotation = rotation

    def set_initial_rotation(self, rotation: float):
        self.__start_rotation = rotation

    def draw(self, screen: pygame.Surface):
        """
        Draw the car on the screen. If an image is provided, draw the image, otherwise draw a blue rectangle.

        Args:
            screen (pygame.Surface): The pygame screen object to draw the car on.
        """
        if self.__image is not None:
            updated_image = pygame.transform.rotate(
                self.__image, -(self.__rotation + 90)
            )
            self.__sprite = updated_image.get_rect(center=self.__pos)
            screen.blit(updated_image, self.__sprite)
        else:
            draw_rectangle(
                screen,
                self.__pos[0],
                self.__pos[1],
                self.__image_width,
                self.__image_height,
                "blue",
                self.__rotation - 90,
            )

    def draw_rays(
        self,
        screen: pygame.Surface,
        boundary: Union[Boundary, List[Boundary]],
        color: str = "white",
    ):
        """
        Draw the rays emitted by the car for debugging purposes.

        Args:
            screen (pygame.Surface): The pygame screen object to draw the rays on.
            boundary (Union[Boundary, List[Boundary]]): The boundary object or list of boundary objects to check for
            intersection.
            color (str, optional): The color of the rays. Defaults to "white".
        """
        distances = self.get_distances(boundary)
        for ray in self.__rays:
            # Draw the ray with the correct length
            pt = ray.get_position() + ray.get_direction() * distances.pop(0)
            pygame.draw.line(screen, color, ray.get_position(), pt)
            pygame.draw.circle(screen, "red", pt.astype(int), 5)

    def get_distances(self, boundary: Union[Boundary, List[Boundary]]) -> List[float]:
        """
        Get the distances to the closest intersection points for each ray.

        Args:
            boundary (Union[Boundary, List[Boundary]]): The boundary object or list of boundary objects to check for
            intersection.

        Returns:
            List[float]: A list of distances to the closest intersection points for each ray.
        """
        distances = []
        for ray in self.__rays:
            distances.append(ray.get_distance(boundary))
        return distances

    def check_collision(self, boundary: Union[Boundary, List[Boundary]]) -> bool:
        """
        Check if the car has collided with a boundary.

        Args:
            boundary (Union[Boundary, List[Boundary]]): The boundary object or list of boundary objects to check for
            intersection.

        Returns:
            bool: True if the car has collided with a boundary, False otherwise.
        """
        collision_distance = 20.0
        # Just check for the four directions
        for r in range(0, self.__num_rays, self.__num_rays // 4):
            if self.__rays[r].get_distance(boundary) < collision_distance:
                return True
        return False

    def get_passed_gate(self, gates: List[Reward_gate]) -> Union[Reward_gate, None]:
        """
        Check if the car has passed through a active reward gate.

        Args:
            gates (List[Reward_gate]): A list of reward gates to check for intersection.

        Returns:
            Union[Reward_gate, None]: The reward gate that the car has passed through, or None if no gate has been passed.
        """
        for gate in gates:
            if gate.is_active() and self.check_collision(gate):
                return gate
        return None

    def reset(self):
        self.__pos = self.__start_pos.copy()
        self.__rotation = self.__start_rotation
        self.__velocity = np.array([0.0, 0.0])
        self.__acceleration = np.array([0.0, 0.0])

    def move_car(self, direction: str):
        """
        Move the car in a given direction.

        Args:
            direction (str): The direction to move the car in. Can be one of "forward", "backward", "left", or "right".
        """
        if direction == "forward":
            force_dir = np.array(
                [
                    np.cos(np.radians(self.__rotation)),
                    np.sin(np.radians(self.__rotation)),
                ]
            )
            self.__acceleration = force_dir * self.__max_acceleration
        elif direction == "backward":
            force_dir = np.array(
                [
                    np.cos(np.radians(self.__rotation)),
                    np.sin(np.radians(self.__rotation)),
                ]
            )
            self.__acceleration = -force_dir * self.__max_acceleration
        elif direction == "left":
            self.__rotation -= self.__turn_speed
        elif direction == "right":
            self.__rotation += self.__turn_speed

    def update(self, boundary: Union[Boundary, List[Boundary]]):
        """
        Update the car's position and velocity.

        Args:
            boundary (Union[Boundary, List[Boundary]]): The boundary object or list of boundary objects to check for
            intersection.
        """
        self.__velocity += self.__acceleration
        # Apply friction if no acceleration
        if np.linalg.norm(self.__acceleration) == 0:
            self.__velocity *= 1 - self.__friction
        # Limit the speed
        self.__velocity = np.clip(self.__velocity, -self.__max_speed, self.__max_speed)

        self.__pos += self.__velocity

        self.__acceleration = np.array([0.0, 0.0])

        for a in range(0, 360, 360 // self.__num_rays):
            self.__rays[a // (360 // self.__num_rays)].update(
                self.__pos[0], self.__pos[1], self.__rotation + a
            )

        if self.check_collision(boundary):
            self.__destroyed = True


class Car_env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self, render_mode: Optional[str] = None, track_path: str = "tracks/track.json"
    ):
        """
        Initialize the Car_env object.

        Args:
            render_mode (Optional[str], optional): The rendering mode. Defaults to None.
            track_path (str, optional): The path to the track file. Defaults to "tracks/track.json".
        """
        super(Car_env, self).__init__()

        self.__width: int = 1280
        self.__height: int = 720
        self.__time_step: int = 0
        self.__time_limit: int = 1000

        self.__car = Car(
            0,
            0,
            image="car.png",
        )

        self.__current_track_path: str = track_path

        # Define the action and observation space
        low_obs = np.array([0.0, 0.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        high_obs = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        # Append the rays distances to the observation space
        for _ in range(self.__car.get_num_rays()):
            low_obs = np.append(low_obs, np.float32(0.0))
            high_obs = np.append(high_obs, np.float32(1.0))

        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float32
        )
        self.action_space = spaces.Discrete(9)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def load_track(self, track_path: str) -> dict:
        """
        Load the track data from a JSON file and upscale it.

        Args:
            track (str): The path to the track file.

        Returns:
            dict: The track data as a dictionary.
        """
        with open(track_path, "r") as f:
            data = json.load(f)

        # Upscale the track data
        scale_factor_x = self.__width
        scale_factor_y = self.__height
        data["outer_track_points"] = [
            [x * scale_factor_x, y * scale_factor_y]
            for x, y in data["outer_track_points"]
        ]
        data["inner_track_points"] = [
            [x * scale_factor_x, y * scale_factor_y]
            for x, y in data["inner_track_points"]
        ]
        data["reward_gates"] = [
            [x * scale_factor_x, y * scale_factor_y] for x, y in data["reward_gates"]
        ]
        data["initial_position"] = [
            data["initial_position"][0] * scale_factor_x,
            data["initial_position"][1] * scale_factor_y,
        ]

        return data

    def _get_obs(self):
        """
        Get the observation of the environment.

        Returns:
            np.array: The observation array.
        """
        obs = []

        # Normalize the car's position
        obs.append(self.__car.get_position()[0] / self.__width)  # x position
        obs.append(self.__car.get_position()[1] / self.__height)  # y position

        # Normalize the car's velocity
        obs.append(
            self.__car.get_velocity()[0] / self.__car.get_max_speed()
        )  # x velocity
        obs.append(
            self.__car.get_velocity()[1] / self.__car.get_max_speed()
        )  # y velocity

        # Use the car's rotation as the last two observations
        x, y = np.cos(np.radians(self.__car.get_rotation())), np.sin(
            np.radians(self.__car.get_rotation())
        )
        obs.append(x)  # x component of rotation
        obs.append(y)  # y component of rotation

        # Append the rays distances to the observation space
        distances = self.__car.get_distances(self.__boundaries)
        for d in distances:
            obs.append(d / 1000.0)  # Normalize the distance

        obs = np.array(obs, dtype=np.float32)

        return obs

    def _get_info(self):
        return {
            "gates_passed": self.__passed_reward_gates,
            "time_passed": self.__time_step,
        }

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state. Possible options include setting "no_time_limit" to True to remove
        the time limit and setting "track_path" to the path of the track file as a string.

        Args:
            seed (int, optional): The random seed for the environment. Defaults to None.
            options (dict, optional): Additional options for resetting the environment. Defaults to None.

        Returns:
            tuple: A tuple containing the observation and information about the environment.
        """
        super().reset(seed=seed)

        self.__time_step = 0
        if options and "no_time_limit" in options:
            if options["no_time_limit"]:
                self.__time_limit = np.inf
        else:
            self.__time_limit = 1000

        if options and "track_path" in options:
            # set the new track path
            self.__current_track_path = options["track_path"]
            try:
                self.__track_data = self.load_track(self.__current_track_path)
            except FileNotFoundError:
                print(f"Track file not found at {self.__current_track_path}.")
                return

        self.__outer_track_points = self.__track_data["outer_track_points"]
        self.__inner_track_points = self.__track_data["inner_track_points"]
        self.__reward_gates_points = self.__track_data["reward_gates"]
        self.__car.set_position(
            self.__track_data["initial_position"][0],
            self.__track_data["initial_position"][1],
        )
        self.__car.set_initial_position(
            self.__track_data["initial_position"][0],
            self.__track_data["initial_position"][1],
        )
        self.__car.set_rotation(self.__track_data["initial_angle"])
        self.__car.set_initial_rotation(self.__track_data["initial_angle"])
        self.__boundaries = []
        self.__reward_gates = []
        for b in range(len(self.__outer_track_points) - 1):
            self.__boundaries.append(
                Boundary(
                    self.__outer_track_points[b][0],
                    self.__outer_track_points[b][1],
                    self.__outer_track_points[b + 1][0],
                    self.__outer_track_points[b + 1][1],
                )
            )
        for b in range(len(self.__inner_track_points) - 1):
            self.__boundaries.append(
                Boundary(
                    self.__inner_track_points[b][0],
                    self.__inner_track_points[b][1],
                    self.__inner_track_points[b + 1][0],
                    self.__inner_track_points[b + 1][1],
                )
            )
        for i, (g_start, g_end) in enumerate(
            zip(self.__reward_gates_points[::2], self.__reward_gates_points[1::2])
        ):
            self.__reward_gates.append(
                Reward_gate(g_start[0], g_start[1], g_end[0], g_end[1], i)
            )
        self.__total_reward_gates = len(self.__reward_gates)
        self.__passed_reward_gates = 0
        self.__remaining_reward_gates = self.__total_reward_gates
        self.__next_gate_index = 0

        self.__car.reset()
        self.__car.set_destroyed(False)
        for gate in self.__reward_gates:
            gate.restore_gate()
        self.__car.update(self.__boundaries)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        reward = 0
        # action translation
        if action == 0:
            self.__car.move_car("forward")
            # reward for moving forward
            reward += 0.01
        elif action == 1:  # backward
            self.__car.move_car("backward")
        elif action == 2:  # left
            self.__car.move_car("left")
        elif action == 3:  # right
            self.__car.move_car("right")
        elif action == 4:  # forward left
            self.__car.move_car("forward")
            self.__car.move_car("left")
            reward += 0.01
        elif action == 5:  # forward right
            self.__car.move_car("forward")
            self.__car.move_car("right")
            reward += 0.01
        elif action == 6:  # backward left
            self.__car.move_car("backward")
            self.__car.move_car("left")
        elif action == 7:  # backward right
            self.__car.move_car("backward")
            self.__car.move_car("right")
        elif action == 8:  # do nothing
            pass

        # Update Reward Gates
        gate = self.__car.get_passed_gate(self.__reward_gates)
        if gate is not None and gate.get_index() == self.__next_gate_index:
            reward += 1  # reward for passing a gate
            self.__remaining_reward_gates -= 1
            # If all gates are passed, reset them
            if self.__remaining_reward_gates == 0:
                gate.pass_gate()
                # reward for completing a lap
                reward += 10
                self.__passed_reward_gates += 1
                for gate in self.__reward_gates:
                    gate.restore_gate()
                self.__remaining_reward_gates = self.__total_reward_gates
                self.__next_gate_index = 0
            else:
                gate.pass_gate()
                self.__passed_reward_gates += 1
                self.__next_gate_index += 1

        # Update car state and check for termination
        self.__car.update(self.__boundaries)
        self.__time_step += 1
        terminated = False
        if self.__car.is_destroyed():
            terminated = True
            reward -= 10  # penalty for collision
        elif self.__time_step >= self.__time_limit:
            terminated = True
            reward += 2  # reward for surviving till time limit

        # Observation and rendering
        obs = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self.__render_frame()

        return obs, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.__render_frame()

    def __render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.__width, self.__height))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.__width, self.__height))

        canvas.fill((11, 102, 35))

        # outer polygon
        pygame.draw.polygon(canvas, "gray", self.__outer_track_points)
        # inner polygon same color as background
        pygame.draw.polygon(canvas, (11, 102, 35), self.__inner_track_points)

        for boundary in self.__boundaries:
            boundary.draw(canvas, "black", 5)
        self.__car.draw(canvas)

        for gate in self.__reward_gates:
            if gate.is_active():
                if gate.get_index() == self.__next_gate_index:
                    gate.draw(canvas, "yellow", 5)
                else:
                    gate.draw(canvas, "green", 5)

        self.__car.draw_rays(canvas, self.__boundaries)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(pygame.surfarray.pixels3d(canvas), axes=(1, 0, 2))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


def game_loop():
    """
    A simple game loop for testing the environment with arrow keys.
    """
    env = Car_env(render_mode="human")
    _, _ = env.reset(options={"track_path": "tracks/track.json", "no_time_limit": True})
    done = False
    while not done:
        # Wait until pygame is initialized
        action = 8
        if pygame.get_init():
            # Get the user's key inputs
            keys = pygame.key.get_pressed()

            # Map the key inputs to actions
            if keys[pygame.K_UP] and keys[pygame.K_LEFT]:
                action = 4
            elif keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
                action = 5
            elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
                action = 6
            elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
                action = 7
            elif keys[pygame.K_UP]:
                action = 0
            elif keys[pygame.K_DOWN]:
                action = 1
            elif keys[pygame.K_LEFT]:
                action = 2
            elif keys[pygame.K_RIGHT]:
                action = 3
            else:
                action = 8

        obs, _, done, _, info = env.step(action)
        print(obs)
        env.render()
    env.close()


if __name__ == "__main__":
    game_loop()
