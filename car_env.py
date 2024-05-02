import os
import sys
from typing import List, Optional, Union

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


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

    def draw(self, screen, color: str = "white", thickness: int = 2):
        pygame.draw.line(screen, color, self.__a, self.__b, thickness)

    def get_points(self):
        return self.__a, self.__b


class Reward_gate(Boundary):
    """
    A class representing a reward gate in a car driving environment.
    """

    # class attributes
    __gates_remaining: int = 0
    __gates_total: int = 0
    __gates_passed: int = 0

    def __init__(
        self,
        x1: Union[int, float],
        y1: Union[int, float],
        x2: Union[int, float],
        y2: Union[int, float],
    ):
        """
        Initialize a Reward_gate object.

        Args:
            x1 (Union[int, float]): x-coordinate of the starting point of the reward gate.
            y1 (Union[int, float]): y-coordinate of the starting point of the reward gate.
            x2 (Union[int, float]): x-coordinate of the ending point of the reward gate.
            y2 (Union[int, float]): y-coordinate of the ending point of the reward gate.
        """
        super().__init__(x1, y1, x2, y2)
        self.__active: bool = True
        Reward_gate.__gates_total += 1
        Reward_gate.__gates_remaining += 1

    def draw(self, screen, color: str = "green", thickness: int = 2):
        if self.__active:
            pygame.draw.line(screen, color, self.__a, self.__b, thickness)
        else:
            pygame.draw.line(screen, "red", self.__a, self.__b, thickness)

    def pass_gate(self):
        if self.__active:
            Reward_gate.__gates_remaining -= 1
            Reward_gate.__gates_passed += 1
            self.__active = False

    def restore_gate(self):
        if not self.__active:
            Reward_gate.__gates_remaining += 1
            self.__active = True

    @classmethod
    def get_gates_remaining(cls) -> int:
        return cls.__gates_remaining

    @classmethod
    def get_gates_total(cls) -> int:
        return cls.__gates_total

    @classmethod
    def get_gates_passed(cls) -> int:
        return cls.__gates_passed


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

    def draw(self, screen, color: str = "white", thickness: int = 1):
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
        """
        self.__start_pos = np.array([float(x), float(y)])
        self.__pos = np.array([float(x), float(y)])
        self.__start_rotation = rotation
        self.__rotation = rotation
        if image is not None:
            self.__image = pygame.image.load(image)
            self.__image = pygame.transform.scale(self.__image, (50, 50))
            self.__sprite = self.__image.get_rect()
        else:
            self.__image = None
            self.__sprite = None
        self.__turn_speed = turn_speed
        self.__max_speed = max_speed
