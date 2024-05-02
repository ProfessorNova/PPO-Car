import json
import os
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pygame


def editor_init():
    """
    Initializes the Pygame library and sets up the global variables for the track editor.

    Globals:
        WIDTH: int - The initial width of the game window.
        HEIGHT: int - The initial height of the game window.
        ASPECT_RATIO: float - The aspect ratio of the game window.
        screen: pygame.Surface - The surface representing the game window.
        clock: pygame.time.Clock - The clock object for managing the game's frame rate.
        running: bool - A flag indicating whether the game is currently running.
        mouse_pressed: bool - A flag indicating whether the mouse button is currently pressed.
        track_data: dict - The dictionary containing track data.
            - outer_track_points: list - The list of tuples representing the x and y coordinates of the outer boundary points of the track.
            - inner_track_points: list - The list of tuples representing the x and y coordinates of the inner boundary points of the track.
            - reward_gates: list - The list of tuples representing the x and y coordinates of the reward gates on the track. One reward gate consists of two points.
                                 The first reward gate is the start/finish line.
            - initial_position: tuple - The initial position of the car on the track.
            - initial_angle: int - The initial angle of the car on the track.
        current_mode: str - The current mode of the track editor (e.g. outer_track_points, inner_track_points).
        current_width: int - The current width of the game window.
        current_height: int - The current height of the game window.
    """
    pygame.init()
    # globals
    global WIDTH, HEIGHT, ASPECT_RATIO, screen, clock, running, mouse_pressed, track_data, current_mode, current_width, current_height

    WIDTH = 1280
    HEIGHT = 720
    ASPECT_RATIO = WIDTH / HEIGHT

    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    running = True
    mouse_pressed = False

    track_data = {
        "outer_track_points": [],
        "inner_track_points": [],
        "reward_gates": [],  # the first reward gate is the start/finish line
        "initial_position": None,
        "initial_angle": None,
    }
    current_mode = "outer_track_points"

    current_width, current_height = WIDTH, HEIGHT


def normalize_pos(pos: tuple, original_size: tuple) -> tuple:
    """
    Normalizes the position of a point based on the original size of the game window.

    Args:
        pos: tuple - The x and y coordinates of the point.
        original_size: tuple - The original width and height of the game window.

    Returns:
        tuple: The normalized x and y coordinates of the point.
    """
    normalized_x = round(pos[0] / original_size[0], 4)
    normalized_y = round(pos[1] / original_size[1], 4)
    return normalized_x, normalized_y


def scale_pos(normalized_pos: tuple, current_size: tuple) -> tuple:
    """
    Scales the normalized position of a point based on the current size of the game window.

    Args:
        normalized_pos: tuple - The normalized x and y coordinates of the point.
        current_size: tuple - The current width and height of the game window.

    Returns:
        tuple: The scaled x and y coordinates of the point.
    """
    return int(normalized_pos[0] * current_size[0]), int(
        normalized_pos[1] * current_size[1]
    )


def save_track_data():
    """
    Saves the track data to a JSON file, with the file name chosen through a file explorer dialog,
    defaulting to 'track.json'.
    """
    # Check if the data is complete
    if current_mode is not None:
        print("Track data still incomplete!")
        return

    # Set up the Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window

    # Define the folder path and ensure it exists
    folder_path = "tracks"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Open the save file dialog with a default filename
    file_path = filedialog.asksaveasfilename(
        initialdir=folder_path,
        title="Save track data",
        defaultextension=".json",
        filetypes=[("JSON Files", "*.json")],
        initialfile="track.json",  # Set the default file name
    )
    root.destroy()  # Close the tkinter instance

    # Check if a file path was selected
    if file_path:
        try:
            with open(file_path, "w") as file:
                json.dump(track_data, file)
                print(f"Track data saved to {file_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("No file selected. Saving aborted.")


def select_file():
    """
    Opens a file dialog to select a JSON file from the tracks folder.

    Returns:
        str: The path to the selected file, or None if no file was selected.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window
    # Set the directory to the tracks folder and filter to show only JSON files
    file_path = filedialog.askopenfilename(
        initialdir="tracks",
        title="Select track data file",
        filetypes=[("JSON Files", "*.json")],
    )
    root.destroy()  # Close the tkinter instance
    return file_path


def load_track_data():
    """
    Opens a file selection dialog and loads the track data from the chosen JSON file.
    """
    file_path = select_file()
    if file_path:
        try:
            with open(file_path, "r") as file:
                global track_data
                track_data = json.load(file)
                # Set the current mode to None to indicate that the track data is complete
                global current_mode
                current_mode = None
                print(f"Loaded track data from {file_path}")
        except FileNotFoundError:
            print("File not found. Please select a valid file.")
        except Exception as e:
            print(f"An error occurred: {e}")
    else:
        print("No file selected.")


def clear_track_data():
    """
    Clears the track data.
    """
    global track_data, current_mode
    track_data = {
        "outer_track_points": [],
        "inner_track_points": [],
        "reward_gates": [],
        "initial_position": None,
        "initial_angle": None,
    }
    current_mode = "outer_track_points"


def recalculate_width_height(new_width, new_height):
    """
    Recalculates the width and height of the game window while maintaining the aspect ratio.

    Args:
        new_width: int - The new width of the game window.
        new_height: int - The new height of the game window.

    Returns:
        tuple: The recalculated width and height of the game window.
    """
    new_aspect = new_width / new_height
    if new_aspect > ASPECT_RATIO:
        new_width = int(new_height * ASPECT_RATIO)
    else:
        new_height = int(new_width / ASPECT_RATIO)

    return new_width, new_height


def next_mode():
    """
    Switches to the next mode of the track editor.
    """
    global current_mode, track_data
    if current_mode == "outer_track_points":
        track_data["outer_track_points"].append(track_data["outer_track_points"][0])
        current_mode = "inner_track_points"
    elif current_mode == "inner_track_points":
        track_data["inner_track_points"].append(track_data["inner_track_points"][0])
        current_mode = "reward_gates"
    elif current_mode == "reward_gates":
        current_mode = "initial_position"
    elif current_mode == "initial_position":
        current_mode = "initial_angle"
    elif current_mode == "initial_angle":
        current_mode = None


def add_point(pos):
    """
    Adds a point to the current mode of the track editor.

    Args:
        pos: tuple - The x and y coordinates of the point. The coordinates are expected to be normalized.
    """
    global track_data, current_mode
    # different handling for initial position and angle
    if current_mode == "initial_position":
        track_data["initial_position"] = pos
        next_mode()
    elif current_mode == "initial_angle":
        scaled_initial_position = scale_pos(
            track_data["initial_position"], (current_width, current_height)
        )
        scaled_current_position = scale_pos(pos, (current_width, current_height))
        angle_rad = np.arctan2(
            scaled_current_position[1] - scaled_initial_position[1],
            scaled_current_position[0] - scaled_initial_position[0],
        )
        track_data["initial_angle"] = np.degrees(angle_rad)
        next_mode()
    # handling the rest of the points
    elif current_mode is not None:
        track_data[current_mode].append(pos)


def draw_arrow(screen, origin, angle, length=30, color=(255, 255, 0), thickness=2):
    """
    Draws an arrow on a given Pygame screen at a specified origin point, angle, length, and color.

    Args:
        screen (pygame.Surface): The Pygame surface on which to draw the arrow.
        origin (tuple): The (x, y) coordinates for the start of the arrow.
        angle (float): The angle in degrees for the direction of the arrow, where 0 degrees is to the right.
        length (int, optional): The length of the arrow. Defaults to 30 pixels.
        color (tuple, optional): The color of the arrow in RGB format. Defaults to yellow (255, 255, 0).
        thickness (int, optional): The thickness of the arrow line. Defaults to 2 pixels.

    Notes:
        The arrow is drawn with the specified angle using trigonometric functions to calculate the endpoint.
        The arrowhead is created by calculating two additional lines forming a simple triangle at the end of the main line.
    """
    # Convert angle from degrees to radians
    angle_rad = np.radians(angle)

    # Calculate the endpoint of the arrow
    end_point = (
        origin[0] + length * np.cos(angle_rad),
        origin[1] + length * np.sin(angle_rad),
    )

    # Draw the main line of the arrow
    pygame.draw.line(screen, color, origin, end_point, thickness)

    # Calculate arrowhead points (simple triangle)
    arrowhead_length = 10  # Length of the arrowhead sides
    arrowhead_angle = np.pi / 6  # 30 degrees in radians for each side of the arrowhead

    arrowhead_point1 = (
        end_point[0] - arrowhead_length * np.cos(angle_rad - arrowhead_angle),
        end_point[1] - arrowhead_length * np.sin(angle_rad - arrowhead_angle),
    )
    arrowhead_point2 = (
        end_point[0] - arrowhead_length * np.cos(angle_rad + arrowhead_angle),
        end_point[1] - arrowhead_length * np.sin(angle_rad + arrowhead_angle),
    )

    # Draw the arrowhead
    pygame.draw.polygon(screen, color, [end_point, arrowhead_point1, arrowhead_point2])


def draw_text(screen, text, position, font, color=(255, 255, 255)):
    """
    Draws multiline text on the given Pygame screen.

    Args:
        screen (pygame.Surface): The Pygame surface to draw on.
        text (str): The text to be displayed.
        position (tuple): The (x, y) position where the text will start.
        font (pygame.font.Font): The font object used for rendering the text.
        color (tuple, optional): The color of the text in RGB format. Defaults to white (255, 255, 255).
    """
    x, y = position
    line_height = font.get_linesize()  # Get the height of a line of text
    for line in text.split("\n"):  # Split the text into lines
        line_surface = font.render(line, True, color)
        screen.blit(line_surface, (x, y))
        y += line_height  # Move down to draw the next line


def draw():
    """
    Draws the track editor interface.
    """
    # setting up colors
    background_color = (30, 30, 30)  # dark grey
    outer_track_color = (255, 30, 30)  # lighter red
    inner_track_color = (255, 100, 0)  # red
    road_color = (128, 128, 128)  # grey
    finish_line_color = (255, 255, 0)  # yellow
    reward_gate_color = (0, 255, 0)  # green
    initial_position_color = (0, 255, 255)  # cyan

    screen.fill(background_color)

    # Draw the track data
    # Outer track points
    scaled_outer_track_points = [
        scale_pos(point, (current_width, current_height))
        for point in track_data["outer_track_points"]
    ]
    if len(track_data["outer_track_points"]) > 1:
        pygame.draw.lines(screen, outer_track_color, False, scaled_outer_track_points)
    # Draw one last line to the cursor
    if (
        current_mode == "outer_track_points"
        and len(track_data["outer_track_points"]) > 0
    ):
        pygame.draw.line(
            screen,
            outer_track_color,
            scaled_outer_track_points[-1],
            pygame.mouse.get_pos(),
            3,
        )

    # Inner track points
    scaled_inner_track_points = [
        scale_pos(point, (current_width, current_height))
        for point in track_data["inner_track_points"]
    ]
    if len(track_data["inner_track_points"]) > 1:
        pygame.draw.lines(screen, inner_track_color, False, scaled_inner_track_points)
    # Draw one last line to the cursor
    if (
        current_mode == "inner_track_points"
        and len(track_data["inner_track_points"]) > 0
    ):
        pygame.draw.line(
            screen,
            inner_track_color,
            scaled_inner_track_points[-1],
            pygame.mouse.get_pos(),
            3,
        )

    # draw polygon if outer and inner track points are complete
    # check if the first and last points are the same
    if (
        len(track_data["outer_track_points"]) > 1
        and len(track_data["inner_track_points"]) > 1
        and track_data["outer_track_points"][0] == track_data["outer_track_points"][-1]
        and track_data["inner_track_points"][0] == track_data["inner_track_points"][-1]
    ):
        pygame.draw.polygon(screen, road_color, scaled_outer_track_points)
        pygame.draw.polygon(screen, background_color, scaled_inner_track_points)

    # Reward gates
    # the first reward gate is the start/finish line
    if len(track_data["reward_gates"]) > 1:
        for i in range(0, len(track_data["reward_gates"]), 2):
            if i + 1 < len(track_data["reward_gates"]):
                pygame.draw.lines(
                    screen,
                    (
                        # special color for the start/finish line
                        finish_line_color
                        if i == 0
                        else reward_gate_color
                    ),
                    False,
                    [
                        scale_pos(point, (current_width, current_height))
                        for point in track_data["reward_gates"][i : i + 2]
                    ],
                )
    # Draw one last line to the cursor
    if current_mode == "reward_gates" and len(track_data["reward_gates"]) % 2 == 1:
        pygame.draw.line(
            screen,
            (
                finish_line_color
                if len(track_data["reward_gates"]) == 1
                else reward_gate_color
            ),
            scale_pos(track_data["reward_gates"][-1], (current_width, current_height)),
            pygame.mouse.get_pos(),
            3,
        )

    # Draw the initial position and angle
    # If the initial position is not set, draw it at the current position of the cursor with angle 0
    if track_data["initial_position"] is None and current_mode == "initial_position":
        initial_position = normalize_pos(
            pygame.mouse.get_pos(), (current_width, current_height)
        )
        scaled_initial_position = scale_pos(
            initial_position, (current_width, current_height)
        )
        pygame.draw.circle(screen, initial_position_color, scaled_initial_position, 5)
        draw_arrow(screen, scaled_initial_position, 0, 30, initial_position_color, 2)

    # If the initial position is set but the angle is not, draw the angle from the initial position to the cursor
    elif track_data["initial_angle"] is None and current_mode == "initial_angle":
        scaled_initial_position = scale_pos(
            track_data["initial_position"],
            (current_width, current_height),
        )
        pygame.draw.circle(screen, initial_position_color, scaled_initial_position, 5)
        angle_rad = np.arctan2(
            pygame.mouse.get_pos()[1] - scaled_initial_position[1],
            pygame.mouse.get_pos()[0] - scaled_initial_position[0],
        )
        draw_arrow(
            screen,
            scaled_initial_position,
            np.degrees(angle_rad),
            30,
            initial_position_color,
            2,
        )

    # If the initial position and angle are set, draw the angle from the initial position
    elif (
        track_data["initial_position"] is not None
        and track_data["initial_angle"] is not None
    ):
        scaled_initial_position = scale_pos(
            track_data["initial_position"],
            (current_width, current_height),
        )
        pygame.draw.circle(screen, initial_position_color, scaled_initial_position, 5)
        draw_arrow(
            screen,
            scaled_initial_position,
            track_data["initial_angle"],
            30,
            initial_position_color,
            2,
        )

    # Draw the cursor
    if current_mode == "outer_track_points":
        cursor_color = outer_track_color
    elif current_mode == "inner_track_points":
        cursor_color = inner_track_color
    elif current_mode == "reward_gates":
        if len(track_data["reward_gates"]) < 2:
            cursor_color = finish_line_color
        else:
            cursor_color = reward_gate_color
    elif current_mode == "initial_position":
        cursor_color = initial_position_color
    else:
        cursor_color = (255, 255, 255)

    pygame.draw.circle(
        screen,
        cursor_color,
        pygame.mouse.get_pos(),
        3,
    )

    # Set up the font for the info text
    info_font = pygame.font.SysFont(None, 20)
    info_text = ""

    if current_mode == "outer_track_points":
        info_text = "Mode: Draw Outer Track Points\n"
        info_text += "Left click to add points to the outer track.\n"
        info_text += "Press 'N' to close the loop\n"
        info_text += "and switch to the next mode.\n"
        info_text += "Press 'C' to clear the track data.\n\n"

    elif current_mode == "inner_track_points":
        info_text = "Mode: Draw Inner Track Points\n"
        info_text += "Left click to add points to the inner track.\n"
        info_text += "Press 'N' to close the loop\n"
        info_text += "and switch to the next mode.\n"
        info_text += "Press 'C' to clear the track data.\n\n"

    elif current_mode == "reward_gates":
        info_text = "Mode: Draw Reward Gates\n"
        info_text += "The first reward gate is the start/finish line.\n"
        info_text += "Left click to add reward gates.\n"
        info_text += "Press 'N' to switch to the next mode.\n"
        info_text += "Press 'C' to clear the track data.\n\n"

    elif current_mode == "initial_position":
        info_text = "Mode: Set Initial Position\n"
        info_text += "Left click to set the initial position of the car.\n"
        info_text += "Press 'C' to clear the track data.\n\n"

    elif current_mode == "initial_angle":
        info_text = "Mode: Set Initial Angle\n"
        info_text += "Left click to set the initial angle of the car.\n"
        info_text += "Press 'C' to clear the track data.\n\n"

    elif current_mode is None:
        info_text = "Track data complete!\n"
        info_text += "Press 'S' to save the track data.\n"
        info_text += "Press 'C' to clear the track data.\n\n"

    info_text += "Or press 'L' to load a track.\n"
    info_text += "Press 'Q' to quit."

    # Draw the information text
    draw_text(
        screen,
        info_text,
        (
            current_width - 300,
            current_height - 150,
        ),
        info_font,
        (255, 255, 0),
    )

    pygame.display.flip()


def main():
    global running, current_width, current_height, track_data, mouse_pressed
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                new_width, new_height = recalculate_width_height(event.w, event.h)
                screen = pygame.display.set_mode(
                    (new_width, new_height), pygame.RESIZABLE
                )
                current_width, current_height = new_width, new_height

            elif event.type == pygame.KEYDOWN:
                # Switch to the next mode
                if event.key == pygame.K_n:
                    next_mode()
                # Save the track data
                elif event.key == pygame.K_s:
                    # TODO: Implement input for the file name
                    save_track_data()
                # Load the track data
                elif event.key == pygame.K_l:
                    # TODO: Implement input for the file name
                    load_track_data()
                # Clear the track data
                elif event.key == pygame.K_c:
                    clear_track_data()
                # Quit the editor
                elif event.key == pygame.K_q:
                    running = False

            # Add a point on mouse click
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if not mouse_pressed:
                    add_point(
                        normalize_pos(
                            pygame.mouse.get_pos(), (current_width, current_height)
                        )
                    )
                if event.button == 1:
                    mouse_pressed = True
            # Reset the mouse_pressed flag on mouse button release
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_pressed = False

        draw()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    editor_init()
    main()
