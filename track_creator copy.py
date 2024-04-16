import pygame
import json

# pygame setup
pygame.init()
WIDTH, HEIGHT = 1280, 720
ASPECT_RATIO = WIDTH / HEIGHT
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
clock = pygame.time.Clock()
running = True

mouse_pressed = False
corners = {"outer": [], "inner": []}
current_mode = "outer"

current_width, current_height = WIDTH, HEIGHT

# Helper functions to scale coordinates
def normalize_pos(pos, original_size):
    return pos[0] / original_size[0], pos[1] / original_size[1]

def scale_pos(normalized_pos, current_size):
    return int(normalized_pos[0] * current_size[0]), int(normalized_pos[1] * current_size[1])

def draw(finished=False):
    screen.fill((30, 30, 30))

    pygame.draw.circle(screen, "red", pygame.mouse.get_pos(), 3)

    for group in corners.keys():
        if len(corners[group]) > 1:
            scaled_corners = [scale_pos(corner, (current_width, current_height)) for corner in corners[group]]
            pygame.draw.lines(screen, "red", False, scaled_corners)

            if finished:
                pygame.draw.polygon(screen, (100, 100, 100), [scale_pos(corner, (current_width, current_height)) for corner in corners["outer"]])
                pygame.draw.polygon(screen, (30, 30, 30), [scale_pos(corner, (current_width, current_height)) for corner in corners["inner"]])

while running:
    # poll for events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        elif event.type == pygame.VIDEORESIZE:
            new_width = event.w
            new_height = event.h
            
            # Maintain aspect ratio
            new_aspect = new_width / new_height
            if new_aspect > ASPECT_RATIO:
                new_width = int(new_height * ASPECT_RATIO)
            else:
                new_height = int(new_width / ASPECT_RATIO)

            # Resize the screen
            screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
            current_width, current_height = new_width, new_height

        elif event.type == pygame.KEYDOWN:
            # complete the group
            if event.key == pygame.K_n:
                # append the first corner to the end of the array
                corners[current_mode][-1] = corners[current_mode][0]
                if current_mode == "outer":
                    current_mode = "inner"
                elif current_mode == "inner":
                    current_mode = None
            if event.key == pygame.K_s and current_mode is None:
                # save the track to a file
                with open("track.json", "w") as f:
                    json.dump(corners, f)
            if event.key == pygame.K_l:
                # load the track from a file
                with open("track.json", "r") as f:
                    corners = json.load(f)
                current_mode = None
            if event.key == pygame.K_c:
                # clear the track
                corners = {"outer": [], "inner": []}
                current_mode = "outer"

    normalized_mouse_pos = normalize_pos(pygame.mouse.get_pos(), (current_width, current_height))

    if current_mode is not None:
        # append the normalised position of the mouse to the end of the array
        if len(corners[current_mode]) > 1:
            corners[current_mode][-1] = normalized_mouse_pos
        elif len(corners[current_mode]) == 1:
            corners[current_mode].append(normalized_mouse_pos)

        # write the normalised position of the mouse to the array on click
        if pygame.mouse.get_pressed()[0] and not mouse_pressed:
            mouse_pressed = True
            corners[current_mode].append(normalized_mouse_pos)
        elif not pygame.mouse.get_pressed()[0] and mouse_pressed:
            mouse_pressed = False

    print(corners)

    # draw everything to the screen
    draw(current_mode is None)

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    dt = clock.tick(60) / 1000

pygame.quit()
