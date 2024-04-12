import pygame

# pygame setup
pygame.init()
WIDTH, HEIGHT = 1280, 720  # Baseline dimensions
ASPECT_RATIO = WIDTH / HEIGHT
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
clock = pygame.time.Clock()
running = True

# Helper functions to scale coordinates and dimensions
def scale_pos(pos, original_size, current_size):
    return int(pos[0] * current_size[0] / original_size[0]), int(pos[1] * current_size[1] / original_size[1])

def scale_radius(radius, original_size, current_size):
    # Average scale factor (simple approximation)
    scale_factor = (current_size[0] / original_size[0] + current_size[1] / original_size[1]) / 2
    return int(radius * scale_factor)

mouse_pressed = False
corners = []

current_width, current_height = WIDTH, HEIGHT

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

            screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
            current_width, current_height = new_width, new_height

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("grey")

    # Scaling the mouse position and drawing a circle
    scaled_mouse_pos = scale_pos(pygame.mouse.get_pos(), (WIDTH, HEIGHT), (current_width, current_height))
    pygame.draw.circle(screen, "red", scaled_mouse_pos, scale_radius(5, (WIDTH, HEIGHT), (current_width, current_height)))

    if pygame.mouse.get_pressed()[0] and not mouse_pressed:
        mouse_pressed = True
        corners.append(scaled_mouse_pos)
    elif not pygame.mouse.get_pressed()[0] and mouse_pressed:
        mouse_pressed = False

    if len(corners) > 1:
        corners[-1] = scaled_mouse_pos
        scaled_corners = [scale_pos(corner, (WIDTH, HEIGHT), (current_width, current_height)) for corner in corners]
        pygame.draw.lines(screen, "red", False, scaled_corners)
    elif len(corners) == 1:
        corners.append(scaled_mouse_pos)

    print(corners)

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    dt = clock.tick(60) / 1000

pygame.quit()
