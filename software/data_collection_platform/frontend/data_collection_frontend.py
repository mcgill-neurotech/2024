import sys
import pygame
from pygame.locals import *
import threading


def update(
    on_start_cb,
    on_stop_cb,
    on_left_cb,
    on_right_cb,
    on_go_cb,
    on_rest_cb,
    on_left_done_cb,
    on_right_done_cb,
    on_go_done_cb,
    on_rest_done_cb,
):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Start the timer for the "GO" (clench) ellipse

                threading.Timer(10, on_go_done_cb).start()
                on_go_cb()

            elif event.key == pygame.K_LEFT:
                # Start the timer for the left ellipse

                threading.Timer(10, on_left_done_cb).start()
                on_left_cb()
            elif event.key == pygame.K_RIGHT:
                # Start the timer for the right ellipse

                threading.Timer(10, on_right_done_cb).start()
                on_right_cb()
            elif event.key == pygame.K_RETURN:
                # Start the timer for the rest rectangle

                threading.Timer(10, on_rest_done_cb).start()
                on_rest_cb()


def show_text(screen, text, position, font_size=80, color=(255, 255, 255)):
    font = pygame.font.SysFont("Times New Roman", font_size, True, False)
    surface = font.render(text, True, color)
    screen.blit(surface, position)


def draw_arrowhead(screen, color, start_pos, end_pos, size):
    # Calculate the direction vector and normalize it
    direction = pygame.math.Vector2(end_pos) - pygame.math.Vector2(start_pos)
    direction = direction.normalize()

    # Perpendicular vector to the direction
    perp_direction = pygame.math.Vector2(-direction.y, direction.x)

    # Define the points of the triangle relative to the end_pos
    point1 = pygame.math.Vector2(end_pos)
    side_length = size / (3**0.5)  # Calculate the side length of the triangle
    point2 = point1 - direction * size + perp_direction * side_length
    point3 = point1 - direction * size - perp_direction * side_length

    # Draw the triangle
    pygame.draw.polygon(screen, color, [point1, point2, point3])


def draw(
    screen, go_text_active, left_ellipse_active, right_ellipse_active, rest_rect_active
):
    # Background
    screen.fill((255, 255, 255))
    blue = pygame.Color("#284387")

    # Determine colors based on timers
    left_ellipse_color = (0, 255, 0) if left_ellipse_active else (220, 220, 220)
    right_ellipse_color = (0, 255, 0) if right_ellipse_active else (220, 220, 220)
    rest_rect_color = (blue) if rest_rect_active > 0 else (220, 220, 220)
    go_ellipse_color = (
        (0, 255, 0) if go_text_active > 0 else (220, 220, 220)
    )  # Color for "GO" ellipse

    # Ellipses
    ellipse_rect_left = pygame.Rect(25, 40, 200, 200)
    pygame.draw.ellipse(
        screen, left_ellipse_color, ellipse_rect_left, 0
    )  # Use dynamic color
    ellipse_rect_right = pygame.Rect(580, 40, 200, 200)
    pygame.draw.ellipse(
        screen, right_ellipse_color, ellipse_rect_right, 0
    )  # Use dynamic color
    ellipse_rect_go = pygame.Rect(230, 180, 330, 200)
    pygame.draw.ellipse(screen, go_ellipse_color, ellipse_rect_go, 0)

    # Rectangle
    rest_rect = pygame.Rect(250, 410, 300, 160)
    pygame.draw.rect(screen, rest_rect_color, rest_rect, 0)  # Use dynamic color

    # Lines
    arrow_color_l = (250, 250, 250)
    start_pos_l = (40, 133)
    end_pos_l = (170, 133)
    pygame.draw.line(screen, arrow_color_l, start_pos_l, end_pos_l, 30)
    draw_arrowhead(screen, (250, 250, 250), (190, 133), (210, 133), 50)

    arrow_color_r = (250, 250, 250)
    start_pos_r = (600, 133)
    end_pos_r = (760, 133)
    pygame.draw.line(screen, arrow_color_r, start_pos_r, end_pos_r, 30)
    draw_arrowhead(screen, (250, 250, 250), (580, 133), (600, 133), 50)

    # Text color logic

    show_text(screen, "GO", (330, 250))
    show_text(screen, "REST", (295, 470))

    pygame.display.flip()


def runPyGame(
    on_start=lambda: print("start"),
    on_stop=lambda: print("stop"),
    on_left=lambda: print("left"),
    on_right=lambda: print("right"),
    on_go=lambda: print("go"),
    on_rest=lambda: print("rest"),
    on_left_done=lambda: print("left done"),
    on_right_done=lambda: print("right done"),
    on_go_done=lambda: print("go done"),
    on_rest_done=lambda: print("rest done"),
):
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Data Collection UI")
    clock = pygame.time.Clock()
    fps = 60.0

    go_text_active = False
    left_ellipse_active = False
    right_ellipse_active = False
    rest_rect_active = False

    def on_left_cb():
        nonlocal left_ellipse_active
        left_ellipse_active = True
        on_left()

    def on_right_cb():
        nonlocal right_ellipse_active
        right_ellipse_active = True
        on_right()

    def on_go_cb():
        nonlocal go_text_active
        go_text_active = True
        on_go()

    def on_rest_cb():
        nonlocal rest_rect_active
        rest_rect_active = True
        on_rest()

    def on_left_done_cb():
        nonlocal left_ellipse_active
        left_ellipse_active = False
        on_left_done()

    def on_right_done_cb():
        nonlocal right_ellipse_active
        right_ellipse_active = False
        on_right_done()

    def on_go_done_cb():
        nonlocal go_text_active
        go_text_active = False
        on_go_done()

    def on_rest_done_cb():
        nonlocal rest_rect_active
        rest_rect_active = False
        on_rest_done()

    def on_start_cb():
        on_start()

    def on_stop_cb():
        on_stop()

    while True:
        # dt = clock.tick(fps) / 1000.0  # Convert milliseconds to seconds

        update(
            on_start_cb,
            on_stop_cb,
            on_left_cb,
            on_right_cb,
            on_go_cb,
            on_rest_cb,
            on_left_done_cb,
            on_right_done_cb,
            on_go_done_cb,
            on_rest_done_cb,
        )
        draw(
            screen,
            go_text_active,
            left_ellipse_active,
            right_ellipse_active,
            rest_rect_active,
        )


if __name__ == "__main__":
    runPyGame()
