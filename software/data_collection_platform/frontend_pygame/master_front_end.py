import sys
import pygame
from pygame.locals import *
import threading


class Context:
    def __init__(
        self,
        train_sequence,
        work_duration,
        rest_duration,
        on_left,
        on_right,
        on_go,
        on_rest,
        on_start,
        on_stop,
    ):
        self.train_sequence = train_sequence
        self.go_text_active = False
        self.left_ellipse_active = False
        self.right_ellipse_active = False
        self.rest_rect_active = False
        self.train_index = 0
        self.work_duration = work_duration
        self.rest_duration = rest_duration

        self._on_left = on_left
        self._on_right = on_right
        self._on_go = on_go
        self._on_rest = on_rest

        self._on_start = on_start
        self._on_stop = on_stop

        self.current_active = ""

    def on_left(self):
        self.current_active = "left"
        self._on_left()
        threading.Timer(self.work_duration, self.on_work_done).start()

    def on_right(self):
        self.current_active = "right"
        self._on_right()
        threading.Timer(self.work_duration, self.on_work_done).start()

    def on_go(self):
        self.current_active = "go"
        self._on_go()
        threading.Timer(self.work_duration, self.on_work_done).start()

    def on_rest(self, rest_duration=None):
        self.current_active = "rest"
        self._on_rest()

        if self.train_index >= len(self.train_sequence):
            print("finished.")
            self._on_stop()
            return

        if rest_duration is None:
            rest_duration = self.rest_duration

        threading.Timer(rest_duration, self.on_rest_done).start()

    def on_work_done(self):
        self.current_active = ""
        self.train_index += 1
        self.on_rest()

    def on_rest_done(self):
        self.current_active = ""
        if self.train_sequence[self.train_index] == 0:
            self.on_left()
        elif self.train_sequence[self.train_index] == 1:
            self.on_right()
        elif self.train_sequence[self.train_index] == 2:
            self.on_go()

    def on_start(self):
        self._on_start()

        # rest for 5 seconds before starting the first task
        self.on_rest(5)

    def get_active(self):
        return self.current_active


def update(
    on_start_cb,
    #    on_left_cb,
    #    on_right_cb,
    #    on_go_cb,
    #    on_rest_cb,
    #    on_left_done_cb,
    #    on_right_done_cb,
    #    on_go_done_cb,
    #    on_rest_done_cb,
):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                on_start_cb()

            # if event.key == pygame.K_SPACE:
            #     # Start the timer for the "GO" (clench) ellipse

            #     threading.Timer(10, on_go_done_cb).start()
            #     on_go_cb()

            # elif event.key == pygame.K_LEFT:
            #     # Start the timer for the left ellipse

            #     threading.Timer(10, on_left_done_cb).start()
            #     on_left_cb()
            # elif event.key == pygame.K_RIGHT:
            #     # Start the timer for the right ellipse

            #     threading.Timer(10, on_right_done_cb).start()
            #     on_right_cb()
            # elif event.key == pygame.K_RETURN:
            #     # Start the timer for the rest rectangle

            #     threading.Timer(10, on_rest_done_cb).start()
            #     on_rest_cb()


def show_text(screen, text, position, font_size=80, color=(255, 255, 255)):
    font = pygame.font.SysFont("Times New Roman", font_size, True, False)
    surface = font.render(text, True, color)
    screen.blit(surface, position)


def show_buttons_text(screen, text, position, font_size=20, color=(255, 255, 255)):
    font = pygame.font.SysFont("Times New Roman", font_size, True, False)
    surface = font.render(text, True, color)
    screen.blit(surface, position)


def draw_arrowhead_l(screen, color, start_pos, end_pos, size):
    # Calculate the direction vector and normalize it
    direction_l = pygame.math.Vector2(end_pos) - pygame.math.Vector2(start_pos)
    direction_l = direction_l.normalize()

    # Perpendicular vector to the direction
    perp_direction = pygame.math.Vector2(-direction_l.y, direction_l.x)

    # Define the points of the triangle relative to the end_pos
    point1 = pygame.math.Vector2(end_pos)
    side_length = size / (3**0.5)  # Calculate the side length of the triangle
    point2 = point1 - direction_l * size + perp_direction * side_length
    point3 = point1 - direction_l * size - perp_direction * side_length

    # Draw the triangle
    pygame.draw.polygon(screen, color, [point1, point2, point3])


def draw_arrowhead_r(screen, color, start_pos, end_pos, size):
    # Calculate the direction vector and normalize it
    direction_r = pygame.math.Vector2(start_pos) - pygame.math.Vector2(end_pos)
    direction_r = direction_r.normalize()

    # Perpendicular vector to the direction
    perp_direction = pygame.math.Vector2(-direction_r.y, direction_r.x)

    # Define the points of the triangle relative to the end_pos
    point1 = pygame.math.Vector2(end_pos)
    side_length = size / (3**0.5)  # Calculate the side length of the triangle
    point2 = point1 - direction_r * size + perp_direction * side_length
    point3 = point1 - direction_r * size - perp_direction * side_length

    # Draw the triangle
    pygame.draw.polygon(screen, color, [point1, point2, point3])


def draw(screen, ctx: Context):

    # Background
    screen.fill((255, 255, 255))
    blue = pygame.Color("#284387")

    active = ctx.get_active()
    left_ellipse_color = (0, 255, 0) if active == "left" else (220, 220, 220)
    right_ellipse_color = (0, 255, 0) if active == "right" else (220, 220, 220)
    rest_rect_color = (blue) if active == "rest" else (220, 220, 220)
    go_ellipse_color = (0, 255, 0) if active == "go" else (220, 220, 220)

    start_rect_color = blue
    stop_rect_color = blue

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

    # Rest Rectangle
    rest_rect = pygame.Rect(250, 440, 300, 140)
    pygame.draw.rect(screen, rest_rect_color, rest_rect, 0)  # Use dynamic color

    # Start button
    start_rect = pygame.Rect(660, 0, 52, 27)
    pygame.draw.rect(screen, start_rect_color, start_rect, 0)

    # Stop button
    stop_rect = pygame.Rect(720, 0, 52, 27)
    pygame.draw.rect(screen, stop_rect_color, stop_rect, 0)

    # Lines
    arrow_color_l = (250, 250, 250)
    start_pos_l = (40, 133)
    end_pos_l = (170, 133)
    pygame.draw.line(screen, arrow_color_l, start_pos_l, end_pos_l, 30)
    draw_arrowhead_l(screen, (250, 250, 250), (190, 133), (210, 133), 50)

    arrow_color_r = (250, 250, 250)
    start_pos_r = (620, 133)
    end_pos_r = (760, 133)
    pygame.draw.line(screen, arrow_color_r, start_pos_r, end_pos_r, 30)
    draw_arrowhead_r(screen, (250, 250, 250), (480, 133), (590, 133), 50)

    # Text color logic

    show_text(screen, "GO", (330, 250))
    show_text(screen, "REST", (295, 470))
    show_buttons_text(screen, "Start", (663, 3))
    show_buttons_text(screen, "Stop", (724, 3))

    pygame.display.flip()


def runPyGame(
    train_sequence=[0, 1, 2],
    on_start=lambda: print("start"),
    on_stop=lambda: print("stop"),
    on_left=lambda: print("left"),
    on_right=lambda: print("right"),
    on_go=lambda: print("go"),
    on_rest=lambda: print("rest"),
    rest_duration=20,
    work_duration=10,
):
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Data Collection UI")
    # clock = pygame.time.Clock()
    # fps = 60.0
    ctx = Context(
        train_sequence=train_sequence,
        work_duration=work_duration,
        rest_duration=rest_duration,
        on_left=on_left,
        on_right=on_right,
        on_go=on_go,
        on_rest=on_rest,
        on_start=on_start,
        on_stop=on_stop,
    )

    while True:
        # dt = clock.tick(fps) / 1000.0  # Convert milliseconds to seconds

        update(ctx.on_start)
        draw(screen, ctx)


if __name__ == "__main__":
    runPyGame()
