import datetime
from backend import CSVDataRecorder
from backend import MarkerOutlet

from frontend_pygame.master_front_end import runPyGame
import numpy as np
import logging

import pathlib


collector = CSVDataRecorder(find_streams=False)
marker_outlet = MarkerOutlet()

log_path = pathlib.Path(f"logs/data_collection_platform.log")
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=log_path,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def on_start():
    collector.find_streams()

    if collector.ready:
        name = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
        collector.start(filename=f"{name}.csv")

    else:
        print("data not ready - quit and try again.")


def on_stop():
    collector.stop()


def on_left():
    marker_outlet.send_left()


def on_right():
    marker_outlet.send_right()


def on_go():
    marker_outlet.send_clench()


def on_rest():
    marker_outlet.send_rest()


np.random.seed(4)


def create_train_sequence():
    seq = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    np.random.shuffle(seq)
    return seq


def main():
    sequence = create_train_sequence()
    print("sequence: ", sequence)
    runPyGame(
        train_sequence=sequence,
        on_start=on_start,
        on_stop=on_stop,
        on_left=on_left,
        on_right=on_right,
        on_go=on_go,
        on_rest=on_rest,
        # on_go_done=on_go_done,
        # on_left_done=on_left_done,
        # on_right_done=on_right_done,
        # on_rest_done=on_rest_done,
        rest_duration=2,
        work_duration=1,
    )


if __name__ == "__main__":
    main()
