import time
import datetime
from backend import CSVDataRecorder
from backend import MarkerOutlet

from frontend.data_collection_frontend import runPyGame


collector = CSVDataRecorder(find_streams=False)
marker_outlet = MarkerOutlet()


def on_start():
    if collector.marker_inlet is None:
        print("Finding Marker Inlet...")
        collector.find_marker_input()

    if collector.eeg_inlet is None:
        print("Finding EEG Inlet...")
        collector.find_eeg_inlet()

    name = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    collector.start(filename=f"{name}.csv")


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


def on_left_done():
    marker_outlet.send_rest()


def on_right_done():
    marker_outlet.send_rest()


def on_go_done():
    marker_outlet.send_rest()


def on_rest_done():
    marker_outlet.send_rest()


def main():
    runPyGame(
        on_start=on_start,
        on_stop=on_stop,
        on_go=on_go,
        on_left=on_left,
        on_right=on_right,
        on_rest=on_rest,
        on_go_done=on_go_done,
        on_left_done=on_left_done,
        on_right_done=on_right_done,
        on_rest_done=on_rest_done,
    )


if __name__ == "__main__":
    main()
