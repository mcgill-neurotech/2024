import pylsl
import time
import threading
import pandas as pd
import numpy as np
import typing


# g_markers = ["done", "cross", "beep", "left", "right"]
g_markers = ["done", "XXX", "Blah", "Marker", "Testtest"]  # Used with liesl
# NOTE: I (Ezra) couldn't find a way to manually set the mock marker values with liesl, so I just used the default ones.
# Maybe someone else can figure it out (https://github.com/pyreiz/pyliesl/blob/develop/liesl/streams/mock.py#L112)


def find_bci_inlet(debug=False):
    """Find an EEG stream and return an inlet to it.

    Args:
        debug (bool, optional): Print extra info. Defaults to False.

    Returns:
        pylsl.StreamInlet: Inlet to the EEG stream
    """

    print("Looking for an EEG stream...")
    streams = pylsl.resolve_stream("type", "EEG")
    # Block until stream found
    inlet = pylsl.StreamInlet(
        streams[0], processing_flags=pylsl.proc_dejitter | pylsl.proc_clocksync
    )

    print(
        f"Connected to stream: {streams[0].name()}, Stream channel_count: {streams[0].channel_count()}"
    )

    if debug:
        print(f"Stream info dump:\n{streams[0].as_xml()}")

    return inlet


def find_marker_inlet(debug=False):
    """Find a marker stream and return an inlet to it.

    Args:
        debug (bool, optional): Print extra info. Defaults to False.

    Returns:
        pylsl.StreamInlet: Inlet to the marker stream
    """

    print("Looking for a marker stream...")
    streams = pylsl.resolve_stream("type", "Marker")
    # Block until stream found
    inlet = pylsl.StreamInlet(
        streams[0], processing_flags=pylsl.proc_dejitter | pylsl.proc_clocksync
    )

    print(f"Found {len(streams)} streams")
    print(f"Connected to stream: {streams[0].name()}")

    if debug:
        print(f"Stream info dump:\n{streams[0].as_xml()}")

    return inlet


class CSVDataRecorder:
    """Class to record EEG and marker data to a CSV file."""

    def __init__(self, find_streams=True):
        self.eeg_inlet = find_bci_inlet() if find_streams else None
        self.marker_inlet = find_marker_inlet() if find_streams else None

        self.recording = False
        self.ready = self.eeg_inlet is not None and self.marker_inlet is not None

        print("DataRecorder ready:", self.ready)

    def find_streams(self):
        """Find EEG and marker streams. Updates the ready flag."""

        self.find_eeg_inlet()
        self.find_marker_input()
        self.ready = self.eeg_inlet is not None and self.marker_inlet is not None

    def find_eeg_inlet(self):
        """Find the EEG stream and update the inlet."""
        self.eeg_inlet = find_bci_inlet(debug=False)
        print("EEG Inlet found:", self.eeg_inlet)

    def find_marker_input(self):
        """Find the marker stream and update the inlet."""
        self.marker_inlet = find_marker_inlet(debug=False)
        print("Marker Inlet found:", self.marker_inlet)

        self.ready = self.eeg_inlet is not None and self.marker_inlet is not None

    def start(self, filename="test_data_0.csv"):
        """Start recording data to a CSV file. The recording will continue until stop() is called.
        The filename is the name of the file to save the data to. If the file already exists, it will be overwritten.
        If the LSL streams are not available, the function will print a message and return without starting the recording.
        Note that the output file will only be written to disk when the recording is stopped.
        """

        if not self.ready:
            print("Error: not ready to start recording")
            print("EEG Inlet:", self.eeg_inlet)
            print("Marker Inlet:", self.marker_inlet)
            return

        self.recording = True

        worker_args = [filename]
        t = threading.Thread(target=self._start_recording_worker, args=worker_args)
        t.start()

    def _start_recording_worker(self, filename):
        """Worker function to record the data to a CSV file.
        This function should not be called directly. Use start() instead.
        """

        # Flush the inlets to remove old data
        self.eeg_inlet.flush()
        self.marker_inlet.flush()

        df = pd.DataFrame(
            columns=[
                "timestamp",
                "ch1",
                "ch2",
                "ch3",
                "ch4",
                "ch5",
                "ch6",
                "ch7",
                "ch8",
                "cross",
                "beep",
                "left",
                "right",
            ]
        )

        timestamp_list = np.array([])
        channel_lists: typing.List[np.ndarray] = list()

        # see g_markers for the marker values
        cross_list = np.array([], dtype=int)
        beep_list = np.array([], dtype=int)
        left_list = np.array([], dtype=int)
        right_list = np.array([], dtype=int)

        current_marker = 0
        for i in range(8):
            channel_lists.append(np.array([]))

        while self.recording:
            # PROBLEM - we need to merge the two LSL streams into one
            # Assume we never get two markers for one sample
            # Therefore when we pull a marker, we can assume the next sample is the eeg sample and attach the marker to it
            # This effectively discards the marker timestamps but the EEG is recorded so quickly that it doesn't matter (?)
            # Otherwise all EEG samples will have an empty marker

            eeg_sample, eeg_timestamp = self.eeg_inlet.pull_sample()
            marker_sample, marker_timestamp = self.marker_inlet.pull_sample(0.0)

            if marker_sample is not None and marker_sample[0] is not None:
                marker = marker_sample[0]

                for i in range(len(g_markers)):
                    if marker == g_markers[i]:
                        current_marker = i
                        break

            timestamp_list = np.append(timestamp_list, eeg_timestamp)
            for i in range(8):
                channel_lists[i] = np.append(channel_lists[i], eeg_sample[i])

            cross_list = np.append(cross_list, 1 if current_marker == 1 else 0)
            beep_list = np.append(beep_list, 1 if current_marker == 2 else 0)
            left_list = np.append(left_list, 1 if current_marker == 3 else 0)
            right_list = np.append(right_list, 1 if current_marker == 4 else 0)

        df["timestamp"] = timestamp_list
        for i in range(8):
            df[f"ch{i+1}"] = channel_lists[i]
        df["cross"] = cross_list
        df["beep"] = beep_list
        df["left"] = left_list
        df["right"] = right_list

        df.to_csv(filename, index=False)

    def stop(self):
        """Finish recording data to a CSV file."""
        self.recording = False


def test_recorder():
    collector = CSVDataRecorder(find_streams=True)

    # mock collect for 2 seconds, then sleep for 1 second 5 times
    for i in range(5):
        print(f"Starting test run {i+1}")

        collector.start(filename=f"test_data_{i+1}.csv")
        time.sleep(2)
        collector.stop()
        time.sleep(1)


test_recorder()
