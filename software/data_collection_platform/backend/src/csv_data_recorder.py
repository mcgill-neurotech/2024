import pylsl
import time
import threading
import pandas as pd
import numpy as np
import typing
import logging
import os
import constants
from pathlib import Path

# Edited from NTX McGill 2021 stream.py, lines 16-23
# https://github.com/NTX-McGill/NeuroTechX-McGill-2021/blob/main/software/backend/dcp/bci/stream.py
log_path = Path(f"logs/test.log")
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)


def find_bci_inlet(debug=False):
    """Find an EEG stream and return an inlet to it.

    Args:
        debug (bool, optional): Print extra info. Defaults to False.

    Returns:
        pylsl.StreamInlet: Inlet to the EEG stream
    """

    logger.info("Looking for an EEG stream...")
    streams = pylsl.resolve_stream("type", "EEG")
    # Block until stream found
    inlet = pylsl.StreamInlet(
        streams[0], processing_flags=pylsl.proc_dejitter | pylsl.proc_clocksync
    )

    logger.info(
        f"Connected to stream: {streams[0].name()}, Stream channel_count: {streams[0].channel_count()}"
    )

    if debug:
        logger.info(f"Stream info dump:\n{streams[0].as_xml()}")

    return inlet


def find_marker_inlet(debug=False):
    """Find a marker stream and return an inlet to it.

    Args:
        debug (bool, optional): Print extra info. Defaults to False.

    Returns:
        pylsl.StreamInlet: Inlet to the marker stream
    """

    logger.info("Looking for a marker stream...")
    streams = pylsl.resolve_stream("type", "Markers")
    # Block until stream found
    inlet = pylsl.StreamInlet(
        streams[0], processing_flags=pylsl.proc_dejitter | pylsl.proc_clocksync
    )

    logger.info(f"Found {len(streams)} streams")
    logger.info(f"Connected to stream: {streams[0].name()}")

    if debug:
        logger.info(f"Stream info dump:\n{streams[0].as_xml()}")

    return inlet


class CSVDataRecorder:
    """Class to record EEG and marker data to a CSV file."""

    def __init__(self, find_streams=True):
        self.eeg_inlet = find_bci_inlet() if find_streams else None
        self.marker_inlet = find_marker_inlet() if find_streams else None

        self.recording = False
        self.ready = self.eeg_inlet is not None and self.marker_inlet is not None

        if self.ready:
            logger.info("Ready to start recording.")

    def find_streams(self):
        """Find EEG and marker streams. Updates the ready flag."""
        self.find_eeg_inlet()
        self.find_marker_input()
        self.ready = self.eeg_inlet is not None and self.marker_inlet is not None

    def find_eeg_inlet(self):
        """Find the EEG stream and update the inlet."""
        self.eeg_inlet = find_bci_inlet(debug=False)
        logger.info(f"EEG Inlet found:{self.eeg_inlet}")

    def find_marker_input(self):
        """Find the marker stream and update the inlet."""
        self.marker_inlet = find_marker_inlet(debug=False)
        logger.info(f"Marker Inlet found:{self.marker_inlet}")

        self.ready = self.eeg_inlet is not None and self.marker_inlet is not None

    def start(self, filename="test_data_0.csv"):
        """Start recording data to a CSV file. The recording will continue until stop() is called.
        The filename is the name of the file to save the data to. If the file already exists, it will be overwritten.
        If the LSL streams are not available, the function will print a message and return without starting the recording.
        Note that the output file will only be written to disk when the recording is stopped.
        """

        if not self.ready:
            logger.error("Error: not ready to start recording")
            logger.info(f"EEG Inlet:{self.eeg_inlet}")
            logger.info(f"Marker Inlet:{self.marker_inlet}")
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

        for i in range(8):
            channel_lists.append(np.array([]))

        while self.recording:
            # PROBLEM - we need to merge the two (EEG and Marker) LSL streams into one
            # Assume we never get two markers for one EEG sample
            # Therefore when we pull a marker, we can attach it to the next pulled EEG sample
            # This effectively discards the marker timestamps but the EEG is recorded so quickly that it doesn't matter (?)

            eeg_sample, eeg_timestamp = self.eeg_inlet.pull_sample()
            marker_sample, marker_timestamp = self.marker_inlet.pull_sample(0.0)

            marker = None
            if marker_sample is not None and marker_sample[0] is not None:
                marker_string = marker_sample[0]

                for i in range(len(constants.markers)):
                    if marker_string == constants.markers[i]:
                        marker = i
                        break

            timestamp_list = np.append(timestamp_list, eeg_timestamp)
            for i in range(8):
                channel_lists[i] = np.append(channel_lists[i], eeg_sample[i])

            cross_list = np.append(cross_list, 1 if marker == 0 else 0)
            beep_list = np.append(beep_list, 1 if marker == 1 else 0)
            left_list = np.append(left_list, 1 if marker == 2 else 0)
            right_list = np.append(right_list, 1 if marker == 3 else 0)

        df["timestamp"] = timestamp_list
        for i in range(8):
            df[f"ch{i+1}"] = channel_lists[i]
        df["cross"] = cross_list
        df["beep"] = beep_list
        df["left"] = left_list
        df["right"] = right_list

        filepath = Path(f"test_data/{filename}")
        filepath.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(filepath, index=False)

    def stop(self):
        """Finish recording data to a CSV file."""
        self.recording = False


def test_recorder():
    collector = CSVDataRecorder(find_streams=True)

    # mock collect for 3 seconds, then sleep for 1 second 5 times
    for i in range(5):
        print(f"Starting test run {i+1}")

        collector.start(filename=f"test_data_{i+1}.csv")
        time.sleep(3)
        collector.stop()
        print(f"Finished test run {i+1}")
        time.sleep(1)


# test_recorder()
