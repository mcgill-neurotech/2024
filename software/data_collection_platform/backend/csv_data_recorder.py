import pylsl
import time
import threading
import pandas as pd
import numpy as np
import typing
import logging
from .constants import markers
from pathlib import Path
from einops import rearrange, reduce
from scipy.signal import filtfilt, iirnotch, butter
import pickle
import zmq
from .pre_processing import csp_preprocess
import os
import torch
from torch.nn import functional as F
import sys
import random
print("\n\n\n")
print(os.path.dirname(__file__))
print("\n\n\n")

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from siggy_ml.models.diffusion.classification_heads import EEGNetHead
# Edited from NTX McGill 2021 stream.py, lines 16-23
# https://github.com/NTX-McGill/NeuroTechX-McGill-2021/blob/main/software/backend/dcp/bci/stream.py
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def apply_notch(x, y, fs, notch_freq=50):
    """
    Apply pre-processing before concatenating everything in a single array.
    Easier to manage multiple splits
    By default, it only applies a notch filter at 50 Hz
    """

def apply_notch(x, y, fs, notch_freq=50):
    """
    Apply pre-processing before concatenating everything in a single array.
    Easier to manage multiple splits
    By default, it only applies a notch filter at 50 Hz
    """

    n, d, t = x.shape
    x = rearrange(x, "n t d -> (n d) t")
    b, a = iirnotch(notch_freq, 30, fs)
    x = filtfilt(b, a, x)
    x = rearrange(x, "(n d) t -> n t d", n=n)
    return x, y


def bandpass(
    x,
    fs,
    low,
    high,
):

    nyquist = fs / 2
    b, a = butter(4, [low / nyquist, high / nyquist], "bandpass", analog=False)
    n, d, t = x.shape
    x = rearrange(x, "n t d -> (n d) t")
    x = filtfilt(b, a, x)
    x = rearrange(x, "(n d) t -> n t d", n=n)
    return x


def epoch_preprocess(x, y, fs, notch_freq=50):

    x, y = apply_notch(x, y, fs, notch_freq)

    ax = []

    for i in range(1, 10):
        ax.append(bandpass(x, fs, 4 * i, 4 * i + 4))
    x = np.concatenate(ax, -1)

    mu = np.mean(x, axis=-1)
    sigma = np.std(x, axis=-1)
    x = (x - rearrange(mu, "n d -> n d 1")) / rearrange(sigma, "n d -> n d 1")
    return x, y

def clench_detect(x, y, fs, notch_freq=50):

    x, y = apply_notch(x, y, fs, notch_freq)

    ax = []

    for i in range(1, 10):
        ax.append(bandpass(x, fs, 4 * i, 4 * i + 4))
    x = np.concatenate(ax, -1)

    mu = np.mean(x, axis=-1)
    sigma = np.std(x, axis=-1)
    x = (x - rearrange(mu, "n d -> n d 1"))
    return reduce(x,"n d t -> ()","mean") > 20

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
                "clench",
                "rest",
            ]
        )

        timestamp_list = np.array([])
        channel_lists: typing.List[np.ndarray] = list()

        # see g_markers for the marker values
        cross_list = np.array([], dtype=np.bool_)
        beep_list = np.array([], dtype=np.bool_)
        left_list = np.array([], dtype=np.bool_)
        right_list = np.array([], dtype=np.bool_)
        clench_list = np.array([], dtype=np.bool_)
        rest_list = np.array([], dtype=np.bool_)

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

                for i in range(len(markers)):
                    if marker_string == markers[i]:
                        marker = i
                        break

            timestamp_list = np.append(timestamp_list, eeg_timestamp)
            for i in range(8):
                channel_lists[i] = np.append(channel_lists[i], eeg_sample[i])

            cross_list = np.append(cross_list, marker == 0)
            beep_list = np.append(beep_list, marker == 1)
            left_list = np.append(left_list, marker == 2)
            right_list = np.append(right_list, marker == 3)
            clench_list = np.append(clench_list, marker == 4)
            rest_list = np.append(rest_list, marker == 5)

        df["timestamp"] = timestamp_list
        for i in range(8):
            df[f"ch{i+1}"] = channel_lists[i]
        df["cross"] = cross_list.astype(np.int8)
        df["beep"] = beep_list.astype(np.int8)
        df["left"] = left_list.astype(np.int8)
        df["right"] = right_list.astype(np.int8)
        df["clench"] = clench_list.astype(np.int8)
        df["rest"] = rest_list.astype(np.int8)

        filepath = Path(f"collected_data/{filename}")
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


class DataClassifier:
    """Class to stream the last two seconds of LSL data"""

    def __init__(
        self,
        player,
        find_streams=True,
        use_eegnet=True,
    ):
        self.eeg_inlet = find_bci_inlet() if find_streams else None
        self.player = player
        print(f"Setting player {player}")
        self.recording = False
        self.ready = self.eeg_inlet is not None

        if self.ready:
            logger.info("Ready to start recording.")

        print("Ready to start recording")
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)

        self.connect_zmq()

        self.bufsize = 512
        self.window_length_sec = 512
        self.buffer = np.ndarray((8, self.bufsize))
        self.time_buffer = np.ndarray(self.bufsize)
        self.last_time_index = 0
        self.current_time_index = 0

        self.model = EEGNetHead(18,256)

        eegnet_path = os.path.join(os.path.dirname(__file__), '../eegnet_bands_openbci.pt')

        print("Loading EEGNet")
        self.model.load_state_dict(torch.load(eegnet_path,map_location=DEVICE))

        self.use_eegnet = use_eegnet

    def find_streams(self):
        """Find EEG and ZMQ socket endpoint. Updates the ready flag."""
        self.find_eeg_inlet()
        self.connect_zmq()
        self.ready = self.eeg_inlet is not None

    def find_eeg_inlet(self):
        """Find the EEG stream and update the inlet."""
        self.eeg_inlet = find_bci_inlet(debug=False)
        logger.info(f"EEG Inlet found:{self.eeg_inlet}")

    def connect_zmq(self):
        self.socket.connect("tcp://localhost:3001")
        print("zmq socket connected.")

    def send_categorical_prediction(self, time: float, action: int, player: int):
        topic = f"c{player}"
        self.socket.send_string(topic, zmq.SNDMORE)
        self.socket.send_string(f"{time}", zmq.SNDMORE)
        self.socket.send_string(f"{action}")

    def start(self, filename="test_data_0.csv"):
        """Start recording data to a CSV file. The recording will continue until stop() is called.
        The filename is the name of the file to save the data to. If the file already exists, it will be overwritten.
        If the LSL streams are not available, the function will print a message and return without starting the recording.
        Note that the output file will only be written to disk when the recording is stopped.
        """

        if not self.ready:
            logger.error("Error: not ready to start recording")
            logger.info(f"EEG Inlet:{self.eeg_inlet}")
            return

        self.recording = True

        worker_args = [filename]
        t = threading.Thread(target=self._start_recording_worker, args=worker_args)
        t.start()

    def _start_recording_worker(self, filename):
        # Flush the inlets to remove old data
        self.eeg_inlet.flush()

        count = 0

        model_path = os.path.join(os.path.dirname(__file__), '../model.p')

        if not self.use_eegnet:
            with open(model_path,"rb") as f:
                clf = pickle.load(f)

        while self.recording:
            eeg_sample, eeg_timestamp = self.eeg_inlet.pull_sample()

            two_seconds_before = eeg_timestamp - self.window_length_sec

            self.time_buffer[self.current_time_index] = eeg_timestamp
            self.buffer[:,self.current_time_index] = eeg_sample
            self.current_time_index = (self.current_time_index + 1) % self.bufsize

            while self.time_buffer[self.last_time_index] < two_seconds_before:
                self.last_time_index = (self.last_time_index + 1) % self.bufsize

            available_samples = self.eeg_inlet.samples_available()
            if available_samples > 16:
                # if we are buffering samples, skip prediction to allow fetching the latest samples
                # print(f"available samples: {available_samples} > 10, skipping prediction")
                continue

            x = self.buffer
            c,t = x.shape
            if t >= 512:
                if time.time() - last_send < 0.5:
                    continue

                else:
                    x = rearrange(x[np.array([3,5]),:],"c t -> 1 t c")
                    clench = clench_detect(x,_,256,60)
                    if clench:
                        self.send_categorical_prediction(time.time(),1,self.player)
                        last_send = time.time()
                        continue
                    x,_ = epoch_preprocess(x,None,256,60)
                    x = rearrange(x,"b t c ->b c t")
                    
                    if self.use_eegnet:
                        y = self.model.classify(torch.tensor(x).to(torch.float32))
                        y = F.softmax(y,-1)
                        # use argmax for categorical output
                        y = torch.argmax(y,-1)

                        self.send_categorical_prediction(time.time(),y[0]+1,self.player)
                        last_send = time.time()
                        continue
                    else:
                        y = clf.predict(x)

    def get_buffer_samples(self):
        if self.last_time_index > self.current_time_index:
            begin = self.buffer[self.last_time_index : self.bufsize]
            end = self.buffer[0 : self.current_time_index]
            return np.concatenate((begin, end), 0)
        else:
            return self.buffer[self.last_time_index : self.current_time_index]

    def add_prediction(self, pred_time, pred):
        self.prediction_buffer[self.pred_current_time_index] = [pred_time, pred]
        self.pred_current_time_index = (self.pred_current_time_index + 1) % self.bufsize

        while self.prediction_buffer[self.pred_last_time_index][0] < pred_time - 1:
            self.pred_last_time_index = (self.pred_last_time_index + 1) % self.bufsize

    def get_last_predictions(self):
        if self.pred_last_time_index > self.pred_current_time_index:
            begin = self.prediction_buffer[self.pred_last_time_index : self.bufsize]
            end = self.prediction_buffer[0 : self.pred_current_time_index]
            return np.concatenate((begin, end), 0)
        else:
            return self.prediction_buffer[
                self.pred_last_time_index : self.pred_current_time_index
            ]

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
