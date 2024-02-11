# Description: This script is used to test the writing of the EEG data to a csv file.
# The data is written to a csv file called test.csv. The data is written in the following format:
# timestamp, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8
# The timestamp is the time at which the data was recorded and the ch1 - ch8 are the 8 channels of the EEG data.

from pylsl import StreamInlet, resolve_stream
import pandas as pd
import numpy as np


def main():
    print("looking for an EEG stream...")
    streams = resolve_stream("type", "EEG")

    inlet = StreamInlet(streams[0])
    df = pd.DataFrame(
        columns=["timestamp", "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8"]
    )

    timestamp_list = np.array([])
    channel_lists = list()

    for i in range(8):
        channel_lists.append(np.array([]))
    for i in range(1000):
        sample, timestamp = inlet.pull_sample()
        timestamp_list = np.append(timestamp_list, timestamp)
        for i in range(8):
            channel_lists[i] = np.append(channel_lists[i], sample[i])

    df["timestamp"] = timestamp_list

    for i in range(8):
        df[f"ch{i+1}"] = channel_lists[i]

    df.to_csv("test.csv", index=False)


if __name__ == "__main__":
    main()
