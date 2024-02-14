#!/bin/bash

# Generate mock data streams for testing.
# One stream will send mock EEG, the other will send mock markers.


mock_eeg() {
    liesl mock --type EEG
}

mock_markers() {
    liesl mock --type Markers
}

(trap 'kill 0' SIGINT; mock_eeg & mock_markers)
