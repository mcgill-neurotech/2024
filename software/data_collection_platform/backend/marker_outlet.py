import pylsl
import logging
import constants


logger = logging.getLogger(__name__)


class MarkerOutlet:
    """This class creates and sends markers to an LSL stream."""

    def __init__(self):
        info = pylsl.StreamInfo(
            "Neurotech markers", "Markers", 1, 0, "string", "data-collection-markers"
        )
        self.outlet = pylsl.StreamOutlet(info)

    def send_marker_string(self, marker: str):
        """Send an arbitrary marker string. If you want to send a predefined marker,
        use the other methods such as send_cross, send_beep, send_left, and send_right.
        """

        self.outlet.push_sample([marker])
        logging.info(f"Sending marker: {marker}")

    def send_cross(self):
        self.send_marker_string(constants.markers[0])

    def send_beep(self):
        self.send_marker_string(constants.markers[1])

    def send_left(self):
        self.send_marker_string(constants.markers[2])

    def send_right(self):
        self.send_marker_string(constants.markers[3])
