import pylsl
import time
import constants


def main():
    info = pylsl.StreamInfo(
        "Mock Markers", "Markers", 1, 0, "string", "data-collection-markers"
    )
    outlet = pylsl.StreamOutlet(info)
    current_marker = 0
    while True:
        marker_to_send = constants.markers[current_marker]
        print(f"Sending marker: {marker_to_send}")
        outlet.push_sample([marker_to_send])
        current_marker = (current_marker + 1) % len(constants.markers)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
