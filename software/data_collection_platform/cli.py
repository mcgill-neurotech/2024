from backend import CSVDataRecorder
from backend import MarkerOutlet
import pathlib
import logging


log_path = pathlib.Path(f"logs/cli.log")
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=log_path,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

title_string = """

██████╗░░█████╗░████████╗░█████╗░  ░█████╗░░█████╗░██╗░░░░░██╗░░░░░███████╗░█████╗░████████╗██╗░█████╗░███╗░░██╗
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗  ██╔══██╗██╔══██╗██║░░░░░██║░░░░░██╔════╝██╔══██╗╚══██╔══╝██║██╔══██╗████╗░██║
██║░░██║███████║░░░██║░░░███████║  ██║░░╚═╝██║░░██║██║░░░░░██║░░░░░█████╗░░██║░░╚═╝░░░██║░░░██║██║░░██║██╔██╗██║
██║░░██║██╔══██║░░░██║░░░██╔══██║  ██║░░██╗██║░░██║██║░░░░░██║░░░░░██╔══╝░░██║░░██╗░░░██║░░░██║██║░░██║██║╚████║
██████╔╝██║░░██║░░░██║░░░██║░░██║  ╚█████╔╝╚█████╔╝███████╗███████╗███████╗╚█████╔╝░░░██║░░░██║╚█████╔╝██║░╚███║
╚═════╝░╚═╝░░╚═╝░░░╚═╝░░░╚═╝░░╚═╝  ░╚════╝░░╚════╝░╚══════╝╚══════╝╚══════╝░╚════╝░░░░╚═╝░░░╚═╝░╚════╝░╚═╝░░╚══╝

░█████╗░██╗░░░░░██╗
██╔══██╗██║░░░░░██║
██║░░╚═╝██║░░░░░██║
██║░░██╗██║░░░░░██║
╚█████╔╝███████╗██║
░╚════╝░╚══════╝╚═╝
"""


def cli():
    # Create a marker outlet
    marker_outlet = MarkerOutlet()

    # Create a data recorder
    data_recorder = CSVDataRecorder(find_streams=False)

    print(title_string)

    while True:
        user_input = input(
            "\nPress 0 to exit. Press 1 to send a marker. Press 2 to start recording. Press 3 to stop recording. Press 4 to connect to streams. Enter a command: \n > "
        )

        if user_input == "0":
            print("Exiting the data collection platform.")
            break

        elif user_input == "1":
            marker = input(
                " > Select 1 to send a cross marker. Select 2 to send a beep marker. Select 3 to send a left marker. Select 4 to send a right marker. Select 5 to send a clench marker. Select 6 to send a rest marker. Select anything else to cancel: "
            )

            if marker == "1":
                marker_outlet.send_cross()
            elif marker == "2":
                marker_outlet.send_beep()
            elif marker == "3":
                marker_outlet.send_left()
            elif marker == "4":
                marker_outlet.send_right()
            elif marker == "5":
                marker_outlet.send_clench()
            elif marker == "6":
                marker_outlet.send_rest()
            else:
                print("Invalid input.")
                continue

        elif user_input == "2":
            if not data_recorder.ready:
                print(
                    "EEG or marker stream not found. Please use option 4 to connect to streams first."
                )
                continue

            filename = input("Enter the filename for the recording: ")
            data_recorder.start(filename)
            print("Recording data...")

        elif user_input == "3":
            if not data_recorder.recording:
                print("No recording in progress.")
                continue

            data_recorder.stop()
            print(
                "Data collection finished. Your data should be written in the collected_data folder."
            )

        elif user_input == "4":
            print("Looking for streams...")
            data_recorder.find_streams()
            print("Found streams. Ready to start recording.")


if __name__ == "__main__":
    cli()
