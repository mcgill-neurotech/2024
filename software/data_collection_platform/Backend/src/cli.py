import csv_data_recorder as recorder
import marker_outlet as m


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


def main():
    # Create a marker outlet
    marker_outlet = m.MarkerOutlet()

    # Create a data recorder
    data_recorder = recorder.CSVDataRecorder(find_streams=False)

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
                " > Select 1 to send a cross marker. Select 2 to send a beep marker. Select 3 to send a left marker. Select 4 to send a right marker. Select anything else to cancel: "
            )

            if marker == "1":
                marker_outlet.send_cross()
            elif marker == "2":
                marker_outlet.send_beep()
            elif marker == "3":
                marker_outlet.send_left()
            elif marker == "4":
                marker_outlet.send_right()
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
                "Data collection finished. Your data should be written in the test_data folder."
            )

        elif user_input == "4":
            print("Looking for streams...")
            data_recorder.find_streams()
            print("Found streams. Ready to start recording.")


if __name__ == "__main__":
    main()
