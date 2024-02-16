import csv_data_recorder as recorder
import marker_outlet as m


def main():
    # Create a marker outlet
    marker_outlet = m.MarkerOutlet()

    # Create a data recorder
    data_recorder = recorder.CSVDataRecorder()

    print("Welcome to the data collection platform.")

    print(
        "Press 0 to exit. Press 1 to send a marker. Press 2 to start recording. Press 3 to stop recording. Press 4 to reconnect to streams. Press 5 to send a marker every 5 seconds."
    )

    while True:
        user_input = input("Enter a command: ")

        if user_input == "0":
            print("Exiting the data collection platform.")
            break

        elif user_input == "1":
            marker = input(
                "Press 1 to send a cross marker. Press 2 to send a beep marker. Press 3 to send a left marker. Press 4 to send a right marker."
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

        elif user_input == "2":
            filename = input("Enter the filename for the recording: ")
            data_recorder.start(filename)

        elif user_input == "3":
            data_recorder.stop()


if __name__ == "__main__":
    main()
