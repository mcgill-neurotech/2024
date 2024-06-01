from backend import DataClassifier
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
8 8888          8 8888 `8.`888b           ,8' 8 8888888888                 ,o888888o.    8 8888                  .8.            d888888o.      d888888o.    8 8888 8 8888888888    8 8888 8 8888888888   8 888888888o.   
8 8888          8 8888  `8.`888b         ,8'  8 8888                      8888     `88.  8 8888                 .888.         .`8888:' `88.  .`8888:' `88.  8 8888 8 8888          8 8888 8 8888         8 8888    `88.  
8 8888          8 8888   `8.`888b       ,8'   8 8888                   ,8 8888       `8. 8 8888                :88888.        8.`8888.   Y8  8.`8888.   Y8  8 8888 8 8888          8 8888 8 8888         8 8888     `88  
8 8888          8 8888    `8.`888b     ,8'    8 8888                   88 8888           8 8888               . `88888.       `8.`8888.      `8.`8888.      8 8888 8 8888          8 8888 8 8888         8 8888     ,88  
8 8888          8 8888     `8.`888b   ,8'     8 888888888888           88 8888           8 8888              .8. `88888.       `8.`8888.      `8.`8888.     8 8888 8 888888888888  8 8888 8 888888888888 8 8888.   ,88'  
8 8888          8 8888      `8.`888b ,8'      8 8888                   88 8888           8 8888             .8`8. `88888.       `8.`8888.      `8.`8888.    8 8888 8 8888          8 8888 8 8888         8 888888888P'   
8 8888          8 8888       `8.`888b8'       8 8888                   88 8888           8 8888            .8' `8. `88888.       `8.`8888.      `8.`8888.   8 8888 8 8888          8 8888 8 8888         8 8888`8b       
8 8888          8 8888        `8.`888'        8 8888                   `8 8888       .8' 8 8888           .8'   `8. `88888.  8b   `8.`8888. 8b   `8.`8888.  8 8888 8 8888          8 8888 8 8888         8 8888 `8b.     
8 8888          8 8888         `8.`8'         8 8888                      8888     ,88'  8 8888          .888888888. `88888. `8b.  ;8.`8888 `8b.  ;8.`8888  8 8888 8 8888          8 8888 8 8888         8 8888   `8b.   
8 888888888888  8 8888          `8.`          8 888888888888               `8888888P'    8 888888888888 .8'       `8. `88888. `Y8888P ,88P'  `Y8888P ,88P'  8 8888 8 8888          8 8888 8 888888888888 8 8888     `88. 
"""


def cli():
    # Create a data recorder
    data_recorder = DataClassifier(find_streams=True)
    print(title_string)

    while True:
        user_input = input(
            "\nPress 0 to exit. Press 2 to start recording. Press 3 to stop recording. Press 4 to connect to streams. Enter a command: \n > "
        )

        if user_input == "0":
            print("Exiting the data collection platform.")
            break

        elif user_input == "2":
            if not data_recorder.ready:
                print(
                    "EEG or marker stream not found. Please use option 4 to connect to streams first."
                )
                continue

            filename = "model.p"
            filename_input = input('Enter the model name to import (default "model.p"): ')
            if filename_input != "":
                filename = filename_input
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
