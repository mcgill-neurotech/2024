from OpenBCI_LSL.lib.open_bci_v3 import OpenBCIBoard
import threading
import time


# couldn't connect to OpenBCIBoard on wsl2, maybe someone with a mac can test?
class BciStreamer:
    def __init__(self, port=None):
        # find board by default
        if port is None:
            self.board = OpenBCIBoard()
        else:
            self.board = OpenBCIBoard(port=port)

    def start_streaming(self, on_sample):
        print("Streaming started.\n")
        boardThread = threading.Thread(
            target=self.board.start_streaming, args=(on_sample, -1)
        )
        boardThread.daemon = True  # will stop on exit
        boardThread.start()

    def stop_streaming(self):
        self.board.stop()

        # clean up any leftover bytes from serial port
        # self.board.ser.reset_input_buffer()
        time.sleep(0.1)
        line = ""
        while self.board.ser.inWaiting():
            # print("doing this thing")
            c = self.board.ser.read().decode("utf-8", errors="replace")
            line += c
            time.sleep(0.001)
            if c == "\n":
                line = ""
        print("Streaming paused.\n")
