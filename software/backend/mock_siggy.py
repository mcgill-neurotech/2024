import argparse
import time
import zmq
import random
import numpy as np

# See https://zguide.zeromq.org/docs/chapter2/#Pub-Sub-Message-Envelopes


def softmax(arr: np.ndarray):
    return np.exp(arr) / np.exp(arr).sum()


# it is possible to use something like protocol buffers later once a schema is better defined, but this is good enough for now
def mock_categorical(socket: zmq.Socket, player: int):
    if player == 1:
        topic = "c0"
    else:
        topic = "c1"
    while True:
        action = input(f"(player {player}) select an action: ")
        t = time.time()
        # action = random.randint(0, 3)
        # 0 -> nothing, 1 -> jaw clench, 2 -> left, 3 -> right
        socket.send_string(topic, zmq.SNDMORE)
        socket.send_string(f"{t}", zmq.SNDMORE)
        socket.send_string(f"{action}")
        print("sent")


def mock_distributions(socket: zmq.Socket, player: int):
    if player == 1:
        topic = "d0"
    else:
        topic = "d1"
    while True:
        t = time.time()
        distribution: np = softmax(np.random.randn(4))
        socket.send_string(topic, zmq.SNDMORE)
        socket.send(f"{t}", zmq.SNDMORE)
        socket.send(distribution[0], zmq.SNDMORE)
        socket.send(distribution[1], zmq.SNDMORE)
        socket.send(distribution[2], zmq.SNDMORE)
        socket.send(distribution[3])
        time.sleep(0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributions", action="store_true")
    parser.add_argument("--player", default=1, type=int)
    args = parser.parse_args()

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    print("Connecting to server...")
    socket.connect("tcp://localhost:3001")

    if args.distributions:
        mock_distributions(socket, args.player)
    else:
        mock_categorical(socket, args.player)


if __name__ == "__main__":
    main()
