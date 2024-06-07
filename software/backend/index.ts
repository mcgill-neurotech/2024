import express from "express";
import http from "http";
import { Server } from "socket.io";
import { SiggyListener } from "./siggy_listener";
import { Game } from "./game";

const NUM_PLAYERS = 2;
const WS_PORT = 3000;
const SIGGY_PORT = 3001;
const FRONTEND_URL = "http://localhost:5173";
const DISCONNECT_TIMEOUT = 2 * 60 * 1000; // 2 minutes

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  connectionStateRecovery: {
    // the backup duration of the sessions and the packets
    maxDisconnectionDuration: DISCONNECT_TIMEOUT,
    // whether to skip middlewares upon successful recovery
    skipMiddlewares: true,
  },
  cors: {
    origin: FRONTEND_URL,
  },
});

const game = new Game(
  io,
  NUM_PLAYERS,
  new SiggyListener(NUM_PLAYERS, SIGGY_PORT, true),
);

io.on("connection", (socket) => {
  console.log("websocket connected", socket.id);

  if (socket.recovered) {
  } else {
    const created = game.createPlayer(socket);

    if (!created) {
      console.log("game full, disconnecting client");
      // socket.emit("") to maybe notify the client that the game is full?
      socket.disconnect(true);
    }
  }
});

server.listen(WS_PORT, () => {
  console.log(`websocket server listening on *:${WS_PORT}`);
});
