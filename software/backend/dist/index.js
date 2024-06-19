"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const http_1 = __importDefault(require("http"));
const socket_io_1 = require("socket.io");
const siggy_listener_1 = require("./siggy_listener");
const game_1 = require("./game");
const NUM_PLAYERS = 2;
const WS_PORT = 3000;
const SIGGY_PORT = 3001;
const FRONTEND_URL = "http://localhost:5173";
const DISCONNECT_TIMEOUT = 2 * 60 * 1000; // 2 minutes
const app = (0, express_1.default)();
const server = http_1.default.createServer(app);
const io = new socket_io_1.Server(server, {
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
const game = new game_1.Game(io, NUM_PLAYERS, new siggy_listener_1.SiggyListener(NUM_PLAYERS, SIGGY_PORT, true));
io.on("connection", (socket) => {
    console.log("websocket connected", socket.id);
    if (socket.recovered) {
    }
    else {
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
