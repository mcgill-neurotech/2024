import { io } from 'socket.io-client';
import { Game, GameClient, GameState, Card, Player } from "../../backend/game.ts"

const BACKEND_URL = 'http://localhost:3000';

export const socket = io(BACKEND_URL);
