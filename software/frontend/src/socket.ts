import { io } from 'socket.io-client';

const BACKEND_URL = 'http://localhost:3000';

export const socket = io(BACKEND_URL);
