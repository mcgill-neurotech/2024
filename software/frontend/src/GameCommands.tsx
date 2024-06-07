import { Card, GameState } from "../../backend/game"
import { socket } from "./socket"

// data = array of current player's possible hand
socket.on("Possible Cards", (data) => {

}
)

// data = array of current player's impossible hand
socket.on("Impossible Cards", (data) => {

}
)

// data = string "right" or "left"
socket.on("direction", (data) => {

}
)

// data = current top card
socket.on("Card Played", (data) => {

}
)