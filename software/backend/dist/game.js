"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Player = exports.Card = exports.GameState = exports.GameClient = exports.Game = void 0;
const siggy_listener_1 = require("./siggy_listener");
class GameClient {
    constructor(playerIndex, socket, game) {
        this.playerIndex = -1;
        this.playerIndex = playerIndex;
        this.socket = socket;
        this.id = socket.id;
        this.game = game;
        socket.on("disconnect", (reason) => {
            console.log(`client ${this.id} disconencted due to ${reason}`);
            this.onDisconnect();
        });
    }
    onDisconnect() {
        this.game.handleDisconnect(this);
    }
    onCategoricalPrediction(prediction) {
        console.log("onPredictedAction", this.id, prediction.action);
        this.currentPrediction = prediction.action;
    }
    onDistributionalPrediction(distribution) {
        console.log("onPredictedDistribution", this.id, distribution);
    }
    getCurrentPrediction() {
        return this.currentPrediction;
    }
    sendXMessage() {
        this.socket.emit("<event>", {});
    }
}
exports.GameClient = GameClient;
function sleep(ms) {
    return __awaiter(this, void 0, void 0, function* () {
        return yield new Promise((r) => setTimeout(() => r(), ms));
    });
}
class Game {
    constructor(server, numPlayers, siggyListener) {
        this.clients = new Map();
        this.players = [];
        this.gameState = new GameState();
        this.server = server;
        this.numPlayers = numPlayers;
        this.siggyListener = siggyListener;
    }
    getAvailablePlayers() {
        const available = Array(this.numPlayers).fill(true);
        this.clients.forEach((client) => {
            available[client.playerIndex] = false;
        });
        const availableIndices = [];
        available.forEach((v, i) => {
            if (v)
                availableIndices.push(i);
        });
        return availableIndices;
    }
    createPlayer(socket) {
        const availablePlayers = this.getAvailablePlayers();
        if (availablePlayers.length == 0)
            return false;
        const playerIndex = availablePlayers[0];
        const gameClient = new GameClient(playerIndex, socket, this);
        this.clients.set(socket.id, gameClient);
        this.siggyListener.attachPlayer(playerIndex, gameClient);
        // Ensure players array is properly initialized
        if (!this.players[playerIndex]) {
            this.players[playerIndex] = new Player(socket.id);
        }
        else {
            this.players[playerIndex].player_socket = socket.id;
        }
        const data = [];
        for (const v of this.clients.values()) {
            data.push({
                playerIndex: v.playerIndex,
                ready: this.players[playerIndex].ready,
            });
        }
        gameClient.socket
            .emitWithAck("Joined", data, this.numPlayers, playerIndex)
            .then(() => {
            this.broadcast("Player connection state update", playerIndex, true);
        });
        const remainingSlots = this.getAvailablePlayers().length;
        console.log(remainingSlots, "slots remaining");
        if (remainingSlots === 0) {
            this.startGame();
        }
        return true;
    }
    // Shuffles deck
    shuffleDeck() {
        let deck = this.gameState.deck;
        let currentIndex = deck.length, randomIndex;
        // While there remain elements to shuffle.
        while (currentIndex != 0) {
            // Pick a remaining element.
            randomIndex = Math.floor(Math.random() * currentIndex);
            currentIndex--;
            // And swap it with the current element.
            [deck[currentIndex], deck[randomIndex]] = [
                deck[randomIndex],
                deck[currentIndex],
            ];
        }
    }
    // Sets initial game state -- draw from deck option, 7 cards initially in each player's hands, first top card
    setGame() {
        this.shuffleDeck();
        const starthand = 7;
        let deck = this.gameState.deck;
        // Draws first card on playing deck
        const first_card = deck.pop();
        if (first_card) {
            this.gameState.top_card = first_card;
            this.gameState.played_cards.push(first_card);
            this.broadcast("Card Played", -1, first_card);
        }
        this.players.forEach((player, playerIndex) => {
            player.hand.push(new Card("wild", 14, true));
            for (let j = 0; j < starthand; j++) {
                this.addCard(playerIndex);
            }
            const client = this.clients.get(player.player_socket);
            if (client) {
                player.selected_card = Math.floor(player.hand.length / 2);
                client.socket.emit("Cards", player.hand);
                client.socket.emit("position", player.selected_card);
            }
            else {
                console.log("socket id does not correspond to player");
            }
        });
    }
    /*
    Takes current playerIndex
    Adds a card to the player */
    addCard(playerIndex) {
        const player = this.players[playerIndex];
        let deck = this.gameState.deck;
        if (deck.length != 0) {
            const card = deck.pop();
            if (card) {
                player.hand.push(card);
            }
        }
    }
    // After playing a card, checks if special power is used and changes turn accordingly
    readSpecial(playerIndex, selected) {
        /* 10 = skip; 11 = +2; 12 = +4; 13 = wildcard; 14 = draw card; 15 = solid color*/
        // handle all + cards
        if (selected.number === 12) {
            const victim = (playerIndex + 1) % this.numPlayers;
            for (let i = 0; i < 4; i++)
                this.addCard(victim);
        }
        else if (selected.number === 11) {
            const victim = (playerIndex + 1) % this.numPlayers;
            for (let i = 0; i < 2; i++)
                this.addCard(victim);
        }
        else if (selected.number === 14) {
            this.addCard(playerIndex);
        }
        // handle solid color / wild color swaps
        if (selected.number === 12 || selected.number === 13) {
            this.players[playerIndex].hand_copy = this.players[playerIndex].hand;
            this.players[playerIndex].hand = Card.Colors.map((color) => new Card(color, 15, true));
        }
        else if (selected.number === 15) {
            this.players[playerIndex].hand = this.players[playerIndex].hand_copy;
            this.players[playerIndex].hand_copy = [];
        }
        /* 13 = wildcard; 14 = draw card; 15 = solid color*/
        // handle direction changes
        if (selected.number <= 9 || selected.number >= 14) {
            // normal card
            return (playerIndex + 1) % this.numPlayers;
        }
        else if (selected.number === 10 || selected.number === 11) {
            // skip or +2, respectively
            return (playerIndex + 2) % this.numPlayers;
        }
        else if (selected.number === 12 || selected.number === 13) {
            // +4 or wild, same player (needs to choose the next color)
            return playerIndex;
        }
        else {
            this.error(`Unexpected number for readSpecial: ${selected.number}`);
            return (playerIndex + 1) % this.numPlayers;
        }
    }
    // Method to play Game -- will continue until one player has no cards
    playGame() {
        return __awaiter(this, void 0, void 0, function* () {
            console.log("playing game");
            this.broadcast("Game Started");
            this.setGame();
            let currentPlayerIndex = 0;
            yield sleep(2000);
            // Ensure currentPlayerIndex is valid
            if (!this.players[currentPlayerIndex]) {
                this.error(`Invalid currentPlayerIndex: ${currentPlayerIndex}`);
                return;
            }
            while (this.players[currentPlayerIndex].hand.length > 0) {
                const current_player = this.players[currentPlayerIndex];
                const current_client = this.clients.get(current_player.player_socket);
                if (current_client) {
                    // Listening for move
                    const selected = yield this.players[currentPlayerIndex].moveCard(current_client, this.gameState);
                    console.log("selected: %o for player %d", selected, currentPlayerIndex);
                    // Sends top_card to clients with playerIndex if the player didn't draw
                    if (selected.number !== 14) {
                        this.gameState.top_card = selected;
                        this.broadcast("Card Played", currentPlayerIndex, this.gameState.top_card);
                    }
                    // Performs special functions and changes turn if applicable
                    let prevCurrentPlayerIndex = currentPlayerIndex;
                    currentPlayerIndex = this.readSpecial(currentPlayerIndex, selected);
                    // Ensure currentPlayerIndex is valid after update
                    if (!this.players[currentPlayerIndex]) {
                        this.error(`Invalid currentPlayerIndex after update: ${currentPlayerIndex}`);
                        return;
                    }
                    this.emitToPlayer(prevCurrentPlayerIndex, "Cards", this.players[prevCurrentPlayerIndex].hand);
                    this.emitToPlayer(currentPlayerIndex, "Cards", this.players[currentPlayerIndex].hand);
                }
                else {
                    this.error("socket.id does not correspond to client");
                }
                yield sleep(2000);
            }
            this.endGame(currentPlayerIndex);
        });
    }
    emitToPlayer(playerIndex, topic, ...msg) {
        const player = this.players.at(playerIndex);
        if (!player) {
            this.error(`player at index ${playerIndex} does not exist`);
            return;
        }
        const client = this.clients.get(player.player_socket);
        if (!client) {
            this.error(`socket.id ${player.player_socket} does not correspond to client`);
            return;
        }
        client.socket.emit(topic, ...msg);
    }
    startGame() {
        return __awaiter(this, void 0, void 0, function* () {
            console.log("started game, waiting for clenches");
            if (this.players.length === this.numPlayers) {
                this.broadcast("Ready Listen");
                // all confirm readiness with jaw clench
                let ready = false;
                while (!ready) {
                    // console.log("not all ready");
                    ready = true;
                    for (const player of this.players) {
                        const client = this.clients.get(player.player_socket);
                        if (!!client) {
                            const [clientReady, updated] = player.checkReady(client);
                            ready = ready && clientReady;
                            if (updated) {
                                console.log("updated");
                                this.broadcast("Player ready state update", client.playerIndex, player.ready);
                            }
                        }
                    }
                    yield sleep(100);
                }
                yield this.playGame();
            }
        });
    }
    endGame(winnerIndex) {
        return __awaiter(this, void 0, void 0, function* () {
            this.broadcast("Game Ended", winnerIndex);
            // Timeout after 60000
            const timeoutID = setTimeout(this.closeGame, 60000);
            // P1 then P2 confirm readiness with jaw clench
            let ready = false;
            while (!ready) {
                console.log("not all ready");
                ready = true;
                for (const player of this.players) {
                    const client = this.clients.get(player.player_socket);
                    if (!!client) {
                        const [clientReady, updated] = player.checkReady(client);
                        ready = ready && clientReady;
                        if (updated) {
                            this.broadcast("Player ready state update", client.playerIndex, player.ready);
                        }
                    }
                }
                yield sleep(100);
            }
            clearTimeout(timeoutID);
            this.broadcast("Game Started");
            yield this.playGame();
        });
    }
    closeGame() {
        this.broadcast("Game Closed");
    }
    handleDisconnect(client) {
        this.siggyListener.detachPlayer(client.playerIndex);
        this.clients.delete(client.id);
        const player = this.players.at(client.playerIndex);
        if (player) {
            player.ready = false;
        }
        this.broadcast("Player connection state update", client.playerIndex);
    }
    broadcast(topic, ...msg) {
        // this.server.send()
        this.server.emit(topic, ...msg);
    }
    error(message) {
        console.log(`Error: ${message}`);
    }
}
exports.Game = Game;
class GameState {
    constructor() {
        this.deck = [];
        this.played_cards = [];
        this.top_card = null;
        //build initial game state
        /* 10 = skip; 11 = +2; 12 = +4; 13 = wildcard; 14 = draw card; 15 = solid color*/
        for (let i = 0; i < 14; i++) {
            //build the number cards
            // joker_marker is true for wild cards and solid color cards -- allows solid color to be placed on wild
            let joker_marker = false;
            if (i > 12) {
                joker_marker = true;
            }
            for (let j = 0; j < 4; j++) {
                if (i < 12) {
                    this.deck.push(new Card(Card.Colors[j], i, joker_marker));
                }
                else {
                    this.deck.push(new Card("wild", i, joker_marker));
                }
            }
        }
    }
}
exports.GameState = GameState;
class Card {
    constructor(color, number, joker = false) {
        this.color = "";
        this.number = 0;
        this.joker = false;
        this.color = color;
        this.number = number;
        this.joker = joker;
    }
    canBePlayedOn(card) {
        // "draw " card or wild cards
        if (this.number === 14)
            return true;
        // "normal" cards
        else if (card.number < 12) {
            return (this.number === 12 ||
                this.number === 13 ||
                this.number === card.number ||
                this.color === card.color);
        }
        // "wild" cards - will only be the case if drawn as the first starting card
        else if (card.number === 12 || card.number === 13) {
            return true;
        }
        // "solid color" cards
        else if (card.number === 15) {
            return this.color === card.color;
        }
    }
}
exports.Card = Card;
Card.Colors = ["red", "yellow", "green", "blue"];
class Player {
    constructor(player_socket) {
        this.ready = false;
        this.player_socket = "";
        this.hand = [];
        this.hand_copy = [];
        this.selected_card = 0;
        this.player_socket = player_socket;
        this.hand = [];
        this.selected_card = 0;
    }
    initialize(hand) {
        this.hand = hand;
    }
    checkReady(playerClient) {
        const prev = this.ready;
        this.ready = playerClient.getCurrentPrediction() === siggy_listener_1.Action.Clench;
        return [this.ready, prev !== this.ready];
    }
    moveCard(playerClient, gameState) {
        return __awaiter(this, void 0, void 0, function* () {
            while (true) {
                const action = playerClient.getCurrentPrediction();
                if (action === siggy_listener_1.Action.Clench) {
                    if (gameState.top_card &&
                        this.canPlaySelectedCardOn(gameState.top_card)) {
                        return this.playCard();
                    }
                    else {
                        // maybe create a new message for the client to notify them that try
                        // tried to make an illegal move
                        console.log("illegal card, skipping action");
                        yield sleep(2000);
                        continue;
                    }
                }
                if (action === siggy_listener_1.Action.Right) {
                    this.selected_card =
                        // (this.selected_card + 1) % this.possible_hand.length;
                        (this.selected_card + 1) % this.hand.length;
                }
                else if (action === siggy_listener_1.Action.Left) {
                    this.selected_card =
                        // (this.selected_card - 1 + this.possible_hand.length) %
                        // this.possible_hand.length;
                        (this.selected_card - 1 + this.hand.length) % this.hand.length;
                }
                console.log("emit position", this.selected_card);
                playerClient.socket.emit("position", this.selected_card);
                yield sleep(2000);
            }
        });
    }
    canPlaySelectedCardOn(topCard) {
        const card = this.hand.at(this.selected_card);
        if (!card) {
            console.log(`unexpected error trying to play card: index ${this.selected_card} out of bounds for hand of length ${this.hand}`);
            return false;
        }
        console.log(card, "can be played on", topCard, "?:", card.canBePlayedOn(topCard));
        return card.canBePlayedOn(topCard);
    }
    playCard() {
        const selected = this.hand[this.selected_card];
        if (selected.number !== 14) {
            this.hand.splice(this.selected_card, 1);
        }
        return selected;
    }
}
exports.Player = Player;
