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
        this.players.push(new Player(socket.id));
        const data = [];
        for (const v of this.clients.values()) {
            data.push({ playerIndex: v.playerIndex, ready: this.players[playerIndex].ready });
        }
        gameClient.socket.emitWithAck('Joined', data, this.numPlayers, playerIndex).then(() => {
            this.broadcast('Player connection state update', playerIndex, true);
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
    // Sets initial game state -- 7 cards initially in each player's hands, draw from deck option, first top card
    setGame() {
        this.shuffleDeck();
        const starthand = 7;
        let deck = this.gameState.deck;
        for (let i = 0; i < this.numPlayers; i++) {
            /* Add draw from deck card to both player's possible hand -- OR hand */
            this.players[i].possible_hand.push(new Card("wild", 14, true));
            for (let j = 0; j < starthand; j++) {
                const drawn_card = deck.pop();
                if (drawn_card) {
                    this.players[i].hand.push(drawn_card);
                }
            }
        }
        // Draws first card on playing deck
        const first_card = deck.pop();
        if (first_card) {
            this.gameState.top_card = first_card;
            this.gameState.played_cards.push(first_card);
        }
    }
    /*
    Takes current playerIndex
    Sorts cards in the current player's hand into possible or impossible hand
    */
    sortPossibleHand(playerIndex) {
        const topCard = this.gameState.top_card;
        const color = topCard === null || topCard === void 0 ? void 0 : topCard.color;
        const number = topCard === null || topCard === void 0 ? void 0 : topCard.number;
        const player = this.players[playerIndex];
        const hand = player.hand;
        let possible = player.possible_hand;
        let impossible = player.impossible_hand;
        // splice possible hand from 1 -> end, preserve draw card
        // clear everything in impossible hand
        possible.splice(1, possible.length - 1);
        impossible.length = 0;
        // Handles wild card color choice
        if (number == 12 || number == 13) {
            const colour = ['red', 'yellow', 'green', 'blue'];
            const solidnum = 15;
            // Pops draw card
            possible.pop();
            for (let i = 0; i < colour.length; i++) {
                possible.push(new Card(colour[i], solidnum, true));
            }
        }
        else {
            // Adds draw card if doesn't exist
            if (hand.length == 0) {
                possible.push(new Card("wild", 14, true));
            }
            for (let i = 0; i < hand.length; i++) {
                if (hand[i].color == color ||
                    hand[i].color == "wild" ||
                    hand[i].number == number) {
                    possible.push(hand[i]);
                }
                else {
                    impossible.push(hand[i]);
                }
            }
        }
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
        const number = selected.number;
        const opp = (playerIndex + 1) % 2;
        if (number == 11) { //add 2 cards 
            this.addCard(opp);
            this.addCard(opp);
        }
        else if (number == 12) { //add 4 cards 
            for (let i = 0; i < 4; i++) {
                this.addCard(opp);
            }
        }
        // Returns player index for the next round
        // Same player's turn if skip or +2 or allow player to choose wild card
        if (number) {
            if (number > 9 && number < 14) {
                return playerIndex;
            }
            else {
                return opp;
            }
        }
    }
    // Method to play Game -- will continue until one player has no cards
    playGame() {
        console.log('playing game');
        this.broadcast("Game Started");
        this.setGame();
        let currentPlayerIndex = 0;
        while (this.players[currentPlayerIndex].hand.length > 0) {
            console.log('while');
            // Calculate possible hand and send to specific client
            const current_player = this.players[currentPlayerIndex];
            const current_client = this.clients.get(current_player.player_socket);
            this.sortPossibleHand(currentPlayerIndex);
            if (current_client) {
                current_client.socket.emit("Possible Cards", this.players[currentPlayerIndex].possible_hand);
                current_client.socket.emit("Impossible Cards", this.players[currentPlayerIndex].impossible_hand);
                // Listening for move
                const selected = this.players[currentPlayerIndex].moveCard(current_client, this.gameState);
                // Special functions can be performed
                currentPlayerIndex = Number(this.readSpecial(currentPlayerIndex, selected)); // Performs special functions and changes turn if applicable
                // Sends top_card to clients with playerIndex
                this.broadcast("Card Played", currentPlayerIndex, this.gameState.top_card);
            }
            else {
                this.error("socket.id does not correspond to client");
            }
        }
        this.endGame(currentPlayerIndex);
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
                                console.log('updated');
                                this.broadcast("Player ready state update", client.playerIndex, player.ready);
                            }
                        }
                    }
                    yield new Promise((r) => setTimeout(r, 100)); //sleep for 100 ms
                }
                this.playGame();
            }
        });
    }
    endGame(winnerIndex) {
        return __awaiter(this, void 0, void 0, function* () {
            this.broadcast("Game Ended", winnerIndex);
            // Timeout after 60000
            const timeoutID = setTimeout(this.closeGame, 60000);
            let readyState = new Array(this.numPlayers).fill(false);
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
                yield new Promise((r) => setTimeout(r, 100)); //sleep for 100 ms
            }
            clearTimeout(timeoutID);
            this.broadcast("Game Started");
            this.playGame();
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
        this.broadcast('Player connection state update', client.playerIndex);
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
        const colour = ['red', 'yellow', 'green', 'blue'];
        /* 10 = skip; 11 = +2; 12 = +4; 13 = wildcard 14 = draw card 15 = solid color*/
        for (let i = 0; i < 14; i++) { //build the number cards 
            // joker_marker is true for wild cards and solid color cards -- allows solid color to be placed on wild
            let joker_marker = false;
            if (i > 12) {
                joker_marker = true;
            }
            for (let j = 0; j < 4; j++) {
                if (i < 12) {
                    this.deck.push(new Card(colour[j], i, joker_marker));
                }
                else {
                    this.deck.push(new Card('wild', i, joker_marker));
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
}
exports.Card = Card;
class Player {
    constructor(player_socket) {
        this.ready = false;
        this.player_socket = "";
        this.hand = [];
        this.possible_hand = [];
        this.selected_card = 0;
        this.impossible_hand = [];
        this.player_socket = player_socket;
        this.hand = [];
        this.possible_hand = [];
        this.selected_card = 0;
        this.impossible_hand = [];
    }
    checkReady(playerClient) {
        const prev = this.ready;
        this.ready = playerClient.getCurrentPrediction() === siggy_listener_1.Action.Clench;
        return [this.ready, prev !== this.ready];
    }
    moveCard(playerClient, gameState) {
        while (true) {
            const action = playerClient.getCurrentPrediction();
            if (action === siggy_listener_1.Action.Right) {
                this.selected_card = (this.selected_card + 1) % this.possible_hand.length;
                playerClient.socket.emit("direction", "right");
            }
            else if (action === siggy_listener_1.Action.Left) {
                this.selected_card = (this.selected_card - 1 + this.possible_hand.length) % this.possible_hand.length;
                playerClient.socket.emit("direction", "left");
            }
            else if (action === siggy_listener_1.Action.Clench) {
                return this.playCard(gameState, playerClient);
            }
        }
    }
    // Returns true if card played (new card placed onto played cards), false if no card played (draw card)
    playCard(gameState, playerClient) {
        const selected = this.possible_hand[this.selected_card];
        if (selected.number != 14) {
            gameState.top_card = selected;
            this.possible_hand.splice(this.selected_card, 1);
            gameState.played_cards.push(this.possible_hand[this.selected_card]);
            this.hand.splice(this.selected_card, 1);
        }
        else {
            const drawn = gameState.deck.pop();
            if (drawn) {
                this.hand.push(drawn);
            }
        }
        playerClient.socket.emit("Card Played", playerClient.playerIndex, selected);
        return selected;
    }
}
exports.Player = Player;
