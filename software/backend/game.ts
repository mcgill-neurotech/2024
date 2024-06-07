import { SiggyListener, CategoricalPrediction, Action } from "./siggy_listener";
import { Server, Socket } from "socket.io";
class GameClient {
  id: string;
  playerIndex: number = -1;
  socket: Socket;
  game: Game;
  currentPrediction: Action | undefined;

  constructor(playerIndex: number, socket: Socket, game: Game) {
    this.playerIndex = playerIndex;
    this.socket = socket;
    this.id = socket.id;
    this.game = game;

    socket.on("disconnect", (reason) => {
      console.log(`client ${this.id} disconencted due to ${reason}`);
      this.onDisconnect();
    });
  }

  public onDisconnect() {
    this.game.handleDisconnect(this);
  }

  public onCategoricalPrediction(prediction: CategoricalPrediction) {
    // console.log("onPredictedAction", this.id, prediction.action);
    this.currentPrediction = prediction.action;
  }

  public onDistributionalPrediction(distribution: number[]) {
    console.log("onPredictedDistribution", this.id, distribution);
  }

  public getCurrentPrediction() {
    return this.currentPrediction;
  }

  private sendXMessage() {
    this.socket.emit("<event>", {});
  }
  // ...
}

class Game {
  server: Server;
  clients = new Map<string, GameClient>();
  siggyListener: SiggyListener;
  numPlayers: number;
  players: Player[] = [];
  gameState: GameState = new GameState();

  constructor(
    server: Server,
    numPlayers: number,
    siggyListener: SiggyListener,
  ) {
    this.server = server;
    this.numPlayers = numPlayers;
    this.siggyListener = siggyListener;
  }

  public getAvailablePlayers() {
    const available = Array<boolean>(this.numPlayers).fill(true);
    this.clients.forEach((client) => {
      available[client.playerIndex] = false;
    });
    const availableIndices: number[] = [];
    available.forEach((v, i) => {
      if (v) availableIndices.push(i);
    });
    return availableIndices;
  }

  public createPlayer(socket: Socket) {
    const availablePlayers = this.getAvailablePlayers();
    if (availablePlayers.length == 0) return false;

    const playerIndex = availablePlayers[0];
    const gameClient = new GameClient(playerIndex, socket, this);
    this.clients.set(socket.id, gameClient);
    this.siggyListener.attachPlayer(playerIndex, gameClient);
    this.players.push(new Player(socket.id));

    return true;
  }

  // Shuffles deck
  private shuffleDeck() {
    let deck = this.gameState.deck;
    let currentIndex = deck.length,
      randomIndex;

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
  private setGame() {
    this.shuffleDeck()
    const starthand = 7;
    let deck = this.gameState.deck;
    for (let i = 0; i < this.numPlayers; i++) {
       /* Add draw from deck card to both player's possible hand -- OR hand */
      this.players[i].possible_hand.push(new Card("wild", 14, true))

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
  public sortPossibleHand(playerIndex: number) {
    const topCard = this.gameState.top_card
    const color = topCard?.color
    const number = topCard?.number

    const player = this.players[playerIndex];
    const hand = player.hand;
    let possible = player.possible_hand;
    let impossible = player.impossible_hand;
    
    // splice possible hand from 1 -> end, preserve draw card
    // clear everything in impossible hand 
    possible.splice(1,possible.length-1);
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
        if (
          hand[i].color == color ||
          hand[i].color == "wild" ||
          hand[i].number == number
        ) {
          possible.push(hand[i]);
        } else {
          impossible.push(hand[i]);
        }
      }
    }
  }

  /* 
  Takes current playerIndex 
  Adds a card to the player */ 
  public addCard(playerIndex: number){
    const player = this.players[playerIndex]; 
    let deck = this.gameState.deck; 
    if (deck.length != 0){
      const card = deck.pop(); 
      if (card){
        player.hand.push(card); 
      }
    }
  }

  // After playing a card, checks if special power is used and changes turn accordingly
  public readSpecial(playerIndex: number, selected: Card) {
    const number = selected.number 
    const opp = (playerIndex + 1) % 2

    if (number == 11){ //add 2 cards 
      this.addCard(opp); 
      this.addCard(opp); 
    } else if (number == 12){ //add 4 cards 
      for (let i=0; i < 4; i++) {
        this.addCard(opp);
      }
    };
    
    // Returns player index for the next round
    // Same player's turn if skip or +2 or allow player to choose wild card
    if (number) {
      if (number > 9 && number < 14) {
        return playerIndex;
      }
      else {
        return opp;
      };
    };
    
  }

  // Method to play Game -- will continue until one player has no cards
  public playGame() {
    this.setGame();
    let currentPlayerIndex = 0;
    
    while (this.players[currentPlayerIndex].hand.length > 0) {
      // Calculate possible hand and send to specific client
      const current_player = this.players[currentPlayerIndex]
      const current_client = this.clients.get(current_player.player_socket)
      this.sortPossibleHand(currentPlayerIndex);

      if (current_client) {
        current_client.socket.emit("Possible Cards", current_player.possible_hand);
        current_client.socket.emit("Impossible Cards", current_player.impossible_hand);

        // Listening for move
        const selected = this.players[currentPlayerIndex].moveCard(current_client, this.gameState);

        if (current_player.hand.length == 0) {
          break;
        }

        // Special functions can be performed
        currentPlayerIndex = Number(this.readSpecial(currentPlayerIndex, selected)) // Performs special functions and changes turn if applicable
    
        // Sends top_card to clients 
        this.broadcast("Card Played", this.gameState.top_card);
      }
      else {
        this.error("socket.id does not correspond to client");
      }
    }

    this.endGame(currentPlayerIndex);
  }

  public startGame() {
    let readyState = new Array(this.numPlayers).fill(false);

    // P1 then P2 confirm readiness with jaw clench
    for (let i = 0; i < this.numPlayers; i++) {
      const client = this.clients.get(this.players[i].player_socket)
      if (client) {
        readyState[i] = this.players[i].readyAction(client);
      }
    }

    this.broadcast("Game Started", true);

    this.playGame();
  }

  public endGame(winnerIndex: number) {
    this.broadcast("Game Ended", winnerIndex);
    // Timeout after 60000
    const timeoutID = setTimeout(this.closeGame, 60000);

    let readyState = new Array(this.numPlayers).fill(false);

    // P1 then P2 confirm readiness with jaw clench
    for (let i = 0; i < this.numPlayers; i++) {
      const client = this.clients.get(this.players[i].player_socket)
      if (client) {
        readyState[i] = this.players[i].readyAction(client);
      }
    }
    clearTimeout(timeoutID);

    this.broadcast("Game Started", true);

    this.playGame();
  }

  private closeGame() {
    this.broadcast("Game Closed", true);
  }

  public handleDisconnect(client: GameClient) {
    this.siggyListener.detachPlayer(client.playerIndex);
    this.clients.delete(client.id);
  }

  public broadcast(topic: string, ...msg: any[]) {
    // this.server.send()
    this.server.emit(topic, ...msg);
  }
  
  private error(message: string) {
    console.log(`Error: ${message}`);
  }
}

class GameState {
  deck: Card[] = [];
  played_cards: Card[] = [];
  top_card: Card | null = null;

  constructor() { //build initial game state 
    const colour = ['red', 'yellow', 'green', 'blue']
    /* 10 = skip; 11 = +2; 12 = +4; 13 = wildcard 14 = draw card 15 = solid color*/ 
    for (let i = 0; i < 14; i++){ //build the number cards 
      // joker_marker is true for wild cards and solid color cards -- allows solid color to be placed on wild
      let joker_marker = false; 
      if (i > 12){
        joker_marker = true; 
      } 
      for (let j = 0; j < 4; j++){
        if (i < 12){ 
          this.deck.push(new Card(colour[j], i, joker_marker)); 
        } else { 
          this.deck.push(new Card('wild', i, joker_marker)); 
        }
      }
    }
  }
}

class Card {
  color: string = "";
  number: number = 0;
  joker: boolean = false;

  constructor(color: string, number: number, joker: boolean = false) {
    this.color = color;
    this.number = number;
    this.joker = joker;
  }
}

class Player {
  player_socket: string = "";
  hand: Card[] = [];
  possible_hand: Card[] = [];
  selected_card: number = 0;
  impossible_hand: Card[] = [];

  constructor(player_socket: string) {
    this.player_socket = player_socket;
    this.hand = [];
    this.possible_hand = [];
    this.selected_card = 0;
    this.impossible_hand = [];
  }

  public readyAction(playerClient: GameClient) {
    while (true) {
      const action = playerClient.getCurrentPrediction();
      if (action === Action.Clench) {
        return true;
      }
    }
  }

  public moveCard(playerClient: GameClient, gameState : GameState) {
    while (true) {
      const action = playerClient.getCurrentPrediction();
      if (action === Action.Right) {
          this.selected_card = (this.selected_card + 1) % this.possible_hand.length;
          playerClient.socket.emit("direction", "right");
      } else if (action === Action.Left) {
          this.selected_card = (this.selected_card - 1 + this.possible_hand.length) % this.possible_hand.length;
          playerClient.socket.emit("direction", "left");
      } else if (action === Action.Clench) { 
         return this.playCard(gameState);
      }
    }
  }

  // Returns true if card played (new card placed onto played cards), false if no card played (draw card)
  public playCard(gameState: GameState) {
    const selected = this.possible_hand[this.selected_card];
    if (selected.number != 14) {
      gameState.top_card = selected;
      this.possible_hand.splice(this.selected_card, 1);
      gameState.played_cards.push(this.possible_hand[this.selected_card]);
      this.hand.splice(this.selected_card, 1);
    }
    else {
      const drawn = gameState.deck.pop()
      if (drawn) {
        this.hand.push(drawn);
      }
    }
    return selected;
  }
}

export { Game, GameClient, GameState, Card, Player };
