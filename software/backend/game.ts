import { SiggyListener, CategoricalPrediction } from "./siggy_listener";
import { Socket } from "socket.io";

class GameClient {
  id: string;
  playerIndex: number = -1;
  socket: Socket;
  game: Game;

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
    console.log("onPredictedAction", this.id, prediction.action);
  }

  public onDistributionalPrediction(distribution: number[]) {
    console.log("onPredictedDistribution", this.id, distribution);
  }

  private sendXMessage() {
    this.socket.emit("<event>", {});
  }
  // ...
}

class Game {
  clients = new Map<string, GameClient>();
  siggyListener: SiggyListener;
  numPlayers: number;
  players: Player[];
  gameState: GameState = new GameState();

  constructor(numPlayers: number, siggyListener: SiggyListener) {
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
    this.players.push(new Player());

    return true;
  }

  // Shuffles deck
  public shuffleDeck() {
    let deck = this.gameState.deck;
    let currentIndex = deck.length, randomIndex;

    // While there remain elements to shuffle.
    while (currentIndex != 0) {
  
      // Pick a remaining element.
      randomIndex = Math.floor(Math.random() * currentIndex);
      currentIndex--;
  
      // And swap it with the current element.
      [deck[currentIndex], deck[randomIndex]] = [
        deck[randomIndex], deck[currentIndex]];
    }
  }

  // Deals initial player hands, adds draw card option to possible hands, and puts down initial top card
  public setGame() {
    const starthand = 7;
    let deck = this.gameState.deck
    for (let i = 0; i < this.numPlayers; i++) {
      for (let j = 0; j < starthand; j++) {
        const drawn_card = deck.pop()
        if (drawn_card) {
          this.players[i].hand.push(drawn_card)
        }
        /* Add draw card to both player's possible hands */
      } 
    }
    // Draws first card on playing deck
    const first_card = deck.pop()
    if (first_card) {
      this.gameState.played_cards.push(first_card)
    }
  }

  /* 
  Takes current playerIndex
  Sorts cards in the current player's hand into possible or impossible hand
  */
  public sortPossibleHand(playerIndex) {
    const topCard = this.gameState.top_card
    const color = topCard?.color
    const number = topCard?.number

    const player = this.players[playerIndex]
    const hand = player.hand

    for (let i = 0; i < hand.length; i++) {
      if ((hand[i].color == color || hand[i].color == "wild") || (hand[i].number == number)) {
        player.possible_hand.push(hand[i]);
      }
      else {
        player.impossible_hand.push(hand[i]);
      }
    }
  }

  public handleDisconnect(client: GameClient) {
    this.siggyListener.detachPlayer(client.playerIndex);
    this.clients.delete(client.id);
  }

  public broadcast(topic: string, ...msg: any[]) {
    this.clients.forEach((c) => {
      c.socket.emit(topic, ...msg);
    });
  }
}

class GameState {
  deck: Card[] = [];
  played_cards: Card[] = [];
  top_card: Card | null = null;

  constructor() { //build initial game state 
    this.top_card = this.played_cards[0] || null;
    const colour = ['red', 'yellow', 'green', 'blue']
    /* 10 = skip; 11 = +2; 12 = +4; 13 = wildcard */ 
    for (let i= 0; i<14; i++){ //build the number cards 
      let joker_marker = false; 
      if (i > 9){
        joker_marker = true; 
      } 
      for (let j=0; j<4; j++){
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
  hand: Card[] = [];
  possible_hand: Card[] = [];
  selected_card: number = 0;
  impossible_hand: Card[] = [];

  constructor() {
    this.hand = [];
    this.possible_hand = [];
    this.selected_card = 0;
    this.impossible_hand = [];
  }
}

export { Game, GameClient, GameState, Card, Player };
