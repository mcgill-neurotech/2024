import { SiggyListener, CategoricalPrediction, Action } from "./siggy_listener";
import { Server, Socket } from "socket.io";

class GameClient {
  id: string;
  playerIndex: number = -1;
  socket: Socket;
  game: Game;
  currentPrediction: Action;

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
  players: Player[];
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
    this.players.push(new Player());

    return true;
  }

  // Shuffles deck
  public shuffleDeck() {
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
  public setGame() {
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
    // splice possible hand from 1 -> end, preserve draw card
    // clear everything in impossible hand 
    const topCard = this.gameState.top_card
    const color = topCard?.color
    const number = topCard?.number

    const player = this.players[playerIndex];
    const hand = player.hand;

    for (let i = 0; i < hand.length; i++) {
      if (
        hand[i].color == color ||
        hand[i].color == "wild" ||
        hand[i].number == number
      ) {
        player.possible_hand.push(hand[i]);
      } else {
        player.impossible_hand.push(hand[i]);
      }
    }
  }

  /* 
  Takes current playerIndex 
  Adds a card to the player */ 
  public addCard(playerIndex){
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
  public readTop(playerIndex: number) {
    const topCard = this.gameState.top_card 
    const number = topCard?.number 
    const opp = (playerIndex + 1) % 2

    if (number == 13 || number == 14) {
      // wild card -- could it be handled on front end? 
    } else if (number == 11){ //add 2 cards 
      this.addCard(opp); 
      this.addCard(opp); 
    } else if (number == 13){ //add 4 cards 
      for (let i=0; i < 4; i++) {
      this.addCard(opp);
      }
    };
    
    // Returns player index for the next round
    if (number == 10 || number == 11) {
      return playerIndex;
    }
    else {
      return opp;
    };
    
  }

  // Method to play Game -- will continue until one player has no cards
  public playGame() {
    this.setGame();
    let currentPlayerIndex = 0;

    while (this.players[currentPlayerIndex].hand.length > 0) {
      // Listening for move
      this.players[currentPlayerIndex].moveCard(this.clients[currentPlayerIndex], this.gameState);

      // After move is performed
      currentPlayerIndex = this.readTop(currentPlayerIndex) // Performs special functions and changes turn if applicable

      // Sends top_card to clients 
      this.broadcast("Card Played",this.gameState.top_card)
    }
  }
  */

  private changeTurn(currentPlayerIndex: number) {
    const topNum = this.gameState.top_card?.number
    if ((topNum != 10) && (topNum != 11)) {
      currentPlayerIndex = (currentPlayerIndex + 1) % 2
    }
    return currentPlayerIndex
  }

  public handleDisconnect(client: GameClient) {
    this.siggyListener.detachPlayer(client.playerIndex);
    this.clients.delete(client.id);
  }

  public broadcast(topic: string, ...msg: any[]) {
    // this.server.send()
  }
}

class GameState {
  deck: Card[] = [];
  played_cards: Card[] = [];
  top_card: Card | null = null;

  constructor() { //build initial game state 
    const colour = ['red', 'yellow', 'green', 'blue']
    /* 10 = skip; 11 = +2; 12 = +4; 13 = wildcard 14 = draw card */ 
    for (let i = 0; i < 14; i++){ //build the number cards 
      let joker_marker = false; 
      if (i > 9){
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

 public moveCard(playerClient: GameClient, gameState : GameState) {
  while (true) {
    const action = playerClient.getCurrentPrediction();
    if (action === Action.Right) {
        this.selected_card = (this.selected_card + 1) % this.possible_hand.length;
    } else if (action === Action.Left) {
        this.selected_card = (this.selected_card - 1 + this.possible_hand.length) % this.possible_hand.length;
    } else if (action === Action.Clench) { 
        this.playCard(gameState);
        break;
    }
  }
}

public playCard(gameState: GameState) {
  gameState.top_card = this.possible_hand[this.selected_card];
  this.possible_hand.splice(this.selected_card, 1);
  gameState.played_cards.push(this.possible_hand[this.selected_card]);
  this.hand.splice(this.selected_card, 1);
    
}
  
}

export { Game, GameClient, GameState, Card, Player };
