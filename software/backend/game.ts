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

    return true;
  }

  public startGame(socket: Socket, clients) {
    
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

export { Game, GameClient };
