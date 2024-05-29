import { SiggyListener, CategoricalPrediction } from "./siggy_listener";
import { Socket } from "socket.io";

class GameClient {
  id: string;
  socket: Socket;
  playerIndex: number = -1;

  constructor(socket: Socket) {
    this.id = socket.id;
    this.socket = socket;
  }

  // siggy predictions
  public onCategoricalPrediction(prediction: CategoricalPrediction) {
    console.log(this.id, prediction.action);
  }

  public onDistributionalPrediction(distribution: number[]) {
    console.log("onPredictedDistribution");
  }
}

class Game {
  clients = new Map<string, GameClient>();
  siggy_listener: SiggyListener;
  numPlayers: number;
  turn: number = 0;

  constructor(num_players: number, siggyListener: SiggyListener) {
    this.numPlayers = num_players;
    this.siggy_listener = siggyListener;
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

  public addPlayer = (client: GameClient) => {
    const availablePlayers = this.getAvailablePlayers();
    if (availablePlayers.length == 0) return false;

    const player = availablePlayers[0];
    client.playerIndex = player;
    this.clients.set(client.id, client);
    this.siggy_listener.attachClient(player, client);

    client.socket.on("disconnect", (reason) => {
      console.log(`client ${client.id} disconencted due to ${reason}`);
      this.onClientDisconnect(client);
    });

    return true;
  };

  public onClientDisconnect = (client: GameClient) => {
    if (!this.clients.has(client.id)) return;
    const player = client.playerIndex;
    this.siggy_listener.attachClient(player, null);
    this.clients.delete(client.id);
  };
}

export { Game, GameClient };
