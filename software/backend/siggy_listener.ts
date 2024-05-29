import zmq from "zeromq";

import strftime from "strftime";
import { GameClient } from "./game";

enum Action {
  Rest = "rest",
  Clench = "clench",
  Left = "left",
  Right = "right",
}

const ActionMap = new Map([
  [0, Action.Rest],
  [1, Action.Clench],
  [2, Action.Left],
  [3, Action.Right],
]);

interface Prediction {
  time: number; // seconds since the epoch, see https://en.wikipedia.org/wiki/Unix_time
}

interface CategoricalPrediction extends Prediction {
  action: Action;
}

interface DistributionalPrecition extends Prediction {
  actionDistribution: number[];
}

enum MessageTopics {
  Categorical = "c",
  Distributional = "d",
}

class SiggyListener {
  private port = 3001;
  private siggySocket: zmq.Socket = zmq.socket("sub");
  private playerClients: Array<GameClient | null>;

  constructor(num_players: number, port: number = 3001, bind: boolean = false) {
    this.port = port;
    this.playerClients = Array<GameClient | null>(num_players);

    if (bind) {
      this.bind();
    }
  }

  public attachClient(player: number, client: GameClient | null) {
    this.playerClients[player] = client;
  }

  public bind() {
    try {
      this.siggySocket.bindSync(`tcp://*:${this.port}`);
      console.log(`siggy socket bound to tcp://*:${this.port}`);
    } catch (e) {
      console.log("unexpected error while binding:", e);
      return false;
    }
    this.siggySocket.subscribe(MessageTopics.Categorical);
    this.siggySocket.on("message", (_topic: Buffer, ...message: Buffer[]) => {
      const topic = _topic.toString();
      const category = topic[0];
      let bciIndex: number = -1;

      try {
        bciIndex = parseInt(topic.substring(1));
      } catch (error) {
        console.log("error parsing bci_index");
        return;
      }
      if (!this.playerClients[bciIndex]) return;

      switch (category) {
        case MessageTopics.Categorical: {
          let msg = this.parseCategoricalMessage(...message);
          if (!msg) break;
          this.playerClients[bciIndex]!.onCategoricalPrediction(msg);
          break;
        }
        case MessageTopics.Distributional: {
          let msg = this.parseDistributionalMessage(...message);
          if (!msg) break;
          this.playerClients[bciIndex]!.onDistributionalPrediction(msg);
          break;
        }
        default: {
          console.log("unrecognized category: ", category);
        }
      }
    });
  }

  public close() {
    if (this.siggySocket != null) {
      this.siggySocket.close();
      return true;
    }
    return false;
  }

  private parseCategoricalMessage(...message: Buffer[]) {
    const [time_str, action_str] = message.map((a) => a.toString());
    let time: number = 0;
    let action: Action = Action.Rest;

    try {
      time = parseFloat(time_str);
      action = ActionMap.get(parseInt(action_str))!;
      const msg: CategoricalPrediction = { time: time, action: action };
      return msg;
    } catch (e) {
      console.error("error parsing message:", e);
      return null;
    }
  }

  private parseDistributionalMessage(...message: Buffer[]) {
    return null;
  }
}

export { Action, CategoricalPrediction, SiggyListener };
