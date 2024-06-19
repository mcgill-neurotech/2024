import zmq from "zeromq";

import strftime from "strftime";
import { GameClient } from "./game";

/**
 * The 4 possible actions that can be predicted, mapped into an enum
 */
enum Action {
  Rest = "rest",
  Clench = "clench",
  Left = "left",
  Right = "right",
}

/**
 * Used to map siggy messages to actions
 */
const ActionMap = new Map([
  [0, Action.Rest],
  [1, Action.Clench],
  [2, Action.Left],
  [3, Action.Right],
]);

/**
 * Generic prediction base interface. These predictions get passed to game clients.
 */
interface Prediction {
  /**
   * Time the prediction was made in UNIX seconds since the epoch, see {@link https://en.wikipedia.org/wiki/Unix_time}
   */
  time: number;
}

/**
 * A parsed categorical prediction message
 */
interface CategoricalPrediction extends Prediction {
  /**
   * The action predicted
   *
   * @type {Action}
   */
  action: Action;
}

/**
 * A parsed distributional prediction message
 */
interface DistributionalPrediction extends Prediction {
  /**
   * A length-4 array that represents a probability distribution over the 4 actions.
   * index 0 - rest, index 1 - clench, index 2 - left, index 3 - right
   */
  actionDistribution: number[];
}

/**
 * Base message topics for the zmq SUB socket to subscribe to
 */
enum MessageTopics {
  Categorical = "c",
  Distributional = "d",
}

/**
 * Class that is used to recieve zmq messages created from predictions from siggy
 * and forward relevant messages to their corresponding game clients.
 *
 * How it works:
 *
 * (1) The siggy listener binds a zmq SUB socket to a tcp port on the local machine,
 * subscribing to distributional and categorical messages (message prefixes of
 * 'c' or 'd', see the {@link MessageTopics} enum)
 *
 * (2) {@link GameClients} are attached with the {@link attachPlayer} method, which will then recieve relevant messages
 * of siggy predictions
 *
 * (3) When the zmq SUB socket recieves a message like ["c0", \<time\>, \<action\>], the message
 * will be parsed into a {@link Prediction}, in this case, a {@link CategoricalPrediction}, and forwarded to
 * the respective client if one is attached.
 */
class SiggyListener {
  /**
   * The port to bind the zmq PUB socket on
   */
  private port = 3001;
  /**
   * The zmq socket that will recieve messages from
   */
  private siggySocket: zmq.Socket = zmq.socket("sub");
  /**
   * A map of game clients, where the index of the array is used as a key that corresponds with the
   * player number of a player in the game
   */
  private gameClients: Array<GameClient | null>;

  /**
   * Creates an instance of SiggyListener.
   *
   * @constructor
   * @param num_players The maximum number of players in the game.
   * @param port The port to bind the zmq SUB socket to. Defaults to 3001
   * @param bind Whether or not to bind the zmq SUB socket on creation, Defaults to false.
   */
  constructor(num_players: number, port: number = 3001, bind: boolean = false) {
    this.port = port;
    this.gameClients = Array<GameClient | null>(num_players);

    if (bind) {
      this.bind();
    }
  }

  /**
   * Attach a game client to recieve siggy predictions
   *
   * @public
   * @param {number} playerIndex The player index to attach to. This value is
   * expected to be less than the total number of players that was specified
   * on creation.
   * @param {(GameClient | null)} client The client to attach to recieve predictions
   */
  public attachPlayer(playerIndex: number, client: GameClient) {
    this.gameClients[playerIndex] = client;
  }

  /**
   * Detach a client at a player index
   *
   * @param playerIndex The index at which to detach the player
   */
  public detachPlayer(playerIndex: number) {
    this.gameClients[playerIndex] = null;
  }

  /**
   * Bind the zmq SUB socket
   *
   * @returns whether or not the zmq sucessfully bound
   */
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
      if (!this.gameClients[bciIndex]) return;

      switch (category) {
        case MessageTopics.Categorical: {
          let msg = this.parseCategoricalMessage(...message);
          if (!msg) break;
          this.gameClients[bciIndex]!.onCategoricalPrediction(msg);
          break;
        }
        case MessageTopics.Distributional: {
          let msg = this.parseDistributionalMessage(...message);
          if (!msg) break;
          this.gameClients[bciIndex]!.onDistributionalPrediction(msg);
          break;
        }
        default: {
          console.log("unrecognized category: ", category);
        }
      }
    });
  }

  /**
   * Parse a recieved categorical prediction message
   *
   * @private
   * @param message The parts of the message recieved from zmq
   * @returns The parsed prediction, or null if the message parsing failed
   */
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

  /**
   * Parse a recieved distributional prediction message. Not implemented yet.
   *
   * @private
   * @param {...Buffer[]} message The parts of the message recieved from zmq
   * @returns {*}
   */
  private parseDistributionalMessage(...message: Buffer[]) {
    return null;
  }
}

export { Action, CategoricalPrediction, SiggyListener };
