"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.SiggyListener = exports.Action = void 0;
const zeromq_1 = __importDefault(require("zeromq"));
/**
 * The 4 possible actions that can be predicted, mapped into an enum
 */
var Action;
(function (Action) {
    Action["Rest"] = "rest";
    Action["Clench"] = "clench";
    Action["Left"] = "left";
    Action["Right"] = "right";
})(Action || (Action = {}));
exports.Action = Action;
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
 * Base message topics for the zmq SUB socket to subscribe to
 */
var MessageTopics;
(function (MessageTopics) {
    MessageTopics["Categorical"] = "c";
    MessageTopics["Distributional"] = "d";
})(MessageTopics || (MessageTopics = {}));
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
     * Creates an instance of SiggyListener.
     *
     * @constructor
     * @param num_players The maximum number of players in the game.
     * @param port The port to bind the zmq SUB socket to. Defaults to 3001
     * @param bind Whether or not to bind the zmq SUB socket on creation, Defaults to false.
     */
    constructor(num_players, port = 3001, bind = false) {
        /**
         * The port to bind the zmq PUB socket on
         */
        this.port = 3001;
        /**
         * The zmq socket that will recieve messages from
         */
        this.siggySocket = zeromq_1.default.socket("sub");
        this.port = port;
        this.gameClients = Array(num_players);
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
    attachPlayer(playerIndex, client) {
        this.gameClients[playerIndex] = client;
    }
    /**
     * Detach a client at a player index
     *
     * @param playerIndex The index at which to detach the player
     */
    detachPlayer(playerIndex) {
        this.gameClients[playerIndex] = null;
    }
    /**
     * Bind the zmq SUB socket
     *
     * @returns whether or not the zmq sucessfully bound
     */
    bind() {
        try {
            this.siggySocket.bindSync(`tcp://*:${this.port}`);
            console.log(`siggy socket bound to tcp://*:${this.port}`);
        }
        catch (e) {
            console.log("unexpected error while binding:", e);
            return false;
        }
        this.siggySocket.subscribe(MessageTopics.Categorical);
        this.siggySocket.on("message", (_topic, ...message) => {
            const topic = _topic.toString();
            const category = topic[0];
            let bciIndex = -1;
            try {
                bciIndex = parseInt(topic.substring(1));
            }
            catch (error) {
                console.log("error parsing bci_index");
                return;
            }
            if (!this.gameClients[bciIndex])
                return;
            switch (category) {
                case MessageTopics.Categorical: {
                    let msg = this.parseCategoricalMessage(...message);
                    if (!msg)
                        break;
                    this.gameClients[bciIndex].onCategoricalPrediction(msg);
                    break;
                }
                case MessageTopics.Distributional: {
                    let msg = this.parseDistributionalMessage(...message);
                    if (!msg)
                        break;
                    this.gameClients[bciIndex].onDistributionalPrediction(msg);
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
    parseCategoricalMessage(...message) {
        const [time_str, action_str] = message.map((a) => a.toString());
        let time = 0;
        let action = Action.Rest;
        try {
            time = parseFloat(time_str);
            action = ActionMap.get(parseInt(action_str));
            const msg = { time: time, action: action };
            return msg;
        }
        catch (e) {
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
    parseDistributionalMessage(...message) {
        return null;
    }
}
exports.SiggyListener = SiggyListener;
