import { Card } from "./game"

enum GameEvents {
  PlayerJoined = "player joined",
  PlayerReady = "player ready",
  GameStarted = "game started",
  Scroll = "scroll",
  PlayerPlayedCard = "player played card",
  PlayerPicksUpCardsMessage = "player picks up card",
  PlayerPossibleHand = "player possible hand"
}

interface PlayerJoinedMessage {
  playerIndex: number;
} 

interface PlayerReadyMessage {
  playerIndex: number;
  ready: boolean;
}

interface GameStartedMessage {
  cards: string[];
}

interface ScrollMessage {
  direction: "left" | "right";
}

interface PlayerPlayedCardMessage {
  playerIndex: number;
  card: string;
  gameWon: boolean;
}

interface PlayerPicksUpCardsMessage {
  playerIndex: number;
}

interface PlayerPossibleHandMessage {
  playerIndex: number;
  possible_hand: Card[];
}

// interface PlayerUnoMessage {
//   playerIndex: number;
// }
