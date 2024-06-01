enum GameEvents {
  PlayerJoined = "player joined",
  PlayerReady = "player ready",
  GameStarted = "game started",
  Scroll = "scroll",
  PlayedPlayedCard = "played played card",
  PlayerPicksUpCardsMessage = "player picks up card",
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

// interface PlayerUnoMessage {
//   playerIndex: number;
// }
