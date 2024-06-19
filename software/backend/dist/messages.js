"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var GameEvents;
(function (GameEvents) {
    GameEvents["PlayerJoined"] = "player joined";
    GameEvents["PlayerReady"] = "player ready";
    GameEvents["GameStarted"] = "game started";
    GameEvents["Scroll"] = "scroll";
    GameEvents["PlayerPlayedCard"] = "player played card";
    GameEvents["PlayerPicksUpCardsMessage"] = "player picks up card";
    GameEvents["PlayerPossibleHand"] = "player possible hand";
})(GameEvents || (GameEvents = {}));
// interface PlayerUnoMessage {
//   playerIndex: number;
// }
