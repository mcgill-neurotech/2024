// src/useSocket.ts
import { useEffect, useState } from 'react';
import { socket } from './socket';
import { Card as GameCard } from '../../backend/src/game';
import { useNavigate } from 'react-router-dom';

interface Player {
  connected: boolean;
  ready: boolean;
}

interface JoinInfo {
  playerIndex: number;
  ready: boolean;
}

interface ConnectionInfo {
  isConnected: boolean;
  id: string | undefined;
}

interface IUseSocketParams {
  // data = array of current player's possible hand
  onPossibleCards: (data: GameCard[]) => void;

  // data = array of current player's impossible hand
  onImpossibleCards: (data: GameCard[]) => void;

  // data = index
  onPosition: (position: number) => void;

  // data = total number of players on the game
  onJoined: (
    info: JoinInfo[],
    maxPlayers: number,
    playerIndex: number,
    cb: VoidFunction,
  ) => void;

  // data = index of player to update and if they connected/disconnected
  onPlayerConnectionStateUpdate: (
    playerIndex: number,
    connected: boolean,
  ) => void;

  // data = index of player whose ready state was updated, and their new ready state
  onPlayerReadyStateUpdate: (playerIndex: number, ready: boolean) => void;

  // data = current top card
  onCardPlayed: (payerIndex: number, data: GameCard) => void;

  // broadcast when both players are ready
  onGameStarted: () => void;

  // data = index of winning player
  onGameEnded: (data: number) => void;

  // broadcast when players don't restart game
  onGameClosed: () => void;
}

const useSocket = ({
  onJoined,
  onPlayerReadyStateUpdate,
  onPlayerConnectionStateUpdate,
  onPossibleCards,
  onImpossibleCards,
  onPosition,
  onCardPlayed,
  onGameStarted,
  onGameEnded,
  onGameClosed,
}: IUseSocketParams): ConnectionInfo => {
  const [isConnected, setIsConnected] = useState(socket.connected);
  const [id, setId] = useState(socket.id);

  useEffect(() => {
    function onConnect() {
      setIsConnected(true);
      setId(socket.id);
    }

    function onDisconnect() {
      setIsConnected(false);
      setId('');
    }

    socket.on('connect', onConnect);
    socket.on('Joined', onJoined);
    socket.on('Player ready state update', onPlayerReadyStateUpdate);
    socket.on('Player connection state update', onPlayerConnectionStateUpdate);
    socket.on('disconnect', onDisconnect);
    socket.on('Possible Cards', onPossibleCards);
    socket.on('Impossible Cards', onImpossibleCards);
    socket.on('position', onPosition);
    socket.on('Card Played', onCardPlayed);
    socket.on('Game Started', onGameStarted);
    socket.on('Game Ended', onGameEnded);
    socket.on('Game Closed', onGameClosed);

    return () => {
      socket.off('connect', onConnect);
      socket.off('Joined', onJoined);
      socket.off('Player ready state update', onPlayerReadyStateUpdate);
      socket.off(
        'Player connection state update',
        onPlayerConnectionStateUpdate,
      );
      socket.off('disconnect', onDisconnect);
      socket.off('Possible Cards', onPossibleCards);
      socket.off('Impossible Cards', onImpossibleCards);
      socket.off('position', onPosition);
      socket.off('Card Played', onCardPlayed);
      socket.off('Game Started', onGameStarted);
      socket.off('Game Ended', onGameEnded);
      socket.off('Game Closed', onGameClosed);
    };
  }, []);

  return { isConnected, id };
};

interface GameInfo {
  players: Player[];
  playerIndex: number;
  playedCards: GameCard[];
  playableCards: GameCard[];
  unplayableCards: GameCard[];
  selectedPlayableCardIndex: number;
  selectedUnplayableCardIndex: number;
}

export interface useGameSocketReturn {
  connectionInfo: ConnectionInfo;
  gameInfo: GameInfo;
}

const useGameSocket = (): useGameSocketReturn => {
  const navigate = useNavigate();

  // internal copy of players
  // necessary because the onPlayerConnectionStateUpdate handler gets fired immediately
  // after the onJoined handler before the component gets a chance to rerender
  // super jank but it works (yay!!)
  let __players: Player[] = [];
  const [players, setPlayers] = useState<Player[]>([]);
  const [playerIndex, setPlayerIndex] = useState(-1);
  let __playableCards: GameCard[] = [];
  const [playableCards, setPlayableCards] = useState<GameCard[]>([]);
  const [unplayableCards, setUnplayableCards] = useState<GameCard[]>([]);
  const [selectedPlayableCardIndex, setSelectedPlayableCardIndex] = useState(0);
  const [selectedUnplayableCardIndex, setSelectedUnplayableCardIndex] =
    useState(0);

  let __playedCards: GameCard[] = [];
  const [playedCards, setPlayedCards] = useState<GameCard[]>([]);

  const { isConnected, id } = useSocket({
    onJoined: (joinInfo, maxPlayers, selfIndex, cb) => {
      const _players: Player[] = [];
      for (let i = 0; i < maxPlayers; i++) {
        _players.push({ connected: false, ready: false });
      }
      for (const info of joinInfo) {
        _players[info.playerIndex].connected = true;
        _players[info.playerIndex].ready = info.ready;
      }
      __players = _players;
      setPlayers([...__players]);
      setPlayerIndex(selfIndex);
      cb(); // notify the server that the client acknowledged the message
    },
    onPlayerConnectionStateUpdate(index, connected) {
      __players[index].connected = connected;
      setPlayers([...__players]);
    },
    onPlayerReadyStateUpdate: (index, ready) => {
      __players[index].ready = ready;
      setPlayers([...__players]);
    },
    onCardPlayed: (index, card) => {
      console.log('on card played', index, card);
      __playedCards = [...__playedCards, card];
      setPlayedCards(__playedCards);
    },
    onImpossibleCards: (data) => {
      console.log('on impossible cards', data);
      setUnplayableCards(data); // maybe sort by color then number for better experience?
      setSelectedUnplayableCardIndex(Math.floor(data.length / 2)); // prevent out of bounds errors
    },
    onPossibleCards: (data) => {
      console.log('on possible cards', data);
      __playableCards = data;
      setPlayableCards([...__playableCards]);
      setSelectedPlayableCardIndex(Math.floor(__playableCards.length / 2)); // prevent out of bounds errors
    },
    onPosition: (position) => {
      console.log('on position', position);
      setSelectedPlayableCardIndex(position);
    },

    onGameStarted: () => {
      console.log('game started!!');
      navigate('/game');
    },

    onGameEnded: (data) => {
      // possibly navigate winner and loser to WinScreen and LoseScreen respectively? Could just be a pop-up with a countdown for disconnect
    },

    onGameClosed: () => {
      socket.disconnect(); // disconnect both players
    },
  });

  useEffect(() => {
    console.log('update');
  }, [players, playerIndex, playableCards, unplayableCards, playedCards]);

  return {
    connectionInfo: {
      isConnected,
      id,
    },
    gameInfo: {
      players,
      playerIndex,
      playedCards,
      playableCards,
      unplayableCards,
      selectedPlayableCardIndex,
      selectedUnplayableCardIndex,
    },
  };
};

export { useSocket, useGameSocket };
