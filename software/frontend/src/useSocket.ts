// src/useSocket.ts
import { useEffect, useState } from 'react';
import { socket } from './socket';
import { Card as GameCard } from '../../backend/game';

type Direction = 'left' | 'right';
interface IUseSocketParams {
  // data = array of current player's possible hand
  onPossibleCards: (data: GameCard[]) => void;

  // data = array of current player's impossible hand
  onInpossibleCards: (data: GameCard[]) => void;

  // data = string "right" or "left"
  onDirection: (direction: Direction) => void;

  // data = current top card
  onCardPlayed: (data: GameCard) => void;

  // data = {???}
  // onGameStarted: (data) => void

  // data = {???}
  // onGameEnded: (data) => void
}

const useSocket = ({
  onPossibleCards,
  onInpossibleCards,
  onDirection,
  onCardPlayed,
}: IUseSocketParams) => {
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
    socket.on('disconnect', onDisconnect);
    socket.on('Possible Cards', onPossibleCards);
    socket.on('Impossible Cards', onInpossibleCards);
    socket.on('direction', onDirection);
    socket.on('Card Played', onCardPlayed);
    // game started handler
    // game ended handler

    return () => {
      socket.off('connect', onConnect);
      socket.off('disconnect', onDisconnect);
      socket.off('Possible Cards', onPossibleCards);
      socket.off('Impossible Cards', onInpossibleCards);
      socket.off('direction', onDirection);
      socket.off('Card Played', onCardPlayed);
      // game started handler
      // game ended handler
    };
  }, []);

  return { isConnected, id };
};

const useGameSocket = () => {
  const [playableCards, setPlayableCards] = useState<GameCard[]>([]);
  const [unplayableCards, setUnplayableCards] = useState<GameCard[]>([]);
  const [selectedPlayableCardIndex, setSelectedPlayableCardIndex] = useState(0);
  const [selectedUnplayableCardIndex, setSelectedUnplayableCardIndex] =
    useState(0);
  const [playedCards, setPlayedCards] = useState<GameCard[]>([]);

  const { isConnected, id } = useSocket({
    onCardPlayed: (data) => {
      setPlayedCards([...playedCards, data]);
    },
    onInpossibleCards: (data) => {
      setUnplayableCards(data);
    },
    onPossibleCards: (data) => {
      setPlayableCards(data);
    },
    onDirection: (data) => {
      if (data === 'left') {
        setSelectedPlayableCardIndex(
          Math.max(0, selectedPlayableCardIndex - 1),
        );
      } else {
        setSelectedPlayableCardIndex(
          Math.min(selectedPlayableCardIndex + 1, playableCards.length - 1),
        );
      }
    },
  });

  return {
    connectionInfo: {
      isConnected,
      id,
    },
    gameInfo: {
      playedCards,
      playableCards,
      unplayableCards,
      selectedPlayableCardIndex,
      selectedUnplayableCardIndex,
    },
  };
};

export { useSocket, useGameSocket };
