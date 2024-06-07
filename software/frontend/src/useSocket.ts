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

    return () => {
      socket.off('connect', onConnect);
      socket.off('disconnect', onDisconnect);
      socket.off('Possible Cards', onPossibleCards);
      socket.off('Impossible Cards', onInpossibleCards);
      socket.off('direction', onDirection);
      socket.off('Card Played', onCardPlayed);
    };
  }, []);

  return { isConnected, id };
};

export default useSocket;
