import React, { useState } from 'react';
import './Gameboard_style.css';
import { useGameSocket } from './useSocket';
import { CardColor, ICardProps } from './Card';
import { CardFanLinear } from './CardFan';
import PlayingPile from './CardPile';
import { Card as GameCard } from '../../backend/game';
import SkipIcon from './SkipIcon';
import SquaresIcon from './SquaresIcon';
import ColorwheelIcon from './ColorwheelIcon';

const cardColorMap = new Map<string, CardColor>([
  ['blue', CardColor.Blue],
  ['yellow', CardColor.Yellow],
  ['green', CardColor.Green],
  ['red', CardColor.Red],
  ['wild', CardColor.Black],
]);

const specialCornerMap = new Map<number, (color: string) => React.ReactNode>([
  [
    10,
    (c) => (
      <SkipIcon color={c} width={50} height={50} className="icon-shadow" />
    ),
  ],
  [11, (c) => <p className="text-white text-4xl text-shadow">{'+2'}</p>],
  [12, (c) => <p className="text-white text-4xl text-shadow">{'+4'}</p>],
  [13, (c) => <ColorwheelIcon width={45} height={45} />],
]);

const specialCenterMap = new Map<number, (color: string) => React.ReactNode>([
  [10, (c) => <SkipIcon color={c} width={50} height={50} strokeWidth={2} />],
  [
    11,
    (c) => (
      <SquaresIcon color={c} width={50} height={50} className="icon-shadow" />
    ),
  ],
  [12, (c) => <ColorwheelIcon height={200} width={200} />],
  [13, (c) => <ColorwheelIcon height={200} width={200} />],
]);

const specialClassNameMap = new Map<number, string>([
  [12, 'bg-black'],
  [13, 'bg-black'],
]);

const makeCardProps = (card: GameCard) => {
  let ret: ICardProps = {
    color: cardColorMap.get(card.color) ?? '#000000',
    center: undefined,
    corners: undefined,
  };

  if (card.number <= 9) {
    ret.corners = (
      <p className="text-white text-4xl text-shadow">{card.number}</p>
    );
    ret.center = (
      <p className="text-6xl text-shadow" style={{ color: ret.color }}>
        {card.number}
      </p>
    );
  } else {
    ret.corners = specialCornerMap.get(card.number)!(card.color);
    ret.center = specialCenterMap.get(card.number)!(card.color);
    ret.centerClassName = specialClassNameMap.get(card.number);
  }

  return ret;
};

const CARD_WIDTH = 100; // Replace with actual card width
const CARD_HEIGHT = 150; // Replace with actual card height

const Gameboard: React.FC = () => {
  const { connectionInfo, gameInfo } = useGameSocket();
  const isConnected = connectionInfo.isConnected;
  const id = connectionInfo.id;
  const {
    playedCards,
    playableCards,
    unplayableCards,
    selectedPlayableCardIndex,
    selectedUnplayableCardIndex,
  } = gameInfo;

  return (
    <div className="gameboard">
      <div className="top-section">
        <div className="my-4 flex flex-col items-center">
          <span
            className={`${
              isConnected ? 'bg-green-300' : 'bg-red-500'
            } px-4 py-2 rounded-md`}
          >
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
          <span className="bg-gray-300 px-3 py-1 mt-1 rounded-md">
            Session id: {id}
          </span>
        </div>
      </div>
      <div className="middle-section relative">
        <PlayingPile
          cards={playedCards.map((c) => makeCardProps(c))}
          cardWidth={CARD_WIDTH}
          cardHeight={CARD_HEIGHT}
        />
      </div>
      <div className="bottom-section">
        <div className="half-section">
          <h2 className="text-center text-white">Unplayable Cards</h2>
          <CardFanLinear
            selected={selectedUnplayableCardIndex}
            spread={1}
            cards={unplayableCards.map((c) => makeCardProps(c))}
          />
        </div>
        <div className="half-section">
          <h2 className="text-center text-white">Playable Cards</h2>
          <CardFanLinear
            selected={selectedPlayableCardIndex}
            spread={1}
            cards={playableCards.map((c) => makeCardProps(c))}
          />
        </div>
      </div>
    </div>
  );
};

export default Gameboard;

//OTHER
//pop up "play card confirmation"
//pop up "uno" len hand  == 0
//with built in timer, queue pop up "Play again? with timer (minute timer) if no selection w jaw clench, then game disconnects when timer runs out

//pop up color change
//pop up +4

//for later:
