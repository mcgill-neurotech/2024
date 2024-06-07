import React, { useCallback, useState, useEffect } from 'react';
import './Gameboard_style.css';
import useSocket from './useSocket';
import Card, { CardColor, ICardProps } from './Card';
import { CardFanLinear } from './CardFan';
import PlayingPile from './CardPile';
import { Card as GameCard } from '../../backend/game';
import SkipIcon from './SkipIcon';
import SquaresIcon from './SquaresIcon';
import ColorwheelIcon from './ColorwheelIcon';

const texts = [...Array.from(Array(10).keys())];
const colors = Object.values(CardColor);

const cartesian = (...sets: any[]): any[] =>
  sets.reduce((a, b) =>
    a.flatMap((d: any) => b.map((e: any) => [d, e].flat())),
  );

const colorMap = (color: string) => {
  if (color === 'blue') return CardColor.Blue;
  else if (color === 'yellow') return CardColor.Yellow;
  else if (color === 'red') return CardColor.Red;
  else if (color === 'green') return CardColor.Green;
  else if (color === 'wild') return '#000000';
  return '#000000';
};

const makeCardProps = (card: GameCard) => {
  let ret: ICardProps = {
    center: undefined,
    corners: undefined,
    color: '',
  };
  ret.center = colorMap(card.color);

  if (card.number <= 9) {
    ret.corners = (
      <p className="text-white text-4xl text-shadow">{card.number}</p>
    );
    ret.center = (
      <p className="text-6xl text-shadow" style={{ color: ret.color }}>
        {card.number}
      </p>
    );
  } else if (card.number === 10) {
    /* 10 = skip; 11 = +2; 12 = +4; 13 = wildcard */
    ret.corners = (
      <SkipIcon
        color={ret.color}
        width={50}
        height={50}
        className="icon-shadow"
      />
    );
    ret.center = (
      <SkipIcon color={ret.color} width={50} height={50} strokeWidth={2} />
    );
  } else if (card.number === 11) {
    ret.corners = <p className="text-white text-4xl text-shadow">{'+2'}</p>;
    ret.center = (
      <SquaresIcon
        color={ret.color}
        width={50}
        height={50}
        className="icon-shadow"
      />
    );
  } else if (card.number === 12) {
    ret.corners = <p className="text-white text-4xl text-shadow">{'+4'}</p>;
    ret.center = <ColorwheelIcon height={200} width={200} />;
    ret.centerClassName = 'bg-black';
  } else if (card.number === 13) {
    ret.corners = <ColorwheelIcon width={45} height={45} />;
    ret.center = <ColorwheelIcon height={200} width={200} />;
    ret.centerClassName = 'bg-black';
  }

  return ret;
};

const CARD_WIDTH = 100; // Replace with actual card width
const CARD_HEIGHT = 150; // Replace with actual card height

const Gameboard: React.FC = () => {
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
          Math.min(selectedPlayableCardIndex + 1, playableCards.length),
        );
      }
    },
  });

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
            onSelected={(i) => setSelectedUnplayableCardIndex(i)}
            cards={unplayableCards.map((c) => makeCardProps(c))}
          />
        </div>
        <div className="half-section">
          <h2 className="text-center text-white">Playable Cards</h2>
          <CardFanLinear
            selected={selectedPlayableCardIndex}
            spread={1}
            onSelected={(i) => setSelectedPlayableCardIndex(i)}
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
