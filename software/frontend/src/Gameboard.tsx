// src/Gameboard.tsx
import React, { useCallback, useState, useEffect } from 'react';
import './Gameboard_style.css';
import useSocket from './useSocket';
import Card, { CardColor, ICardProps } from './Card';
import { CardFanLinear } from './CardFan';
import PlayingPile from './CardPile';

const texts = [...Array.from(Array(10).keys())];
const colors = Object.values(CardColor);

const cartesian = (...sets: any[]): any[] =>
  sets.reduce((a, b) =>
    a.flatMap((d: any) => b.map((e: any) => [d, e].flat())),
  );

const generateDummyCards = (): ICardProps[] => {
  const cards: ICardProps[] = [];

  cards.push(
    ...colors.map((color) => ({
      corners: <p className="text-white text-4xl text-shadow">{'+2'}</p>,
      color,
      center: <p className="text-xl text-shadow" style={{ color: color }}>some svg?</p>,
    })),
  );
  cards.push(
    ...colors.map((color) => ({
      corners: 'reverse',
      color,
      center: 'reverse',
    })),
  );
  cards.push(
    ...colors.map((color) => ({
      corners: <p className="text-white text-4xl text-shadow">ø</p>,
      color,
      center: <p className="text-6xl text-shadow" style={{ color: color }}>ø</p>,
    })),
  );
  cards.push(
    ...colors.map((color) => ({
      corners: <p className="text-white text-sm">{'color wheel svg'}</p>,
      color: '#000000',
      center: <p className="text-xl text-shadow" style={{ color: color }}>color wheel</p>,
    })),
  );
  cards.push(
    ...colors.map((color) => ({
      corners: <p className="text-white text-4xl text-shadow">{'+4'}</p>,
      color: '#000000',
      center: <p className="text-xl text-shadow" style={{ color: color }}>+4 svg?</p>,
    })),
  );
  cards.push(
    ...cartesian(texts, colors).map(([text, color]: [string, CardColor]) => ({
      corners: <p className="text-white text-4xl text-shadow">{text}</p>,
      color,
      center: <p className="text-6xl text-shadow" style={{ color: color }}>{text}</p>,
    })),
  );
  return cards;
};

const CARD_WIDTH = 100; // Replace with actual card width
const CARD_HEIGHT = 150; // Replace with actual card height

const Gameboard: React.FC = () => {
  const [selectedPlayableCardIndex, setSelectedPlayableCardIndex] = useState(0);
  const [selectedUnplayableCardIndex, setSelectedUnplayableCardIndex] = useState(0);
  const [topCenterCard, setTopCenterCard] = useState<ICardProps | null>(null);
  const [playableCards, setPlayableCards] = useState<ICardProps[]>([]);
  const [unplayableCards, setUnplayableCards] = useState<ICardProps[]>([]);
  const cards = useCallback(generateDummyCards, []);
  const { isConnected, id } = useSocket();

  const allCards = cards();

  useEffect(() => {
    if (topCenterCard) {
      const playable = allCards.filter(card => card.color === topCenterCard.color);
      const unplayable = allCards.filter(card => card.color !== topCenterCard.color);
      setPlayableCards(playable);
      setUnplayableCards(unplayable);
    }
  }, [topCenterCard, allCards]);

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
          cards={allCards}
          cardWidth={CARD_WIDTH}
          cardHeight={CARD_HEIGHT}
          setTopCenterCard={setTopCenterCard}
        />
      </div>
      <div className="bottom-section">
        <div className="half-section">
          <h2 className="text-center text-white">Unplayable Cards</h2>
          <CardFanLinear
            selected={selectedUnplayableCardIndex}
            spread={1}
            onSelected={(i) => setSelectedUnplayableCardIndex(i)}
            cards={unplayableCards.map((card, index) => <Card key={index} {...card} />)}
          />
        </div>
        <div className="half-section">
          <h2 className="text-center text-white">Playable Cards</h2>
          <CardFanLinear
            selected={selectedPlayableCardIndex}
            spread={1}
            onSelected={(i) => setSelectedPlayableCardIndex(i)}
            cards={playableCards.map((card, index) => <Card key={index} {...card} />)}
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
