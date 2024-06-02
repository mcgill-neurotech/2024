import React, { useEffect } from 'react';
import Card, { ICardProps } from './Card';

function gaussianRandom(mean = 0, stdev = 1) {
  const u = 1 - Math.random();
  const v = Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  return z * stdev + mean;
}

function generateRandom(n: number, mean = 0, variance = 1) {
  const ret = [];
  for (let i = 0; i < n; i++) {
    ret.push(gaussianRandom(mean, variance));
  }
  return ret;
}

interface IPlayingPileProps {
  cards: ICardProps[];
  cardWidth: number;
  cardHeight: number;
  setTopCenterCard: (card: ICardProps) => void;
}

const rotations = generateRandom(200, 0, Math.PI / 12);
const xPositions = generateRandom(200, 0, 6);
const yPositions = generateRandom(200, 0, 6);

const PlayingPile: React.FC<IPlayingPileProps> = ({
  cards,
  cardWidth,
  cardHeight,
  setTopCenterCard,
}) => {
  useEffect(() => {
    if (cards.length > 0) {
      setTopCenterCard(cards[0]);
    }
  }, [cards, setTopCenterCard]);

  return (
    <div className="flex items-center justify-center">
      <div className="relative items-center justify-center">
        {cards.map((card, i) => (
          <div
            key={i}
            className="absolute"
            style={{
              left: `calc(50% + ${xPositions[i]}px - ${cardWidth / 2}px)`,
              top: `calc(50% + ${yPositions[i]}px - ${cardHeight / 2}px)`,
              transform: `rotate(${rotations[i]}rad)`,
              marginLeft: '-30px',
              marginTop: '-60px',
            }}
          >
            <Card {...card} />
          </div>
        ))}
      </div>
    </div>
  );
};

export default PlayingPile;
