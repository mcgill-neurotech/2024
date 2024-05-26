import React from 'react';

// Standard Normal variate using Box-Muller transform.
function gaussianRandom(mean = 0, stdev = 1) {
  const u = 1 - Math.random(); // Converting [0,1) to (0,1]
  const v = Math.random();
  const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  // Transform to the desired mean and standard deviation:
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
  cards: React.ReactNode[];
}

const rotations = generateRandom(200, 0, Math.PI / 12);
const xPositions = generateRandom(200, 0, 6);
const yPositions = generateRandom(200, 0, 6);
const PlayingPile: React.FC<IPlayingPileProps> = ({ cards }) => {
  return (
    <div className="flex items-center justify-center">
      <div className="relative items-center justify-center">
        {cards.map((card, i) => {
          return (
            <div
              key={i}
              className="absolute"
              style={{
                left: xPositions[i],
                top: yPositions[i],
                transform: `rotate(${rotations[i]}rad)`,
              }}
            >
              {card}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default PlayingPile;
