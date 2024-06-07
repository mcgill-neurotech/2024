import React from 'react';
import Card, { ICardProps } from './Card';

function calculate_fan_positions(N: number, selected: number, spread: number) {
  const B = Math.floor((N - 1) / 2);
  const positions = [];
  for (let n = -B; n <= B; n++) {
    positions.push(n);
  }
  const center = positions[selected];
  return positions.map((i) => ({
    left: (i - center) * spread * 50, // Spread horizontally
    zIndex: -Math.abs(i - center) + B + 1,
  }));
}

interface ICardFanProps {
  cards: ICardProps[];
  spread: number;
  selected: number;
  // onSelected: (index: number) => void; // was only for testing
}

const CardFanLinear: React.FC<ICardFanProps> = ({
  cards,
  spread,
  selected,
  // onSelected,
}) => {
  const positions = calculate_fan_positions(cards.length, selected, spread);
  return (
    <div className="flex items-center justify-center card-fan-container">
      <div className="relative">
        {cards.map((props, i) => {
          const { left, zIndex } = positions[i];
          const active = selected === i;
          const y = active ? -140 : -130; // Adjust this value to move the cards up
          return (
            <div
              // onClick={() => onSelected(i)}
              key={i}
              className="absolute"
              style={{
                left: `calc(50% + ${left}px)`,
                zIndex,
                top: `${y}px`, // Adjust this value to move the cards up
                transform: 'translateX(-50%)',
              }}
            >
              <Card key={i} {...props} />
            </div>
          );
        })}
      </div>
    </div>
  );
};

export { CardFanLinear };
