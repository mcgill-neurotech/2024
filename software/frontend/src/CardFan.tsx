import React from 'react';
import Card, { ICardProps } from './Card';

function calculate_fan_positions(N: number, selected: number, spread: number) {
  const B = Math.floor((N - 1) / 2);
  const positions = [];
  for (let n = -B; n <= B; n++) {
    positions.push(n);
  }
  if (N % 2 == 0) positions.push(B + 1);
  const center = positions[selected];
  return positions.map((i) => ({
    left: (i - center) * spread * 50, // Spread horizontally
    zIndex: -Math.abs(i - center) + B + 10,
  }));
}

interface ICardFanProps {
  cards: ICardProps[];
  spread: number;
  selected: number;
  //onSelected: (index: number) => void;
}

const CardFanLinear: React.FC<ICardFanProps> = ({
  cards,
  spread,
  selected,
  //onSelected,
}) => {
  const positions = calculate_fan_positions(cards.length, selected, spread);
  console.log(cards, positions);

  return (
    <div className="card-fan-wrapper">
      {cards.map((props, i) => {
        const { left, zIndex } = positions[i];
        return (
          <div
            //onClick={() => onSelected(i)}
            key={i}
            className="card"
            style={{
              left: `calc(50% + ${left}px)`,
              zIndex,
              top: `50%`, // Center vertically
              transform: `translate(-50%, -50%)`, // Adjust for centering
            }}
          >
            <Card key={i} {...props} />
          </div>
        );
      })}
    </div>
  );
};

export { CardFanLinear };
