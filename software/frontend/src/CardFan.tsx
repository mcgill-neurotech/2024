// src/CardFan.tsx
import React from 'react';

function calculate_fan_angles(N: number, selected: number, spread: number) {
  const B = N % 2 === 0 ? (N - 1) / 2 : Math.floor(N / 2);
  const arr = [];
  for (let n = -B; n <= B; n++) {
    arr.push(n);
  }
  const center = arr[selected];
  return arr.map((i) => Math.atan(spread * (i - center)) + Math.PI / 2);
}

interface ICardFanProps {
  cards: React.ReactElement[];
  spread: number;
  selected: number;
  onSelected: (index: number) => void;
}

const CardFanLinear: React.FC<ICardFanProps> = ({
  cards,
  spread,
  selected,
  onSelected,
}) => {
  const xPositions = calculate_fan_angles(cards.length, selected, spread).map(
    (x) => (x - Math.PI / 2) * 200, // arbitrary but could be changed
  );
  return (
    <div className="flex items-center justify-center card-fan-container">
      <div className="relative">
        {cards.map((card, i) => {
          const active = selected === i;
          const z = -Math.abs(i - selected) + cards.length / 2 + 20;
          const y = active ? 60 : 30; // Increased y to move cards up
          return (
            <div
              onClick={() => onSelected(i)}
              key={i}
              className="absolute"
              style={{
                left: xPositions[i],
                zIndex: z,
                top: -y,
                marginTop: '-90px',
                marginLeft: '-65px',
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

export { CardFanLinear };
