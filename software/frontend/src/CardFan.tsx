import React from 'react';

function calculate_fan_angles(N: number, selected: number, spread: number) {
  // https://www.desmos.com/calculator/s7251p1bor
  const B = N % 2 == 0 ? (N - 1) / 2 : Math.floor(N / 2);
  const arr = [];
  for (let n = -B; n <= B; n++) {
    arr.push(n);
  }
  const center = arr[selected];
  return arr.map((i) => Math.atan(spread * (i - center)) + Math.PI / 2);
}

interface ICardFanProps {
  cards: React.ReactNode[];
  spread: number;
  selected: number;
  onSelected: (index: number) => void;
}

const CardFanCirular: React.FC<ICardFanProps> = ({
  cards,
  spread,
  selected,
  onSelected,
}) => {
  const radius = 400;
  const angles = calculate_fan_angles(cards.length, selected, spread);
  return (
    <div className="flex items-center justify-center">
      <div className="relative">
        {cards.map((card, i) => {
          let x = radius * Math.cos(angles[i]);
          let y = radius * Math.sin(angles[i]);
          return (
            <div
              onClick={() => onSelected(i)}
              key={i}
              className="absolute"
              style={{
                left: x,
                bottom: y - radius + (selected == i ? 30 : 0),
                transform: `rotate(${Math.PI / 2 - angles[i]}rad)`,
                ...{
                  zIndex: selected == i ? 2 : 0,
                },
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

export default CardFanCirular;
