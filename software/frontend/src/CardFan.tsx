import React from 'react';

function calculate_fan_angles(N: number, selected: number, spread: number) {
  // https://www.desmos.com/calculator/s7251p1bor for an explanation
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
          let active = i == selected;
          let x = -radius * Math.cos(angles[i]);
          let y = radius * Math.sin(angles[i]) - radius + (active ? 30 : 0);
          let r = angles[i] - Math.PI / 2;
          let z = selected == i ? 2 : 0;
          return (
            <div
              onClick={() => onSelected(i)}
              key={i}
              className="absolute"
              style={{
                left: x,
                top: -y, // overflow downwards instead of upwards. An alternative is "bottom: y" to overflow upwards
                transform: `rotate(${r}rad)`,
                zIndex: z,
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
    <div className="flex items-center justify-center">
      <div className="relative">
        {cards.map((card, i) => {
          let active = selected == i;
          let z = -Math.abs(i - selected) + cards.length / 2 + 20;
          let y = active ? 30 : 0;
          return (
            <div
              onClick={() => onSelected(i)}
              key={i}
              className="absolute"
              style={{
                left: xPositions[i],
                zIndex: z,
                top: -y,
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

export { CardFanCirular, CardFanLinear };
