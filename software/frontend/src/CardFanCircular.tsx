/* const CardFanCirular: React.FC<ICardFanProps> = ({
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
  */


 /* TO ADD INTO GAMEBOARD.TSX 

  <CardFanCirular
          selected={selectedCardIndex}
          spread={0.2}
          onSelected={(i) => setSelectedCardIndex(i)}
          cards={cards()}
        />
        */