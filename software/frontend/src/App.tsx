import { useCallback, useEffect, useState } from 'react';
import Card, { CardColor } from './Card';
import { CardFanCirular, CardFanLinear } from './CardFan';
import PlayingPile from './CardPile';
import { socket } from './socket';

const texts = [...Array.from(Array(10).keys())];
const colors = Object.values(CardColor);

/**
 * Cartesion product set operation
 *
 * @param {...any[]} sets A list of arrays (as if they were sets)
 * @returns {*} The cartesion product of the "sets" arrays
 */
const cartesian = (...sets: any[]): any[] =>
  sets.reduce((a, b) =>
    a.flatMap((d: any) => b.map((e: any) => [d, e].flat())),
  );

const generateDummyCards = () => {
  const cards = [];
  // "0-9" cards
  cards.push(
    ...cartesian(texts, colors).map(([text, color]: [string, CardColor]) => (
      <Card
        corners={<p className="text-white text-4xl text-shadow">{text}</p>}
        color={color}
        center={
          <p className="text-6xl text-shadow" style={{ color: color }}>
            {text}
          </p>
        }
      />
    )),
  );
  // "+2" cards
  cards.push(
    ...colors.map((color) => (
      <Card
        corners={<p className="text-white text-4xl text-shadow">{'+2'}</p>}
        color={color}
        center={
          <p className="text-xl text-shadow" style={{ color: color }}>
            some svg?
          </p>
        }
      />
    )),
  );
  // "reverse" cards
  cards.push(
    ...colors.map((color) => (
      <Card corners={'reverse'} color={color} center={'reverse'} />
    )),
  );
  // "skip turn" cards
  cards.push(
    ...colors.map((color) => (
      <Card
        corners={<p className="text-white text-4xl text-shadow">ø</p>}
        color={color}
        center={
          <p className="text-6xl text-shadow" style={{ color: color }}>
            ø
          </p>
        }
      />
    )),
  );
  // "color wheel" cards
  cards.push(
    ...colors.map((color) => (
      <Card
        corners={<p className="text-white text-sm">{'color wheel svg'}</p>}
        color={'#000000'}
        center={
          <p className="text-xl text-shadow" style={{ color: color }}>
            color wheel
          </p>
        }
      />
    )),
  );
  // "+4" cards
  cards.push(
    ...colors.map((color) => (
      <Card
        corners={<p className="text-white text-4xl text-shadow">{'+4'}</p>}
        color={'#000000'}
        center={
          <p className="text-xl text-shadow" style={{ color: color }}>
            +4 svg?
          </p>
        }
      />
    )),
  );

  return cards;
};

function App() {
  const [selectedCardIndex, setSelectedCardIndex] = useState(0);
  const cards = useCallback(generateDummyCards, []);
  const [isConnected, setIsConnected] = useState(socket.connected);
  const [id, setId] = useState(socket.id);

  useEffect(() => {
    function onConnect() {
      setIsConnected(true);
      setId(socket.id);
    }

    function onDisconnect() {
      setIsConnected(false);
      setId('');
    }

    socket.on('connect', onConnect);
    socket.on('disconnect', onDisconnect);

    return () => {
      socket.off('connect', onConnect);
      socket.off('disconnect', onDisconnect);
    };
  }, []);

  return (
    <div>
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
      <div className="flex flex-row flex-wrap">
        {cards().map((c, i) => (
          <div key={i}>{c}</div>
        ))}
      </div>
      <div className="mt-8" />
      <PlayingPile cards={cards()} />
      <div className="mt-[300px]" />
      <CardFanCirular
        selected={selectedCardIndex}
        spread={0.2}
        onSelected={(i) => setSelectedCardIndex(i)}
        cards={cards()}
      />
      <div className="mt-[600px]" />
      <CardFanLinear
        selected={selectedCardIndex}
        spread={1}
        onSelected={(i) => setSelectedCardIndex(i)}
        cards={cards()}
      />
    </div>
  );
}

export default App;
