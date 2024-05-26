import { useState } from 'react';
import Card, { CardColor } from './Card';
import CardFanCirular from './CardFan';

const texts = [...Array.from(Array(10).keys())];
const colors = Object.values(CardColor);

/**
 * Cartesion product set operation
 *
 * @param {...any[]} sets A list of arrays
 * @returns {*} The cartesion product of the arrays
 */
const cartesian = (...sets: any[]): any[] =>
  sets.reduce((a, b) =>
    a.flatMap((d: any) => b.map((e: any) => [d, e].flat())),
  );

function App() {
  const [selectedCardIndex, setSelectedCardIndex] = useState(0);
  console.log(selectedCardIndex);

  return (
    <div>
      <h1>Test</h1>
      {/* 0-9 cards */}
      <div className="flex flex-row flex-wrap">
        {cartesian(texts, colors).map(
          ([text, color]: [string, CardColor], i) => {
            return (
              <Card
                key={i}
                corners={
                  <p className="text-white text-4xl text-shadow">{text}</p>
                }
                color={color}
                center={
                  <p className="text-6xl text-shadow" style={{ color: color }}>
                    {text}
                  </p>
                }
              />
            );
          },
        )}
        {/* +2 card */}
        {colors.map((color, i) => {
          return (
            <Card
              key={i}
              corners={
                <p className="text-white text-4xl text-shadow">{'+2'}</p>
              }
              color={color}
              center={
                <p className="text-xl text-shadow" style={{ color: color }}>
                  some svg?
                </p>
              }
            />
          );
        })}
        {/* reverse card */}
        {colors.map((color, i) => {
          return (
            <Card
              key={i}
              corners={'reverse'}
              color={color}
              center={'reverse'}
            />
          );
        })}
        {/* block card */}
        {colors.map((color, i) => {
          return (
            <Card
              key={i}
              corners={<p className="text-white text-4xl text-shadow">ø</p>}
              color={color}
              center={
                <p className="text-6xl text-shadow" style={{ color: color }}>
                  ø
                </p>
              }
            />
          );
        })}
        {/* color wheel card */}
        {colors.map((color, i) => {
          return (
            <Card
              key={i}
              corners={
                <p className="text-white text-sm">{'color wheel svg'}</p>
              }
              color={'#000000'}
              center={
                <p className="text-xl text-shadow" style={{ color: color }}>
                  color wheel
                </p>
              }
            />
          );
        })}
        {/* +4 card */}
        {colors.map((color, i) => {
          return (
            <Card
              key={i}
              corners={
                <p className="text-white text-4xl text-shadow">{'+4'}</p>
              }
              color={'#000000'}
              center={
                <p className="text-xl text-shadow" style={{ color: color }}>
                  +4 svg?
                </p>
              }
            />
          );
        })}
      </div>
      <div className="mt-8"></div>
      <CardFanCirular
        selected={selectedCardIndex}
        spread={0.2}
        onSelected={(i) => setSelectedCardIndex(i)}
        cards={cartesian(texts, colors).map(
          ([text, color]: [string, CardColor], i) => {
            return (
              <Card
                key={i}
                corners={
                  <p className="text-white text-4xl text-shadow">{text}</p>
                }
                color={color}
                center={
                  <p className="text-6xl text-shadow" style={{ color: color }}>
                    {text}
                  </p>
                }
              />
            );
          },
        )}
      />
    </div>
  );
}

export default App;
