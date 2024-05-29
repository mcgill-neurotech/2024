import React from 'react';
import BrainIcon from './BrainIcon';

export enum CardColor {
  Red = '#FF5555',
  Green = '#55AA55',
  Blue = '#5555FD',
  Yellow = '#FFAA00',
}

interface ICardProps {
  /**
   * The element to place at the center of the card, e.g. a p, svg, img, etc
   */
  center: React.ReactNode;
  /**
   * The element to place at the top left and bottom right of each card, e.g. a p, svg, img, etc
   */
  corners: React.ReactNode;
  /**
   * The background color of the card, #000000 format
   */
  color: string;
}

const Card: React.FC<ICardProps> = ({ color, center, corners }) => {
  return (
    <div className="w-40 h-56 border-2 border-black rounded-md flex justify-center items-stretch font-sans bg-white">
      <div
        className="m-4 rounded-md flex items-stretch flex-col grow"
        style={{ backgroundColor: color }}
      >
        <div className="flex px-2 pt-1 justify-between">
          {corners}
          <BrainIcon style={{ height: 40, width: 40, color: 'white' }} />
        </div>
        <div className="grow relative flex items-stretch">
          <div className="grow flex items-stretch">
            <div className="ellipse bg-white grow" />
            <div className="absolute w-full h-full flex items-center justify-center">
              {center}
            </div>
          </div>
        </div>
        <div className="flex flex-row-reverse px-2 pb-1">{corners}</div>
      </div>
    </div>
  );
};

export default Card;
