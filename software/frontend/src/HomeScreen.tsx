// src/HomeScreen.tsx
import React, { useContext } from 'react';
import './HomeScreen_style.css';
import { GameContext } from './gameContext';

const HomeScreen: React.FC = () => {
  const info = useContext(GameContext);
  console.log(info);
  return (
    <div className="home-screen">
      <div className="top-section">
        <div className="my-4 flex flex-col items-center">
          <span
            className={`${
              info?.connectionInfo.isConnected ? 'bg-green-300' : 'bg-red-500'
            } px-4 py-2 rounded-md`}
          >
            {info?.connectionInfo.isConnected ? 'Connected' : 'Disconnected'}
          </span>
          <span className="bg-gray-300 px-3 py-1 mt-1 rounded-md">
            Session id: {info?.connectionInfo.id}
          </span>
        </div>
      </div>
      <h1 className="title fade">
        Welcome to the Uno-like Game! Waiting for all players to be ready...
      </h1>
      {info && info.gameInfo.playerIndex !== -1 && (
        <>
          <p className="text-white text-lg">{`you are player ${info?.gameInfo.playerIndex}`}</p>
          {info?.gameInfo.players.map((p, i) => {
            return (
              <div key={i} className="text-white">
                <p>{`player ${i}`}</p>
                <p>{`connected: ${p.connected}`}</p>
                <p>{`ready: ${p.ready}`}</p>
                <hr />
              </div>
            );
          })}
        </>
      )}
    </div>
  );
};

export default HomeScreen;
