// src/App.tsx
import { Route, Routes } from 'react-router-dom';
//MAKE SURE TO npm install react-router-dom !!

import HomeScreen from './HomeScreen';
import Gameboard from './Gameboard';
import { GameContext } from './gameContext';
import { useGameSocket } from './useSocket';

function App() {
  const gameInfo = useGameSocket();
  return (
    <GameContext.Provider value={gameInfo}>
      <Routes>
        <Route path="/game" element={<Gameboard />} />
        <Route path="/" element={<HomeScreen />} />
      </Routes>
    </GameContext.Provider>
  );
}

export default App;
