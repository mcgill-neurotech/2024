// src/App.tsx
import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
//MAKE SURE TO npm install react-router-dom !!


import HomeScreen from './HomeScreen';
import Gameboard from './Gameboard';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/game" element={<Gameboard />} />
        <Route path="/" element={<HomeScreen />} />
      </Routes>
    </Router>
  );
}

export default App;
