// src/HomeScreen.tsx
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './HomeScreen_style.css';

const HomeScreen: React.FC = () => {
  const [showTitle, setShowTitle] = useState(false);
  const [showStartButton, setShowStartButton] = useState(false);

  useEffect(() => {
    const timer1 = setTimeout(() => {
      setShowTitle(true);
    }, 500); // Show title after 0.5 seconds

    const timer2 = setTimeout(() => {
      setShowTitle(false);
    }, 1500); // Hide title after 1 second of being visible

    const timer3 = setTimeout(() => {
      setShowStartButton(true);
    }, 2000); // Show start button after title disappears

    return () => {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
    };
  }, []);

  return (
    <div className="home-screen">
      {showTitle && <h1 className="title fade">Welcome to the Uno-like Game!</h1>}
      {showStartButton && (
        <Link to="/game">
          <button className="start-button fade">Start Game</button>
        </Link>
      )}
    </div>
  );
};

export default HomeScreen;
