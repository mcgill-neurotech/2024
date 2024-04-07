import React, { useState} from "react"
import logo from './logo.svg';
import './App.css';

function App() {
  const [cardNum, setCardNum] = useState(0);

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
        
        <div>Card Game!
          <p>You have {cardNum} cards</p>
          <button onClick={() => setCardNum(cardNum-1)}>
            Play a card!
          </button>
        </div>
      </header>
    </div>
  );
}

export default App;
