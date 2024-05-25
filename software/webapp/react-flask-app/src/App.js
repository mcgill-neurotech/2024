import React, { useState } from "react"
import './App.css';

/* Start to display cards in hand
Idea - use position: absolute for card class to allow overlap of card images at the sides

const Card = ({ color, number }) => {
  const [card, setCard] = useState({
  color: {color},
  number: {number}
  });
};

 function CardWheel(cardNum) {
  for (let i = 0; i < cardNum; i++) {

  }
  return (

  );
} */

function App() {
  const [cardNum, setCardNum] = useState(0)

  return (
    <div className="App">
      <header className="App-header">
        <div className="topPane"></div>
        <div className="bottomPane">
          <button onClick={() => setCardNum(cardNum+1)}>
            Pick up a card!
          </button>
          <div className="cardWheel">
            Your hand <br></br>
            {cardNum} cards
          </div>
        </div>
      </header>

    </div>
  );
}

export default App;
