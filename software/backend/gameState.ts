class card {
    color: "";
    number: 0;
    joker: false;
  }

class gameState {
    

      player: {
        hand: [],
        possible_hand: [],
        selected_card: 0,
        impossible_hand: []
      }

    deck: card[];
    played_cards: card[];
    top_card: played_cards[0];
  }

  

  