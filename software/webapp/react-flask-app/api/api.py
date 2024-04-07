from flask import Flask

app = Flask(__name__)

@app.route('/')
def display_game_platform():
    return