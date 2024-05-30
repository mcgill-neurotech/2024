const express = require('express');
const http = require('http');
const webSocket = require('ws');

const port = 8080;
const server = http.createServer(express);
const wss = new webSocket.Server( { server } );

const connections = { }
const users = { }

const handleMessage = (chosenCard, uuid) => {
    // message = {chosencard}
    // user = users[uuid]
    // remove card from user.state.hand 
    // change user.state.turn and opponent's user.state.turn
    // change topcard & display on other client
    // ws.send to send message from client to the server
}

const handleClose = (uuid) => {
    console.log("Disconnected from the WebSocket server");
}

wss.getUniqueID = function() {
    function s4() {
        return Math.floor((1 + Math.random()) * 0x10000).toString(16).substring(1);
    }
    return s4() + s4() + '-' + s4();
};

wss.on('connection', (connection, request) => {
    // Assigns unique ID to clients
    ws.id = wss.getUniqueID();
    wss.clients.forEach(function each(client) {
        console.log('Client.ID: ' + client.id + " connected")
    })

    // Array stores client state
    connections[ws.id] = connection;
    users[ws.id] = {
        state: {
            hand: [],
            choices: [],
            turn: false
        }
    };
    
    // Handles client sending info about the chosen card to the server
    connection.on('message', chosenCard => handleMessage(chosenCard, ws.id))
    
    // Handles client disconnection
    connection.on('close', () => handleClose(ws.id))

    // Handles errors
    connection.on('error', (error) => {
        console.error(`WebSocket error: ${error}`);
    }
    )
})

server.listen(port, function() {
    console.log(`Server is listening on ${port}`)
})