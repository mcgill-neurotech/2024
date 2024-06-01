const express = require('express');
const http = require('http');
const webSocket = require('ws');

const port = 8080;
const server = http.createServer(express);
const wss = new webSocket.Server( { server } );

const connections = { }
const users = { }

const handleMessage = (bytes, uuid) => {
    // message = {chosencard}
    // user = users[uuid]
    // user.state.hand
    // user.state.turn
    // change turn state for other player as well?
    // 
}

wss.getUniqueID = function() {
    function s4() {
        return Math.floor((1 + Math.random()) * 0x10000).toString(16).substring(1);
    }
    return s4() + s4() + '-' + s4();
};

wss.on('connection', (connection, request) => {
    ws.id = wss.getUniqueID();
    wss.clients.forEach(function each(client) {
        console.log('Client.ID ' + client.id)
    })

    // Array allows broadcast to all clients
    connections[ws.id] = connection
    users[ws.id] = {
        state: {
            hand: [],
            choices: [],
            turn: false
        }
    }
    
    // Clients sends message for selected card
    connection.on('message', message => handleMessage(message, ws.id))
    connection.on('close', () => handleClose(ws.id))
})

server.listen(port, function() {
    console.log(`Server is listening on ${port}`)
})