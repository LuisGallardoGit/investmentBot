const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const path = require('path');
const fs = require('fs');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

const PORT = 3000;
const WORKSPACE_DATA = path.join(__dirname, '../data');

app.use(express.static(path.join(__dirname, 'public')));

// Servir el glosario directamente
app.get('/api/glossary', (req, res) => {
    const glossaryPath = path.join(WORKSPACE_DATA, 'glossary.json');
    if (fs.existsSync(glossaryPath)) {
        res.json(JSON.parse(fs.readFileSync(glossaryPath, 'utf8')));
    } else {
        res.status(404).send('Not found');
    }
});

// Watcher para logs de acciones del bot
fs.watch(path.join(WORKSPACE_DATA, 'history.csv'), (event) => {
    if (event === 'change') {
        io.emit('market_update', { timestamp: new Date() });
    }
});

io.on('connection', (socket) => {
    console.log('Client connected to Jarvis Dashboard');
});

server.listen(PORT, () => {
    console.log(`Jarvis Dashboard running at http://localhost:${PORT}`);
});
