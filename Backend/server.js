const express = require('express');
const http = require('http');
const cors = require('cors');
const { Server } = require('socket.io');
const axios = require('axios');
const { parseEngineOutput } = require('./parser');

const app = express();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: '*' } });

app.use(cors());
app.use(express.json());

const PYTHON_API = 'http://127.0.0.1:8000';

// Track overlay connection status
let overlayLastSeen = 0;

// Helper to stream SSE from Python to Socket.IO clients
async function streamFromPython(endpoint, reqBody, socketEventPrefix, reqMethod = 'POST') {
    try {
        const response = await axios({
            method: reqMethod,
            url: `${PYTHON_API}${endpoint}`,
            data: reqBody,
            responseType: 'stream'
        });

        return new Promise((resolve, reject) => {
            let fullResult = null;
            let currentEvent = 'message';
            
            response.data.on('data', chunk => {
                const lines = chunk.toString().split('\n');
                for (let line of lines) {
                    if (line.startsWith('event:')) {
                        currentEvent = line.substring(6).trim();
                    } else if (line.startsWith('data:')) {
                        const dataStr = line.substring(5).trim();
                        if (!dataStr) continue;
                        
                        try {
                            const data = JSON.parse(dataStr);
                            
                            if (currentEvent === 'error') {
                                io.emit(`${socketEventPrefix}_error`, data);
                                reject(new Error(data));
                            } else if (currentEvent === 'result') {
                                fullResult = data;
                            } else {
                                // Progress event
                                io.emit(`${socketEventPrefix}_update`, data);
                            }
                        } catch (e) {
                            console.error('JSON parse error on SSE data', dataStr);
                        }
                    } else if (line === '') {
                        currentEvent = 'message'; // reset
                    }
                }
            });

            response.data.on('end', () => {
                resolve(fullResult);
            });
            response.data.on('error', (err) => reject(err));
        });
    } catch (err) {
        console.error(`Failed to stream from Python ${endpoint}:`, err.message);
        throw err;
    }
}

// ── Standard API Endpoints ───────────────────────────────────────────

app.get('/api/status', async (req, res) => {
    try {
        const pyRes = await axios.get(`${PYTHON_API}/health`);
        const overlayAlive = (Date.now() - overlayLastSeen) < 30000; // 30s timeout
        res.json({ 
            status: 'ok', 
            python_engine: pyRes.data.status,
            overlay_connected: overlayAlive
        });
    } catch (e) {
        res.status(503).json({ status: 'error', python_engine: 'down', overlay_connected: false });
    }
});

app.post('/api/analyze', async (req, res) => {
    try {
        io.emit('pipeline_update', { progress: '> [Browser] Analyze pipeline initiated...' });
        const result = await streamFromPython('/analyze', {}, 'analyze', 'GET');
        io.emit('analyze_complete', { ...result, source: 'browser' });
        res.json(result);
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

app.post('/api/explain', async (req, res) => {
    try {
        io.emit('pipeline_update', { progress: '> [Browser] Explain pipeline initiated...' });
        const result = await streamFromPython('/explain', {}, 'pipeline', 'GET');
        const rawText = result?.raw_explanation || '';
        const structured = parseEngineOutput(rawText);
        structured.source = 'browser';
        
        io.emit('pipeline_complete', structured);
        res.json(structured);
    } catch (e) {
        io.emit('pipeline_error', e.message);
        res.status(500).json({ error: e.message });
    }
});

app.post('/api/autotype/calculate', async (req, res) => {
    try {
        const result = await streamFromPython('/autotype/calculate', req.body, 'autotype');
        res.json(result);
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

app.post('/api/autotype/execute', async (req, res) => {
    try {
        const result = await streamFromPython('/autotype/execute', req.body, 'autotype_execute');
        res.json(result);
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

// ── Overlay Push Endpoints ───────────────────────────────────────────
// These receive data from the Python overlay (web_bridge.py) and
// broadcast via Socket.IO to all connected React clients.

app.post('/api/overlay/progress', (req, res) => {
    overlayLastSeen = Date.now();
    const { message } = req.body;
    if (message) {
        io.emit('pipeline_update', { progress: message });
    }
    res.json({ ok: true });
});

app.post('/api/overlay/result', (req, res) => {
    overlayLastSeen = Date.now();
    const { type, data, raw_text } = req.body;
    
    if (!type || !data) {
        return res.status(400).json({ error: 'Missing type or data' });
    }

    console.log(`[Overlay Push] Received ${type} result — broadcasting to browser clients`);

    if (type === 'analyze') {
        io.emit('analyze_complete', { ...data, source: 'overlay' });
    } else if (type === 'explain') {
        data.source = 'overlay';
        io.emit('pipeline_complete', data);
    } else if (type === 'continue') {
        data.source = 'overlay';
        io.emit('continue_complete', data);
    } else if (type === 'autotype') {
        data.source = 'overlay';
        io.emit('autotype_complete', data);
    }

    res.json({ ok: true, broadcast: type });
});

app.post('/api/overlay/hotkey', (req, res) => {
    overlayLastSeen = Date.now();
    const { action } = req.body;
    io.emit('overlay_hotkey', { action });
    console.log(`[Overlay] Hotkey event: ${action}`);
    res.json({ ok: true });
});

// ── Socket.IO ───────────────────────────────────────────────────────

io.on('connection', (socket) => {
    console.log('Client connected:', socket.id);
    
    // Send current overlay status on connect
    const overlayAlive = (Date.now() - overlayLastSeen) < 30000;
    socket.emit('overlay_status', { connected: overlayAlive });
    
    socket.on('disconnect', () => {
        console.log('Client disconnected:', socket.id);
    });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Node Bridge listening on port ${PORT}`);
});
