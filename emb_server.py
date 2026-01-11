import json
import base64
import queue
import threading
import numpy as np
import pandas as pd
from typing import List
from openai import OpenAI
import os
import websocket
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

# Original setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
API_KEY = os.getenv("OPENAI_API_KEY")
WS_URL = 'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01'

with open('embs_subset.json') as f:
# with open('embs.json') as f:
    embs = json.load(f)
embs_arr = np.array(embs)
df = pd.read_csv('df_subset.csv')
# df = pd.read_csv('df_full.csv')

app = FastAPI()

# ============== WEB UI ==============

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Parts Recognition</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f5f5; color: #1a1a1a; }
        .container { display: flex; height: 100vh; }
        .panel { flex: 1; display: flex; flex-direction: column; background: white; border-right: 1px solid #e0e0e0; padding: 20px; overflow: hidden; }
        .panel:last-child { border-right: none; }
        .panel-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
        .panel-title { font-size: 15px; font-weight: 600; }
        .btn-group { display: flex; gap: 6px; }
        .btn-icon { width: 32px; height: 32px; border: 1px solid #ddd; border-radius: 6px; background: white; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 14px; transition: all 0.2s; }
        .btn-icon:hover { background: #f5f5f5; border-color: #999; }
        .btn-icon.recording { background: #ffe0e0; border-color: #ff4444; }
        .transcript { flex: 1; border: 1px solid #e0e0e0; border-radius: 6px; padding: 14px; overflow-y: auto; font-size: 13px; line-height: 1.5; white-space: pre-wrap; }
        .transcript:empty::before { content: 'Transcript will appear here...'; color: #999; }
        .form-section { margin-bottom: 20px; }
        .form-label { font-size: 11px; font-weight: 600; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; display: block; }
        .form-row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }
        .form-row.full { grid-template-columns: 1fr; }
        input, select { width: 100%; padding: 10px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 13px; font-family: inherit; }
        input:focus, select:focus { outline: none; border-color: #0066ff; background: #f9fbff; }
        input::placeholder { color: #999; }
        .table-section { flex: 1; display: flex; flex-direction: column; }
        .table-wrapper { flex: 1; overflow: hidden; display: flex; flex-direction: column; border: 1px solid #e0e0e0; border-radius: 6px; }
        .table-header { display: grid; grid-template-columns: 1.2fr 0.8fr 1.5fr 1.2fr 1fr 0.6fr; gap: 12px; padding: 12px 14px; background: #f9f9f9; border-bottom: 1px solid #e0e0e0; font-size: 11px; font-weight: 600; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
        .table-scroll { flex: 1; overflow-y: auto; position: relative; }
        .table-row { display: grid; grid-template-columns: 1.2fr 0.8fr 1.5fr 1.2fr 1fr 0.6fr; gap: 12px; padding: 12px 14px; border-bottom: 1px solid #f0f0f0; align-items: center; font-size: 13px; position: relative; }
        .table-row:hover { background: #fafafa; }
        .item-name { font-weight: 500; }
        .item-id { font-family: monospace; font-weight: 600; color: #0066ff; }
        .cross-sell-indicator { cursor: pointer; color: #0066ff; text-decoration: underline; }
        .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000; align-items: center; justify-content: center; }
        .modal.show { display: flex; }
        .modal-content { background: white; border-radius: 8px; padding: 24px; max-width: 600px; width: 90%; max-height: 80vh; overflow-y: auto; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .modal-title { font-size: 16px; font-weight: 600; margin-bottom: 16px; }
        .modal-table { display: grid; grid-template-columns: 1fr 2fr; gap: 12px; }
        .modal-header { display: grid; grid-template-columns: 1fr 2fr; gap: 12px; padding-bottom: 8px; border-bottom: 1px solid #e0e0e0; font-size: 11px; font-weight: 600; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
        .modal-row { display: contents; }
        .modal-row-data { display: grid; grid-template-columns: 1fr 2fr; gap: 12px; padding: 8px 0; border-bottom: 1px solid #f0f0f0; align-items: center; }
        .modal-row-data:last-child { border-bottom: none; }
        .insights-section { display: flex; flex-direction: column; gap: 16px; overflow-y: auto; }
        .insight-box { padding: 14px; background: #f9f9f9; border-radius: 6px; }
        .insight-header { font-size: 11px; font-weight: 600; color: #666; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
        .insight-title { font-size: 14px; font-weight: 600; color: #1a1a1a; margin-bottom: 4px; }
        .insight-desc { font-size: 12px; color: #666; margin-bottom: 8px; }
        .insight-metrics { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; font-size: 11px; }
        .metric-item { }
        .metric-val { font-size: 13px; font-weight: 600; color: #1a1a1a; }
        .metric-label { color: #999; font-size: 10px; }
        .highlight-red { color: #d32f2f; }
        .alert-box { background: #ffebee; border-left: 4px solid #d32f2f; padding: 12px; border-radius: 4px; margin-bottom: 16px; }
        .alert-title { font-weight: 600; color: #d32f2f; margin-bottom: 4px; }
        .alert-desc { font-size: 12px; color: #666; }
        .product-item { display: flex; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid #eee; }
        .product-item:last-child { border-bottom: none; }
        .product-name { font-weight: 600; font-size: 13px; }
        .product-detail { font-size: 11px; color: #999; margin-top: 2px; }
        .product-price { font-weight: 600; font-size: 13px; }
        .product-qty { font-size: 11px; color: #999; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #ddd; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #bbb; }
    </style>
</head>
<body>
    <div class="container">
        <div class="panel" style="flex: 0.2;">
            <div class="panel-header">
                <div class="panel-title">Call Transcript</div>
                <div class="btn-group">
                    <button class="btn-icon" id="muteBtn" title="Mute">🔊</button>
                    <button class="btn-icon" id="recordBtn" title="Record">●</button>
                </div>
            </div>
            <div class="transcript" id="transcript"></div>
            <div style="display: flex; gap: 6px; margin-top: 10px;">
                <input type="text" id="pasteInput" placeholder="Paste text here..." style="flex: 1; padding: 10px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 13px;">
                <button id="submitPasteBtn" style="padding: 10px 14px; background: #0066ff; color: white; border: none; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; white-space: nowrap;">→</button>
            </div>
        </div>

        <div class="panel" style="flex: 0.55;">
            <div class="panel-header">
                <div class="panel-title">Quote Call Data</div>
            </div>

            <div class="form-section">
                <label class="form-label">Company Name</label>
                <div class="form-row">
                    <input type="text" placeholder="Company" value="ABC Supply">
                    <div style="display: flex; align-items: center; gap: 8px;"><span style="color: #0066ff; cursor: pointer;">↗</span></div>
                </div>
            </div>

            <div class="form-section">
                <div class="form-row">
                    <div>
                        <label class="form-label">Associate Name</label>
                        <input type="text" placeholder="Name" value="Reed">
                    </div>
                    <div>
                        <label class="form-label">PO Number</label>
                        <input type="text" placeholder="—">
                    </div>
                </div>
            </div>

            <div class="form-section">
                <div class="form-row full">
                    <label class="form-label">Email</label>
                    <input type="text" placeholder="—">
                </div>
                <div class="form-row full">
                    <label class="form-label">Address</label>
                    <input type="text" placeholder="—">
                </div>
            </div>

            <div class="table-section">
                <label class="form-label" style="margin-bottom: 8px;">Items</label>
                <div class="table-wrapper">
                    <div class="table-header">
                        <div>Item</div>
                        <div>Item ID</div>
                        <div>Catalog Match</div>
                        <div>Cross/Up Sell</div>
                        <div>Manufacturer</div>
                        <div>Qty</div>
                    </div>
                    <div class="table-scroll" id="tbody"></div>
                </div>
            </div>

            <!-- Cross/Upsell Modal -->
            <div class="modal" id="crossSellModal">
                <div class="modal-content">
                    <div class="modal-title" id="modalTitle">Cross/Up Sell Suggestions</div>
                    <div class="modal-header">
                        <div>Item ID</div>
                        <div>Catalog Match</div>
                    </div>
                    <div id="modalBody"></div>
                </div>
            </div>
        </div>

        <div class="panel" style="flex: 0.25;">
            <div class="panel-header">
                <div class="panel-title">Customer Insights</div>
                <button style="background: none; border: none; cursor: pointer; font-size: 16px;">✕</button>
            </div>

            <div class="insights-section">
                <div style="padding: 8px 0; border-bottom: 1px solid #e0e0e0; margin-bottom: 8px;">
                    <div style="font-size: 12px; font-weight: 600; color: #666;">ABC SUPPLY</div>
                </div>

                <div class="alert-box">
                    <div class="alert-title">$ High Price Sensitivity</div>
                    <div class="alert-desc">Negotiates aggressively on price</div>
                    <div style="margin-top: 8px; font-size: 11px;">
                        <span>Customer: 21.2%</span> <span style="margin-left: 12px;">Avg: 29.4%</span> <span style="margin-left: 12px; color: #d32f2f;">-8.3%</span>
                    </div>
                </div>

                <div class="insight-box">
                    <div class="insight-header">KEY METRICS</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                        <div>
                            <div class="metric-label">Total Revenue</div>
                            <div class="metric-val">$5,403,312.42</div>
                        </div>
                        <div>
                            <div class="metric-label">Gross Profit</div>
                            <div class="metric-val">$1,143,517.02</div>
                        </div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px;">
                        <div>
                            <div class="metric-label">Margin</div>
                            <div class="metric-val">21.2%</div>
                        </div>
                        <div>
                            <div class="metric-label">Total Orders</div>
                            <div class="metric-val">636</div>
                        </div>
                    </div>
                    <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid #e0e0e0; font-size: 11px; color: #666;">📅 Last order 99 days ago</div>
                </div>

                <div class="insight-box">
                    <div class="insight-header">MOST COMMON PRODUCTS</div>
                    <div class="product-item">
                        <div>
                            <div class="product-name">18PG</div>
                            <div class="product-detail">1X 18' FLYGEM TRIM</div>
                        </div>
                        <div style="text-align: right;">
                            <div class="product-price">$732,340.80</div>
                            <div class="product-qty">Qty: 14544</div>
                        </div>
                    </div>
                    <div class="product-item">
                        <div>
                            <div class="product-name">110PG</div>
                            <div class="product-detail">1X10 18' FLYGEM TRIM</div>
                        </div>
                        <div style="text-align: right;">
                            <div class="product-price">$474,946.66</div>
                            <div class="product-qty">Qty: 7358</div>
                        </div>
                    </div>
                    <div class="product-item">
                        <div>
                            <div class="product-name">544PCJPC</div>
                            <div class="product-detail">5/4X4 16' J-POCKET CSG WINF PLY...</div>
                        </div>
                        <div style="text-align: right;">
                            <div class="product-price">$471,328.08</div>
                            <div class="product-qty">Qty: 7898</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
<script>
const SAMPLE_RATE = 24000;
let ws, audioContext, processor, source, stream;
let recording = false;
let muted = false;
let matchCount = 0;
let startTime = 0;
const seenIds = new Set();

document.getElementById('recordBtn').onclick = toggle;
document.getElementById('muteBtn').onclick = () => {
    muted = !muted;
    const btn = document.getElementById('muteBtn');
    btn.textContent = muted ? '🔇' : '🔊';
};

async function submitPaste() {
    const input = document.getElementById('pasteInput');
    const text = input.value.trim();
    if (!text) return;

    // Connect if not already connected
    if (!ws || ws.readyState !== 1) {
        try {
            ws = new WebSocket('ws://' + location.host + '/ws');
            await new Promise((resolve, reject) => {
                ws.onopen = resolve;
                ws.onerror = reject;
                setTimeout(() => reject(new Error('Timeout')), 5000);
            });

            // Set up message handler for results
            ws.onmessage = (e) => {
                const msg = JSON.parse(e.data);
                if (msg.type === 'parts') {
                    const tbody = document.getElementById('tbody');
                    msg.items.forEach(item => {
                        if (item && item.item_id && !seenIds.has(item.item_id)) {
                            seenIds.add(item.item_id);
                            const row = document.createElement('div');
                            row.className = 'table-row';

                            const itemName = document.createElement('div');
                            itemName.className = 'item-name';
                            itemName.textContent = item.part_name || '';

                            const itemId = document.createElement('div');
                            itemId.className = 'item-id';
                            itemId.textContent = item.item_id;

                            const desc = document.createElement('div');
                            desc.textContent = item.description || '';

                            const crossSell = document.createElement('div');
                            if (item.cross_sell_suggestions && item.cross_sell_suggestions.length > 0) {
                                const firstSugg = item.cross_sell_suggestions[0];
                                const link = document.createElement('span');
                                link.className = 'cross-sell-indicator';
                                link.textContent = (firstSugg.item_id || '') + (item.cross_sell_suggestions.length > 1 ? ` (+${item.cross_sell_suggestions.length - 1})` : '');
                                link.onclick = () => showCrossSellModal(item.part_name, item.cross_sell_suggestions);
                                crossSell.appendChild(link);
                            }

                            const mfg = document.createElement('div');
                            mfg.textContent = item.manufacturer_name || '';

                            const qty = document.createElement('div');
                            qty.textContent = item.quantity || '1';

                            row.appendChild(itemName);
                            row.appendChild(itemId);
                            row.appendChild(desc);
                            row.appendChild(crossSell);
                            row.appendChild(mfg);
                            row.appendChild(qty);
                            tbody.insertBefore(row, tbody.firstChild);
                        }
                    });
                }
            };
        } catch (err) {
            console.error('Failed to connect:', err);
            return;
        }
    }

    // Append to transcript
    document.getElementById('transcript').textContent += text + ' ';

    // Send to server for extraction
    ws.send(JSON.stringify({
        type: 'paste_transcript',
        text: text
    }));

    // Clear input
    input.value = '';
}

document.getElementById('submitPasteBtn').onclick = submitPaste;
document.getElementById('pasteInput').onkeydown = (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        submitPaste();
    }
};

async function toggle() {
    if (!recording) {
        await startRecording();
    } else {
        stopRecording();
    }
}

function showCrossSellModal(itemName, suggestions) {
    document.getElementById('modalTitle').textContent = `Cross/Up Sell Suggestions for ${itemName}`;
    const modalBody = document.getElementById('modalBody');
    modalBody.innerHTML = '';
    suggestions.forEach(sugg => {
        const row = document.createElement('div');
        row.className = 'modal-row-data';

        const id = document.createElement('div');
        id.className = 'item-id';
        id.textContent = sugg.item_id || '';

        const desc = document.createElement('div');
        desc.textContent = sugg.description || '';

        row.appendChild(id);
        row.appendChild(desc);
        modalBody.appendChild(row);
    });
    document.getElementById('crossSellModal').classList.add('show');
}

// Close modal on click outside
document.getElementById('crossSellModal').onclick = (e) => {
    if (e.target === document.getElementById('crossSellModal')) {
        document.getElementById('crossSellModal').classList.remove('show');
    }
};

async function startRecording() {
    document.getElementById('recordBtn').classList.add('recording');

    try {
        ws = new WebSocket('ws://' + location.host + '/ws');

        await new Promise((resolve, reject) => {
            ws.onopen = resolve;
            ws.onerror = reject;
            setTimeout(() => reject(new Error('Timeout')), 5000);
        });

        ws.onmessage = (e) => {
            const msg = JSON.parse(e.data);
            if (msg.type === 'transcript') {
                document.getElementById('transcript').textContent += msg.text;
            } else if (msg.type === 'parts') {
                const tbody = document.getElementById('tbody');
                msg.items.forEach(item => {
                    if (item && item.item_id && !seenIds.has(item.item_id)) {
                        seenIds.add(item.item_id);
                        const row = document.createElement('div');
                        row.className = 'table-row';

                        // Column 1: Item (part_name from transcript)
                        const itemName = document.createElement('div');
                        itemName.className = 'item-name';
                        itemName.textContent = item.part_name || '';

                        // Column 2: Item ID
                        const itemId = document.createElement('div');
                        itemId.className = 'item-id';
                        itemId.textContent = item.item_id;

                        // Column 3: Catalog Match (description)
                        const desc = document.createElement('div');
                        desc.textContent = item.description || '';

                        // Column 4: Cross/Up Sell
                        const crossSell = document.createElement('div');
                        if (item.cross_sell_suggestions && item.cross_sell_suggestions.length > 0) {
                            const firstSugg = item.cross_sell_suggestions[0];
                            const link = document.createElement('span');
                            link.className = 'cross-sell-indicator';
                            link.textContent = (firstSugg.item_id || '') + (item.cross_sell_suggestions.length > 1 ? ` (+${item.cross_sell_suggestions.length - 1})` : '');
                            link.onclick = () => showCrossSellModal(item.part_name, item.cross_sell_suggestions);
                            crossSell.appendChild(link);
                        }

                        // Column 5: Manufacturer
                        const mfg = document.createElement('div');
                        mfg.textContent = item.manufacturer_name || '';

                        // Column 6: Qty
                        const qty = document.createElement('div');
                        qty.textContent = item.quantity || '1';

                        row.appendChild(itemName);
                        row.appendChild(itemId);
                        row.appendChild(desc);
                        row.appendChild(crossSell);
                        row.appendChild(mfg);
                        row.appendChild(qty);
                        tbody.insertBefore(row, tbody.firstChild);
                    }
                });
            }
        };

        ws.onclose = () => {
            if (recording) stopRecording();
        };

        stream = await navigator.mediaDevices.getUserMedia({
            audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: false }
        });

        audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
        source = audioContext.createMediaStreamSource(stream);
        processor = audioContext.createScriptProcessor(4096, 1, 1);

        processor.onaudioprocess = (e) => {
            if (!recording || muted || ws.readyState !== WebSocket.OPEN) return;
            const float32 = e.inputBuffer.getChannelData(0);
            const int16 = new Int16Array(float32.length);
            for (let i = 0; i < float32.length; i++) {
                int16[i] = Math.max(-32768, Math.min(32767, Math.floor(float32[i] * 32768)));
            }
            const bytes = new Uint8Array(int16.buffer);
            const b64 = btoa(String.fromCharCode.apply(null, bytes));
            ws.send(b64);
        };

        source.connect(processor);
        processor.connect(audioContext.destination);

        recording = true;
        startTime = Date.now();
    } catch (err) {
        console.error(err);
        document.getElementById('recordBtn').classList.remove('recording');
    }
}

function stopRecording() {
    recording = false;
    if (processor) { processor.disconnect(); processor = null; }
    if (source) { source.disconnect(); source = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    if (ws) { ws.close(); ws = null; }
    document.getElementById('recordBtn').classList.remove('recording');
}
</script>
</body>
</html>
"""

@app.get("/")
async def index():
    return HTMLResponse(HTML_PAGE)


@app.websocket("/ws")
async def websocket_endpoint(browser_ws: WebSocket):
    await browser_ws.accept()

    openai_ws = None
    stop_flag = threading.Event()
    transcript_text = []
    seen_item_ids = set()
    send_queue = queue.Queue()

    def receive_from_openai():
        """Thread that receives messages from OpenAI - mirrors realtime.py's receive_messages"""
        try:
            while not stop_flag.is_set():
                try:
                    message = openai_ws.recv()
                    if not message:
                        break
                    data = json.loads(message)
                    event_type = data.get('type', '')

                    if event_type == 'session.created':
                        # Send config exactly like realtime.py
                        config = {
                            "type": "session.update",
                            "session": {
                                "turn_detection": {
                                    "type": "server_vad",
                                    "threshold": 0.5,
                                    "prefix_padding_ms": 300,
                                    "silence_duration_ms": 500
                                },
                                "input_audio_format": "pcm16",
                                "input_audio_transcription": {
                                    "model": "whisper-1"
                                }
                            }
                        }
                        openai_ws.send(json.dumps(config))
                        print('OpenAI session configured')

                    elif event_type == 'conversation.item.input_audio_transcription.delta':
                        delta = data.get('delta', '')
                        if delta:
                            transcript_text.append(delta)
                            send_queue.put({'type': 'transcript', 'text': delta})

                    elif event_type == 'conversation.item.input_audio_transcription.completed':
                        # Transcription complete - extract parts and match
                        full_transcript = ''.join(transcript_text)
                        if full_transcript.strip():
                            try:
                                parts = extract_part_names(full_transcript)
                                if parts:
                                    matched = call_top(parts)
                                    new_items = []
                                    def convert_value(v):
                                        """Convert numpy/pandas types and NaN to JSON-compatible values"""
                                        if v is None:
                                            return None
                                        # Check if it's a numpy or pandas scalar
                                        try:
                                            if np.isscalar(v) and (pd.isna(v) or (isinstance(v, float) and np.isnan(v))):
                                                return None
                                        except (TypeError, ValueError):
                                            pass
                                        # Convert numpy types to Python native types
                                        if hasattr(v, 'item'):  # numpy scalar
                                            return v.item()
                                        return v

                                    for item in matched:
                                        if item is not None:
                                            d = item if isinstance(item, dict) else item.to_dict()
                                            # Convert NaN to None for JSON compatibility
                                            d = {k: convert_value(v) for k, v in d.items()}
                                            # Also convert NaN in cross_sell_suggestions
                                            if 'cross_sell_suggestions' in d and d['cross_sell_suggestions']:
                                                d['cross_sell_suggestions'] = [
                                                    {k: convert_value(v) for k, v in sugg.items()}
                                                    for sugg in d['cross_sell_suggestions']
                                                ]
                                            if d.get('item_id') and d['item_id'] not in seen_item_ids:
                                                seen_item_ids.add(d['item_id'])
                                                new_items.append(d)
                                    if new_items:
                                        send_queue.put({'type': 'parts', 'items': new_items})
                            except Exception as e:
                                print(f'Extract/match error: {e}')

                    elif event_type == 'error':
                        print(f'OpenAI error: {data}')

                except Exception as e:
                    if not stop_flag.is_set():
                        print(f'OpenAI recv error: {e}')
                    break
        finally:
            print('OpenAI receiver stopped')

    async def send_to_browser():
        """Send queued messages to browser"""
        import asyncio
        while not stop_flag.is_set():
            try:
                while not send_queue.empty():
                    msg = send_queue.get_nowait()
                    await browser_ws.send_json(msg)
            except:
                pass
            await asyncio.sleep(0.05)

    try:
        # Connect to OpenAI exactly like realtime.py
        openai_ws = websocket.create_connection(
            WS_URL,
            header=[
                f'Authorization: Bearer {API_KEY}',
                'OpenAI-Beta: realtime=v1'
            ]
        )
        print('Connected to OpenAI')

        # Start receiver thread
        recv_thread = threading.Thread(target=receive_from_openai, daemon=True)
        recv_thread.start()

        # Start sender task
        import asyncio
        send_task = asyncio.create_task(send_to_browser())

        # Receive audio from browser and forward to OpenAI
        while True:
            try:
                data = await browser_ws.receive_text()

                # Try to parse as JSON (paste_transcript)
                try:
                    msg = json.loads(data)
                    if msg.get('type') == 'paste_transcript':
                        text = msg.get('text', '')
                        if text:
                            print(f'[PASTED] {text}')
                            # Extract parts from pasted text
                            parts = extract_part_names(text)
                            if parts:
                                matched = call_top(parts)
                                new_items = []

                                def convert_value(v):
                                    if v is None:
                                        return None
                                    try:
                                        if np.isscalar(v) and (pd.isna(v) or (isinstance(v, float) and np.isnan(v))):
                                            return None
                                    except (TypeError, ValueError):
                                        pass
                                    if hasattr(v, 'item'):
                                        return v.item()
                                    return v

                                for item in matched:
                                    if item is not None:
                                        d = item if isinstance(item, dict) else item.to_dict()
                                        d = {k: convert_value(v) for k, v in d.items()}
                                        if 'cross_sell_suggestions' in d and d['cross_sell_suggestions']:
                                            d['cross_sell_suggestions'] = [
                                                {k: convert_value(v) for k, v in sugg.items()}
                                                for sugg in d['cross_sell_suggestions']
                                            ]
                                        if d.get('item_id') and d['item_id'] not in seen_item_ids:
                                            seen_item_ids.add(d['item_id'])
                                            new_items.append(d)
                                if new_items:
                                    send_queue.put({'type': 'parts', 'items': new_items})
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass

                # Otherwise treat as base64 PCM16 audio
                audio_bytes = base64.b64decode(data)
                encoded = base64.b64encode(audio_bytes).decode('utf-8')
                openai_ws.send(json.dumps({
                    'type': 'input_audio_buffer.append',
                    'audio': encoded
                }))
            except Exception as e:
                # Silently break on disconnect - this is normal when client closes connection
                break

    except Exception as e:
        print(f'WebSocket error: {e}')
    finally:
        stop_flag.set()
        if openai_ws:
            try:
                openai_ws.close()
            except:
                pass
        print('Connection closed')


def extract_part_names(transcript: str):
    """Extract part names and quantities from transcript using GPT"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {
                    "role": "system",
                    "content": 'Extract part names and quantities from the transcript. Return only a JSON array with objects like [{"part_name":"part1","quantity":2},{"part_name":"part2","quantity":1}]. If quantity is not mentioned, default to 1. No backticks or markdown.'
                },
                {"role": "user", "content": transcript}
            ]
        )
        extracted = json.loads(response.choices[0].message.content)
        print(f'[EXTRACTED] {json.dumps(extracted)}')
        return extracted
    except Exception as e:
        print(f'Extract error: {e}')
        return []


def call_top(parts_with_qty: list, k: int = 10):
    """Call the top matching logic with part names and quantities"""
    import random

    # Extract just the part names for embedding
    item_names = [p['part_name'] for p in parts_with_qty]
    quantities = [p.get('quantity', 1) for p in parts_with_qty]

    embs_query = np.array(embed(item_names)).T
    scores = embs_arr @ embs_query
    top_indices = np.argpartition(scores, -k, axis=0)[-k:]
    prompts = [map_results_to_resolution_prompt(top_indices[:, i].tolist(), item_names[i])
               for i in range(len(item_names))]
    responses = [client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'system', 'content': prompt}]
    ).choices[0].message.content for prompt in prompts]
    postprocessed = [try_parse_int(z) for z in responses]
    indices_in_df = [int(top_indices[z, i]) if z is not None else None
                     for i, z in enumerate(postprocessed)]
    matched_df_rows = [df.iloc[z] if z is not None else None for z in indices_in_df]

    # Add quantity and cross/upsell suggestions to each matched row
    result = []
    for part_name, row, qty in zip(item_names, matched_df_rows, quantities):
        if row is not None:
            row_dict = row.to_dict() if hasattr(row, 'to_dict') else row
            row_dict['part_name'] = part_name  # Original transcript part name
            row_dict['quantity'] = qty

            # Generate cross/upsell suggestions: 2-5 random parts
            num_suggestions = random.randint(2, 5)
            random_indices = np.random.choice(len(df), size=num_suggestions, replace=False)
            cross_sell_rows = []
            for idx in random_indices:
                sugg_row = df.iloc[idx]
                sugg_dict = sugg_row.to_dict() if hasattr(sugg_row, 'to_dict') else sugg_row
                cross_sell_rows.append(sugg_dict)
            row_dict['cross_sell_suggestions'] = cross_sell_rows

            result.append(row_dict)
        else:
            result.append(None)

    print(f'[MATCHED] {json.dumps(result, default=str, indent=2)}')
    return result


# ============== ORIGINAL ENDPOINTS ==============

@app.post("/top")
async def top_endpoint(item_names: List[str], k: int = 10):
    # Convert to new format with default quantities
    parts = [{"part_name": name, "quantity": 1} for name in item_names]
    return call_top(parts, k)


def embed(lst: List[str]):
    response = client.embeddings.create(
        model='text-embedding-3-small',
        input=lst,
    )
    return [z.embedding for z in response.data]


def try_parse_int(s: str) -> int | None:
    try:
        return int(s)
    except:
        return None


def map_results_to_resolution_prompt(row_of_ixs: List[int], item_name: str):
    prompt = f"""We are a manufacturing automation business. We extracted that the customer asked for "{item_name}". We cosine similarity matched it to the following top items:

    {df.iloc[row_of_ixs].reset_index(drop=True).reset_index().to_json(orient='records',indent=4)}

    If one of them are what the user asked for, output its index as an int, 0-9. Otherwise, output the string "NONE". E.g., the user could have said "two-and-a-half inch fire lock T", and that would match "2 1/2 FIRELOCK TEE", so if it had index=5, you would output 5. Please don't output an index unless there is a strong semantic match. Other examples: query "three-quarter inch chrome up cut chin" would match "3/4 Chrome Cup 401 Escutcheon" because they sound the same (transcription isn't perfect), query "half-inch gate valve whole part" would match "1/2 BRZ GATE VLV TE FULL PRT". One bug that you run into is matching user query "b" to part name "2 1\\/2 FIRELOCK TEE", which doesn't make sense, don't do that. Another false positive you made was that the user said "any free system 6x2" and you matched "SIGN - BLANK 6 X 2", nice job on the 6x2 but the rest doesn't match enough. Here is another error that you made. The transcript said "I want an antifreeze system, six by two. I want two antifreeze systems, five by seven" and you extracted part_name=antifreeze system quantity=2 and part_name=antifreeze system quantity=1, but really you should have included the 6X2 and 5X7 in the part names. Here's another error, the transcript said "I wonder if I can have...21 over 2 inch, 2000 SS, LF, Aussie, FXG", though that was the customer trying to describe 21/2" 2000SS LF OSY FXG, so as you can see, the transcript will often include commas in between words, because part numbers are compound, and unlike normal language, but you should look past this, and if a series of words with commas in between looks like it should be one part, try to only extract one part. Another error you made was that the item name was "valve", which was very generic and shouldn't have been matched to more specific part names, but it was matched to "VALVE, 05781AJ,VALVE", a very specific part name, which is incorrect, since there are many valves in the dataset, so the general rule is to not match a fully generic word to a part name with some specific identifiers.
    """
    return prompt
