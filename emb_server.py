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
    embs = json.load(f)
embs_arr = np.array(embs)
df = pd.read_csv('df_subset.csv')

app = FastAPI()

# ============== WEB UI ==============

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Realtime Transcription</title>
    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; margin: 0; padding: 20px; display: flex; gap: 20px; height: 100vh; }
        .panel { display: flex; flex-direction: column; }
        #left { flex: 0 0 50%; }
        #right { flex: 1; }
        h3 { margin: 0 0 10px 0; }
        button { padding: 12px 24px; font-size: 16px; cursor: pointer; margin-bottom: 10px; }
        button.recording { background: #ff4444; color: white; }
        #transcript { flex: 1; border: 1px solid #ccc; padding: 10px; overflow-y: auto; white-space: pre-wrap; font-size: 14px; background: #fafafa; }
        #status { font-size: 12px; color: #666; margin-bottom: 5px; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        th { background: #f0f0f0; }
        #tableContainer { flex: 1; overflow-y: auto; }
    </style>
</head>
<body>
    <div id="left" class="panel">
        <button id="btn">Start Recording</button>
        <div id="status">Ready</div>
        <h3>Transcript</h3>
        <div id="transcript"></div>
    </div>
    <div id="right" class="panel">
        <h3>Matched Parts</h3>
        <div id="tableContainer">
            <table>
                <thead><tr><th>Item ID</th><th>Description</th><th>Manufacturer</th></tr></thead>
                <tbody id="tbody"></tbody>
            </table>
        </div>
    </div>
<script>
const SAMPLE_RATE = 24000;
let ws, audioContext, processor, source, stream;
let recording = false;
const seenIds = new Set();

document.getElementById('btn').onclick = toggle;

async function toggle() {
    if (!recording) {
        await startRecording();
    } else {
        stopRecording();
    }
}

async function startRecording() {
    document.getElementById('transcript').textContent = '';
    document.getElementById('tbody').innerHTML = '';
    seenIds.clear();
    document.getElementById('status').textContent = 'Connecting...';

    // Connect WebSocket first
    ws = new WebSocket('ws://' + location.host + '/ws');

    await new Promise((resolve, reject) => {
        ws.onopen = resolve;
        ws.onerror = reject;
        setTimeout(() => reject(new Error('Connection timeout')), 5000);
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
                    const row = tbody.insertRow();
                    row.insertCell(0).textContent = item.item_id;
                    row.insertCell(1).textContent = item.description || '';
                    row.insertCell(2).textContent = item.manufacturer_name || '';
                }
            });
        }
    };

    ws.onclose = () => {
        if (recording) stopRecording();
        document.getElementById('status').textContent = 'Disconnected';
    };

    // Get microphone with specific sample rate
    stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: false }
    });

    audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
    source = audioContext.createMediaStreamSource(stream);

    // ScriptProcessor to get raw samples (deprecated but widely supported)
    processor = audioContext.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (e) => {
        if (!recording || ws.readyState !== WebSocket.OPEN) return;

        const float32 = e.inputBuffer.getChannelData(0);
        // Convert Float32 [-1,1] to Int16 PCM
        const int16 = new Int16Array(float32.length);
        for (let i = 0; i < float32.length; i++) {
            int16[i] = Math.max(-32768, Math.min(32767, Math.floor(float32[i] * 32768)));
        }
        // Send as base64
        const bytes = new Uint8Array(int16.buffer);
        const b64 = btoa(String.fromCharCode.apply(null, bytes));
        ws.send(b64);
    };

    source.connect(processor);
    processor.connect(audioContext.destination);

    recording = true;
    document.getElementById('btn').textContent = 'Stop Recording';
    document.getElementById('btn').classList.add('recording');
    document.getElementById('status').textContent = 'Recording... speak now';
}

function stopRecording() {
    recording = false;
    if (processor) { processor.disconnect(); processor = null; }
    if (source) { source.disconnect(); source = null; }
    if (audioContext) { audioContext.close(); audioContext = null; }
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    if (ws) { ws.close(); ws = null; }
    document.getElementById('btn').textContent = 'Start Recording';
    document.getElementById('btn').classList.remove('recording');
    document.getElementById('status').textContent = 'Stopped';
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
                                    for item in matched:
                                        if item is not None:
                                            d = item.to_dict() if hasattr(item, 'to_dict') else item
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
                # Data is base64 PCM16 audio
                audio_bytes = base64.b64decode(data)
                encoded = base64.b64encode(audio_bytes).decode('utf-8')
                openai_ws.send(json.dumps({
                    'type': 'input_audio_buffer.append',
                    'audio': encoded
                }))
            except Exception as e:
                print(f'Browser receive error: {e}')
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


def extract_part_names(transcript: str) -> List[str]:
    """Extract part names from transcript using GPT"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {
                    "role": "system",
                    "content": 'Extract part names from the transcript. Return only a JSON array of strings, no backticks. Example: ["part1", "part2"]'
                },
                {"role": "user", "content": transcript}
            ]
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f'Extract error: {e}')
        return []


def call_top(item_names: List[str], k: int = 10):
    """Call the top matching logic - same as original"""
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
    return matched_df_rows


# ============== ORIGINAL ENDPOINTS ==============

@app.post("/top")
async def top_endpoint(item_names: List[str], k: int = 10):
    return call_top(item_names, k)


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

    If one of them are what the user asked for, output its index as an int, 0-9. Otherwise, output the string "NONE". E.g., the user could have said "two-and-a-half inch fire lock T", and that would match "2 1/2 FIRELOCK TEE", so if it had index=5, you would output 5. Please don't output an index unless there is a strong semantic match. Other examples: query "three-quarter inch chrome up cut chin" would match "3/4 Chrome Cup 401 Escutcheon" because they sound the same (transcription isn't perfect), query "half-inch gate valve whole part" would match "1/2 BRZ GATE VLV TE FULL PRT". One bug that you run into is matching user query "b" to part name "2 1\\/2 FIRELOCK TEE", which doesn't make sense, don't do that.
    """
    return prompt
