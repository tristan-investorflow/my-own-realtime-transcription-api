import base64
import json
import os
import queue
import threading
import time
import urllib.request
import pyaudio
import websocket

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Set OPENAI_API_KEY environment variable")

WS_URL = 'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01'

CHUNK_SIZE = 1024
RATE = 24000
FORMAT = pyaudio.paInt16

mic_queue = queue.Queue()
extract_queue = queue.Queue()
stop_event = threading.Event()


def mic_callback(in_data, frame_count, time_info, status):
    mic_queue.put(in_data)
    return (None, pyaudio.paContinue)


def send_mic_audio(ws):
    while not stop_event.is_set():
        if not mic_queue.empty():
            chunk = mic_queue.get()
            encoded = base64.b64encode(chunk).decode('utf-8')
            try:
                ws.send(json.dumps({'type': 'input_audio_buffer.append', 'audio': encoded}))
            except Exception as e:
                print(f'Send error: {e}')


def receive_messages(ws):
    while not stop_event.is_set():
        try:
            message = ws.recv()
            if not message:
                break
            data = json.loads(message)
            event_type = data['type']

            if event_type == 'session.created':
                send_session_config(ws)

            elif event_type == 'conversation.item.input_audio_transcription.delta':
                delta = data.get('delta', '')
                with open('transcript.txt', 'a') as f:
                    f.write(delta)
                # print(delta, end='', flush=True)
                # print("IS DELTA")
                extract_queue.put(1)

            elif event_type == 'conversation.item.input_audio_transcription.completed':
                with open('transcript.txt', 'a') as f:
                    f.write('\n')
                # print()
                # print("IS COMPLETED")

        except Exception as e:
            print(f'Receive error: {e}')
            break


def send_session_config(ws):
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
    ws.send(json.dumps(config))


def extract_parts():
    while not stop_event.is_set():
        try:
            extract_queue.get(timeout=1)
        except queue.Empty:
            continue

        # Drain the queue
        while not extract_queue.empty():
            try:
                extract_queue.get_nowait()
            except queue.Empty:
                break

        # Read full transcript
        try:
            with open('transcript.txt', 'r') as f:
                transcript = f.read()
        except FileNotFoundError:
            continue

        if not transcript.strip():
            continue

        # Query gpt-4o-mini for part names
        req_body = json.dumps({
            "model": "gpt-4o-mini-2024-07-18",
            "messages": [
                {
                    "role": "system",
                    "content": """We are a manufacturer taking orders over the phone. Extract names of parts we manufacture from the transcript. Return only a JSON array of strings, no backticks or markdown. Example: [\"part1\", \"part2\"]. Examples of part names we have follow.
A7084 1-1/2 S10 304SS FLG NPL
2X2X1/2   6000 FS THRD TEE
2 FIRELOCK 90 ELBOW
5 FIRELOCK 90 ELBOW
PSM-1/6 REMOTE ACCESS
ANGLE WALL POST KIT FIG 551
OIL BREATHER
1/2" LIQUID RELIEF VALVE
POP-SAFETY VALVE 1/4 NPT
2 1/2 FIRELOCK TEE
ANGLE WALL POST KIT FIG 551 B
QUICK CK BARBED ADPTR W/TUBING
ANGLE WALL POST KIT FIG 551 B
8 FIRELOCK 45 ELBOW
2 FLEX GRV COUPLING"""
                },
                {
                    "role": "user",
                    "content": transcript
                }
            ]
        }).encode('utf-8')

        req = urllib.request.Request(
            'https://api.openai.com/v1/chat/completions',
            data=req_body,
            headers={
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json'
            }
        )

        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                parts_str = result['choices'][0]['message']['content']
                print(f'\n[PARTS] {parts_str}')

                # Parse the JSON array and send to local API
                parts_list = json.loads(parts_str)
                if parts_list:
                    top_req = urllib.request.Request(
                        'http://127.0.0.1:8000/top',
                        data=json.dumps(parts_list).encode('utf-8'),
                        headers={'Content-Type': 'application/json'}
                    )
                    with urllib.request.urlopen(top_req) as top_resp:
                        top_result = json.loads(top_resp.read().decode('utf-8'))
                        print(f'[TOP] {json.dumps(top_result, indent=2)}\n')
        except Exception as e:
            print(f'Extract error: {e}')


def main():
    p = pyaudio.PyAudio()
    mic_stream = p.open(
        format=FORMAT,
        channels=1,
        rate=RATE,
        input=True,
        stream_callback=mic_callback,
        frames_per_buffer=CHUNK_SIZE
    )

    try:
        ws = websocket.create_connection(
            WS_URL,
            header=[
                f'Authorization: Bearer {API_KEY}',
                'OpenAI-Beta: realtime=v1'
            ]
        )
        print('Connected. Transcribing to transcript.txt...')

        recv_thread = threading.Thread(target=receive_messages, args=(ws,))
        recv_thread.start()

        send_thread = threading.Thread(target=send_mic_audio, args=(ws,))
        send_thread.start()

        extract_thread = threading.Thread(target=extract_parts)
        extract_thread.start()

        mic_stream.start_stream()

        while not stop_event.is_set():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print('\nShutting down...')
        stop_event.set()

    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        p.terminate()
        try:
            ws.close()
        except:
            pass


if __name__ == '__main__':
    main()
