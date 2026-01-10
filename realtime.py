import base64
import json
import os
import queue
import threading
import time
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
                print(delta, end='', flush=True)

            elif event_type == 'conversation.item.input_audio_transcription.completed':
                with open('transcript.txt', 'a') as f:
                    f.write('\n')
                print()

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
