import os
import sys
import time
from flask import Flask, render_template, request, jsonify
from mss import mss
import cv2
import numpy as np
import threading
import google.generativeai as genai

app = Flask(__name__)

API_KEY = 'AIzaSyDk9S7ZB34LLqowBYsea1zDGMAd0CYOwb8'
genai.configure(api_key=API_KEY)

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="learnlm-1.5-pro-experimental",
    generation_config=generation_config,
)

recorded_video_path = os.path.join(os.getcwd(), 'screen_recording.mp4')
is_recording = False
recording_thread = None
analysis_result = ''

class RecordingThread(threading.Thread):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.is_recording = True

    def run(self):
        with mss() as sct:
            monitor = sct.monitors[0]
            width, height = monitor["width"], monitor["height"]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.path, fourcc, 15.0, (width, height))

            while self.is_recording:
                screenshot = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
                out.write(frame)
                time.sleep(0.066)  # Adjust for desired frame rate

            out.release()

    def stop(self):
        self.is_recording = False

@app.route('/')
def index():
    return render_template('index.html', result=analysis_result)

@app.route('/start_recording')
def start_recording():
    global is_recording, recording_thread, analysis_result
    if not is_recording:
        is_recording = True
        analysis_result = ''
        recording_thread = RecordingThread(recorded_video_path)
        recording_thread.start()
        return jsonify({'status': 'Recording started'})
    else:
        return jsonify({'status': 'Already recording'})

@app.route('/stop_recording')
def stop_recording():
    global is_recording, recording_thread, analysis_result
    if is_recording:
        recording_thread.stop()
        recording_thread.join()
        is_recording = False
        return jsonify({'status': 'Recording stopped'})
    else:
        return jsonify({'status': 'Not recording'})

@app.route('/analyze_recording', methods=['POST'])
def analyze_recording():
    global analysis_result
    custom_instructions = request.form.get('instructions', '')
    if not custom_instructions:
        custom_instructions = "Analyze the contents of this screen recording and provide a detailed description."

    file = genai.upload_file(recorded_video_path, mime_type="video/mp4")
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")

    while file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(5)
        file = genai.get_file(file.name)
    if file.state.name != "ACTIVE":
        raise Exception(f"File processing failed: {file.name}")

    chat_session = model.start_chat(
        history=[{"role": "user", "parts": [file]}]
    )

    response = chat_session.send_message(custom_instructions)
    analysis_result = response.text

    return jsonify({'result': analysis_result})

if __name__ == '__main__':
    app.run(debug=True)