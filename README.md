# Scalable-NLP-Pipeline-for-Sentiment-and-Readability-Analysis

Project Overview
This project implements a robust NLP pipeline for extracting, preprocessing, and analyzing textual data. The system supports batch processing of large datasets, computes sentiment scores, and evaluates text readability using advanced linguistic metrics.

Features
Text Extraction: Dynamically scrapes and processes web-sourced textual data.
Preprocessing: Tokenizes text and removes stop words for cleaner analysis.
Sentiment Analysis: Leverages custom dictionaries for polarity and subjectivity scoring.
Readability Metrics: Calculates metrics like Fog Index, Average Sentence Length, and Complex Word Ratio.
Scalability: Supports batch processing and domain-specific analysis through modular design.

Tools and Technologies
Programming Language: Python

Libraries Used:
NLTK (Natural Language Toolkit)
BeautifulSoup (Web Scraping)
Pandas, NumPy (Data Handling and Computation)

How to Use
Clone this repository to your local machine.
Install dependencies using pip install -r requirements.txt.
Run the Jupyter Notebook to process and analyze your text data.

Results
The pipeline provides insights into sentiment and readability, assisting in evaluating text complexity and emotional tone efficiently.


torch
opencv-python
flask


import torch
import cv2
import threading
from flask import Flask, Response
import time

app = Flask(__name__)

# Load YOLOv7 model
model = torch.hub.load('yolov7', 'custom', 'yolov7.pt', source='local')
model.eval()

streams = {}
lock = threading.Lock()

def process_stream(stream_id, source):
    cap = cv2.VideoCapture(source)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        processed_frame = results.render()[0]

        with lock:
            streams[stream_id] = processed_frame

        time.sleep(0.01)

    cap.release()

def generate_stream(stream_id):
    while True:
        with lock:
            frame = streams.get(stream_id, None)
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.05)

@app.route('/stream/<int:stream_id>')
def stream(stream_id):
    return Response(generate_stream(stream_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    sources = {
        0: 0,  # local webcam
        1: 'http://192.168.0.101:8080/video'  # replace with real IP camera
    }

    for stream_id, src in sources.items():
        t = threading.Thread(target=process_stream, args=(stream_id, src))
        t.daemon = True
        t.start()

    app.run(host='0.0.0.0', port=5000, threaded=True)

https://docs.google.com/document/d/11AGcvljvg2Q9Rb0i7Kni2GoWC9eg5Bmpj8BZKdJWen4/edit?usp=drivesdk

