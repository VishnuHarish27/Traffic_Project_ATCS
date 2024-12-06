import cv2
import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import sqlite3
from datetime import datetime
import supervision as sv
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
from threading import Thread, Timer
import torch
import time
from collections import deque
import json
app = Flask(__name__)

# Configuration
PROCESSED_FOLDER = 'static/processed/'
DATABASE = 'traffic_sahi000.db'
BATCH_SIZE = 1
FRAME_SKIP = 5
MAX_QUEUE_SIZE = 32
PROCESSING_TIMES = deque(maxlen=50)
RTSP_URL = "http://109.236.111.203/mjpg/video.mjpg"
JSON_UPDATE_INTERVAL = 5

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Updated regions with only R1 and R2
REGIONS = {
    'R1': {
        'vertices': np.array([(561,984), (393,609), (732,582), (1105,896)], dtype=np.int32),
        'weight': 0.5,  # Updated weight to split between two regions
        'color': (0, 255, 0)
    },
    'R2': {
        'vertices': np.array([(1116,880), (752,577), (1077,555), (1579,809)], dtype=np.int32),
        'weight': 0.5,  # Updated weight to split between two regions
        'color': (255, 0, 0)
    }
}
NUM_BLOCKS = 5

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = mp.cpu_count()

try:
    local_model_path = 'models/yolov8n.pt'
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=local_model_path,
        confidence_threshold=0.3,
        device=DEVICE
    )
    yolo_model = YOLO(local_model_path)
    yolo_model.to(DEVICE)
except Exception as e:
    print(f"Error loading models: {e}")
    detection_model = None
    yolo_model = None

# Modified database initialization
def init_db(database_name):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS traffic_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            r1_vehicle_count INTEGER NOT NULL,
            r1_density REAL NOT NULL,
            r2_vehicle_count INTEGER NOT NULL,
            r2_density REAL NOT NULL,
            weighted_vehicle_count REAL NOT NULL,
            weighted_density REAL NOT NULL,
            vdc1 BOOLEAN NOT NULL,
            processing_time REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db(DATABASE)

class FrameProcessor:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.result_queue = queue.Queue()
        self.thread_pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
        self.process_pool = ProcessPoolExecutor(max_workers=NUM_WORKERS)
        self.region_blocks = {
            region: self.divide_region_into_blocks(data['vertices'], NUM_BLOCKS)
            for region, data in REGIONS.items()
        }

    @staticmethod
    def is_point_in_polygon(x, y, vertices):
        return cv2.pointPolygonTest(vertices, (x, y), False) >= 0

    @staticmethod
    def divide_region_into_blocks(vertices, num_blocks):
        min_y = min(vertices[:, 1])
        max_y = max(vertices[:, 1])
        block_height = (max_y - min_y) / num_blocks
        return [(min_y + i * block_height, min_y + (i + 1) * block_height) for i in range(num_blocks)]

    def process_batch(self, frames):
        try:
            batch_start_time = time.time()
            batch_results = []

            # Process frames in parallel using SAHI
            def process_single_frame(frame):
                return get_sliced_prediction(
                    frame,
                    detection_model,
                    slice_height=640,
                    slice_width=640,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                    perform_standard_pred=True,
                    postprocess_type="NMM",
                    postprocess_match_threshold=0.5,
                    verbose=0
                )

            # Use thread pool for parallel processing
            batch_results = list(self.thread_pool.map(process_single_frame, frames))

            batch_time = time.time() - batch_start_time
            print(f"Batch processing time: {batch_time*1000:.2f}ms, "
                  f"Per frame: {(batch_time/len(frames))*1000:.2f}ms")

            return batch_results
        except Exception as e:
            print(f"Error in batch processing: {e}")
            return None

    def analyze_region(self, detections, region_name):
        vertices = REGIONS[region_name]['vertices']
        blocks = self.region_blocks[region_name]
        filled_blocks = set()
        vehicles_in_region = 0

        for pred in detections.object_prediction_list:
            x1, y1, x2, y2 = pred.bbox.to_xyxy()
            x_center = int((x1 + x2) / 2.0)
            y_center = int((y1 + y2) / 2.0)

            if self.is_point_in_polygon(x_center, y_center, vertices):
                vehicles_in_region += 1
                for idx, (y_min, y_max) in enumerate(blocks):
                    if y_min <= y_center <= y_max:
                        filled_blocks.add(idx)
                        break

        density = (len(filled_blocks) / NUM_BLOCKS) * 100
        return vehicles_in_region, density, filled_blocks

    def draw_region_blocks(self, frame, region_name, filled_blocks):
        vertices = REGIONS[region_name]['vertices']
        color = REGIONS[region_name]['color']
        blocks = self.region_blocks[region_name]

        # Draw region outline
        cv2.polylines(frame, [vertices], isClosed=True, color=color, thickness=2)

        # Draw blocks
        for idx, (y_min, y_max) in enumerate(blocks):
            block_color = color if idx in filled_blocks else (128, 128, 128)
            pts = np.array([
                [vertices[0][0], int(y_min)],
                [vertices[1][0], int(y_min)],
                [vertices[1][0], int(y_max)],
                [vertices[0][0], int(y_max)]
            ], np.int32)
            cv2.polylines(frame, [pts], isClosed=True, color=block_color, thickness=1)

    def process_detections(self, frame, detections):
        frame_start_time = time.time()
        processed_frame = frame.copy()
        region_metrics = {}
        total_vehicles = 0
        vdc1 = False  # VDC1 flag

        if detections is not None:
            # Process regions in parallel
            def process_region(region_name):
                vehicles, density, filled_blocks = self.analyze_region(detections, region_name)
                return region_name, vehicles, density, filled_blocks

            # Use thread pool for parallel region processing
            region_results = list(self.thread_pool.map(process_region, REGIONS.keys()))

            for region_name, vehicles, density, filled_blocks in region_results:
                region_metrics[region_name] = {'vehicles': vehicles, 'density': density}
                total_vehicles += vehicles
                self.draw_region_blocks(processed_frame, region_name, filled_blocks)

                # Add region metrics to frame
                text_position = REGIONS[region_name]['vertices'][0]
                cv2.putText(processed_frame,
                          f'{region_name}: {vehicles} vehicles, {density:.1f}%',
                          (text_position[0], text_position[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, REGIONS[region_name]['color'], 2)

            # Calculate weighted density
            weighted_density = sum(
                region_metrics[region]['density'] * REGIONS[region]['weight']
                for region in REGIONS.keys()
            )

            # Set the VDC1 flag based on weighted density
            vdc1 = weighted_density > 50

            # Add metrics to frame
            cv2.putText(processed_frame,
                      f'Total Vehicles: {total_vehicles}, Weighted Density: {weighted_density:.1f}%',
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frame_time = time.time() - frame_start_time
        PROCESSING_TIMES.append(frame_time)
        avg_time = sum(PROCESSING_TIMES) / len(PROCESSING_TIMES)
        print(f"Frame processing time: {frame_time*1000:.2f}ms, "
              f"Average: {avg_time*1000:.2f}ms")

        return processed_frame, region_metrics, total_vehicles, weighted_density, vdc1, frame_time

def save_to_database(region_metrics, total_vehicles, weighted_density, vdc1, processing_time):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''INSERT INTO traffic_data 
                       (timestamp, r1_vehicle_count, r1_density,
                        r2_vehicle_count, r2_density,
                        weighted_vehicle_count, weighted_density, vdc1,
                        processing_time)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (timestamp,
                     region_metrics['R1']['vehicles'], region_metrics['R1']['density'],
                     region_metrics['R2']['vehicles'], region_metrics['R2']['density'],
                     total_vehicles, weighted_density, vdc1, processing_time))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"Database error: {e}")

def save_to_json():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''SELECT weighted_density, vdc1 FROM traffic_data ORDER BY id DESC LIMIT 1''')
        latest_data = cursor.fetchone()
        conn.close()

        if latest_data:
            weighted_density, vdc1 = latest_data
            data = {'weighted_density': weighted_density, 'vdc1': int(vdc1)}
            with open('static/traffic_data.json', 'w') as f:
                json.dump(data, f)
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        Timer(JSON_UPDATE_INTERVAL, save_to_json).start()

def process_video_stream():
    if detection_model is None and yolo_model is None:
        return

    frame_processor = FrameProcessor()
    cap = cv2.VideoCapture(RTSP_URL)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    processed_path = os.path.join(PROCESSED_FOLDER, 'processed_video.mp4')
    out = cv2.VideoWriter(processed_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    batch_frames = []
    last_save_time = time.time()

    save_to_json()  # Start the JSON file update loop

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        batch_frames.append(frame)

        if len(batch_frames) >= BATCH_SIZE:
            batch_results = frame_processor.process_batch(batch_frames)

            if batch_results:
                for frame, result in zip(batch_frames, batch_results):
                    if result is not None:
                        processed_frame, region_metrics, total_vehicles, weighted_density, vdc1, proc_time = \
                            frame_processor.process_detections(frame, result)

                        current_time = time.time()
                        if current_time - last_save_time >= 1.0:
                            Thread(target=save_to_database,
                                  args=(region_metrics, total_vehicles, weighted_density, vdc1, proc_time)).start()
                            last_save_time = current_time

                        out.write(processed_frame)

                        ret, buffer = cv2.imencode('.jpg', processed_frame)
                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                  b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            batch_frames = []

    cap.release()
    out.release()
    frame_processor.thread_pool.shutdown()
    frame_processor.process_pool.shutdown()

@app.route('/VehicleDetect/<string:device_id>', methods=['GET'])
def vehicle_detect(device_id):
    try:
        with open('static/traffic_data.json', 'r') as f:
            data = json.load(f)
        response = {'DeviceID': device_id, 'VDC1': data['vdc1']}
        return jsonify(response)
    except FileNotFoundError:
        return jsonify({'error': 'JSON file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analytics')
def analytics():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''SELECT timestamp, 
                                r1_vehicle_count, r1_density,
                                r2_vehicle_count, r2_density,
                                weighted_vehicle_count, weighted_density, vdc1 
                         FROM traffic_data 
                         ORDER BY id DESC LIMIT 100''')
        data = cursor.fetchall()
        conn.close()
        return render_template('analytics.html', data=data[::-1])
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return f"Database error: {e}", 500


@app.route('/video_feed')
def video_feed():
    return Response(process_video_stream(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

