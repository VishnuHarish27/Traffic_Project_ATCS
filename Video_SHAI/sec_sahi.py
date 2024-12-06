import cv2
import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, Response
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
from threading import Thread
import torch
import time
from collections import deque

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
DATABASE = 'traffic_sahi.db'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
BATCH_SIZE = 1  # Process multiple frames at once
FRAME_SKIP = 5 # Skip frames to reduce processing load
MAX_QUEUE_SIZE = 32  # Maximum frames to keep in queue
PROCESSING_TIMES = deque(maxlen=50)  # Store last 50 frame processing times

# Enable CUDA optimization settings
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# Ensure upload and processed folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define regions with their vertices and weightages
REGIONS = {
    'R1': {
        'vertices': np.array([(301,751), (883,792), (1218,312), (967,306)], dtype=np.int32),
        'weight': 0.25,
        'color': (0, 255, 0)
    },
    'R2': {
        'vertices': np.array([(910,780), (1438,793), (1420,320), (1230,312)], dtype=np.int32),
        'weight': 0.25,
        'color': (255, 0, 0)
    },
    'R3': {
        'vertices': np.array([(998,296), (1411,306), (1425,170), (1354,167)], dtype=np.int32),
        'weight': 0.50,
        'color': (0, 0, 255)
    }
}

NUM_BLOCKS = 5

# Initialize models with CUDA if available
DEVICE = 'cpu'
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

# Initialize database
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
            r3_vehicle_count INTEGER NOT NULL,
            r3_density REAL NOT NULL,
            weighted_vehicle_count REAL NOT NULL,
            weighted_density REAL NOT NULL,
            processing_time REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db(DATABASE)
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
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

    def analyze_region(self, boxes, region_name):
        vertices = REGIONS[region_name]['vertices']
        blocks = self.region_blocks[region_name]
        filled_blocks = set()
        vehicles_in_region = 0

        for box in boxes:
            x1, y1, x2, y2 = box
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

        if detections is not None:
            boxes = detections.xyxy

            # Draw detection boxes
            for box, conf, class_id in zip(boxes, detections.confidence, detections.class_id):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                label = f'Vehicle {conf:.2f}'
                cv2.putText(processed_frame, label, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            # Process regions in parallel
            def process_region(region_name):
                vehicles, density, filled_blocks = self.analyze_region(boxes, region_name)
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

            # Add metrics to frame
            cv2.putText(processed_frame,
                      f'Total Vehicles: {total_vehicles}, Weighted Density: {weighted_density:.1f}%',
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frame_time = time.time() - frame_start_time
        PROCESSING_TIMES.append(frame_time)
        avg_time = sum(PROCESSING_TIMES) / len(PROCESSING_TIMES)
        print(f"Frame processing time: {frame_time*1000:.2f}ms, "
              f"Average: {avg_time*1000:.2f}ms")

        return processed_frame, region_metrics, total_vehicles, weighted_density, frame_time

def process_video(video_path, processed_path):
    if detection_model is None and yolo_model is None:
        return

    frame_processor = FrameProcessor()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    batch_frames = []
    last_save_time = time.time()

    def save_to_database(metrics, total_vehicles, weighted_density, processing_time):
        try:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('''INSERT INTO traffic_data 
                           (timestamp, r1_vehicle_count, r1_density,
                            r2_vehicle_count, r2_density,
                            r3_vehicle_count, r3_density,
                            weighted_vehicle_count, weighted_density,
                            processing_time)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (timestamp,
                         metrics['R1']['vehicles'], metrics['R1']['density'],
                         metrics['R2']['vehicles'], metrics['R2']['density'],
                         metrics['R3']['vehicles'], metrics['R3']['density'],
                         total_vehicles, weighted_density, processing_time))
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"Database error: {e}")

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
                        boxes = []
                        confidences = []
                        class_ids = []
                        vehicle_classes = [2, 3, 5, 7]

                        for pred in result.object_prediction_list:
                            if pred.category.id in vehicle_classes:
                                boxes.append(pred.bbox.to_xyxy())
                                confidences.append(pred.score.value)
                                class_ids.append(pred.category.id)

                        if boxes:
                            detections = sv.Detections(
                                xyxy=np.array(boxes),
                                confidence=np.array(confidences),
                                class_id=np.array(class_ids)
                            )

                            processed_frame, metrics, total_vehicles, weighted_density, proc_time = \
                                frame_processor.process_detections(frame, detections)

                            current_time = time.time()
                            if current_time - last_save_time >= 1.0:
                                Thread(target=save_to_database,
                                      args=(metrics, total_vehicles, weighted_density, proc_time)).start()
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

# Flask routes remain the same
@app.route('/')
def index():
    filename = request.args.get('filename')
    return render_template('index.html', filename=filename)

@app.route('/analytics')
def analytics():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''SELECT timestamp, 
                                r1_vehicle_count, r1_density,
                                r2_vehicle_count, r2_density,
                                r3_vehicle_count, r3_density,
                                weighted_vehicle_count, weighted_density 
                         FROM traffic_data 
                         ORDER BY id DESC LIMIT 100''')
        data = cursor.fetchall()
        conn.close()
        return render_template('analytics.html', data=data[::-1])
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return f"Database error: {e}", 500

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM traffic_data')
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"Error clearing database: {e}")

        processed_filename = f'processed_{filename}'
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)

        if os.path.exists(processed_path):
            os.remove(processed_path)

        thread = Thread(target=lambda: list(process_video(filepath, processed_path)))
        thread.start()

        return redirect(url_for('index', filename=filename))
    return redirect(request.url)

@app.route('/video_feed')
def video_feed():
    filename = request.args.get('filename')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    processed_filename = f'processed_{filename}'
    processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
    return Response(process_video(filepath, processed_path),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
