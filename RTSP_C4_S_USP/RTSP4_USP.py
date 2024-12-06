import cv2
import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from ultralytics import YOLO
import sqlite3
from datetime import datetime
import supervision as sv
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import queue
from threading import Thread, Timer
import torch
import time
from collections import deque
import json

app = Flask(__name__)

# Configuration
PROCESSED_FOLDER = 'static/processed/'
DATABASE = 'traffic_multi.db'
FRAME_SKIP = 5
MAX_QUEUE_SIZE = 32
PROCESSING_TIMES = {f'cam{i+1}': deque(maxlen=50) for i in range(4)}
IMAGE_SAVE_INTERVAL = 2  # Save images every 10 seconds
PROCESS_DURATION = 2  # Process each stream for 2 seconds

# RTSP URLs for each camera
RTSP_URLS = {
    'cam1': "http://takemotopiano.aa1.netvolante.jp:8190/nphMotionJpeg?Resolution=640x480&Quality=Standard&Framerate=30",
    'cam2': "http://195.196.36.242/mjpg/video.mjpg",
    'cam3': "http://195.196.36.242/mjpg/video.mjpg",
    'cam4': "http://camera.buffalotrace.com/mjpg/video.mjpg"
}

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

os.makedirs(PROCESSED_FOLDER, exist_ok=True)
for cam in RTSP_URLS.keys():
    os.makedirs(os.path.join(PROCESSED_FOLDER, cam), exist_ok=True)

# Define regions for each camera
REGIONS = {
    'cam1': {
        'R1': {
            'vertices': np.array([(59, 378), (479, 410), (474, 185), (311, 177)], dtype=np.int32),
            'weight': 0.5,
            'color': (0, 255, 0)
        },
        'R2': {
            'vertices': np.array([(315, 172), (473, 180), (464, 87), (435, 82)], dtype=np.int32),
            'weight': 0.5,
            'color': (255, 0, 0)
        }
    },
    'cam2': {
        'R1': {
            'vertices': np.array([(386, 444), (238, 302), (361, 276), (638, 385)], dtype=np.int32),
            'weight': 0.5,
            'color': (0, 255, 0)
        },
        'R2': {
            'vertices': np.array([(238, 298), (355, 272), (214, 188), (175, 202)], dtype=np.int32),
            'weight': 0.5,
            'color': (255, 0, 0)
        }
    },
    'cam3': {
        'R1': {
            'vertices': np.array([(537, 448), (639, 262), (368, 175), (186, 286)], dtype=np.int32),
            'weight': 0.5,
            'color': (0, 255, 0)
        },
        'R2': {
            'vertices': np.array([(186, 279), (360, 173), (89, 116), (4, 190)], dtype=np.int32),
            'weight': 0.5,
            'color': (255, 0, 0)
        }
    },
    'cam4': {
        'R1': {
            'vertices': np.array([(53,412), (160,208), (318,205), (389,408)], dtype=np.int32),
            'weight': 0.5,
            'color': (0, 255, 0)
        },
        'R2': {
            'vertices': np.array([(401,406), (335,205), (526,194), (737,402)], dtype=np.int32),
            'weight': 0.5,
            'color': (255, 0, 0)
        }
    }
}


NUM_BLOCKS = 5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = mp.cpu_count()
VEHICLE_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
# Load models
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

def init_db(database_name):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()

    # Create tables for each camera
    for cam_id in range(1, 5):
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS traffic_data_cam{cam_id} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                r1_vehicle_count INTEGER NOT NULL,
                r1_density REAL NOT NULL,
                r2_vehicle_count INTEGER NOT NULL,
                r2_density REAL NOT NULL,
                weighted_vehicle_count REAL NOT NULL,
                weighted_density REAL NOT NULL,
                vdc{cam_id} BOOLEAN NOT NULL,
                processing_time REAL NOT NULL
            )
        ''')
    conn.commit()
    conn.close()

init_db(DATABASE)

class CameraProcessor:
    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self.result_queue = queue.Queue()
        self.thread_pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
        self.region_blocks = self.initialize_region_blocks()
        self.last_save_time = time.time()

    def initialize_region_blocks(self):
        blocks = {}
        for region_name in ['R1', 'R2']:
            region_data = REGIONS[f'cam{self.cam_id}'][region_name]
            blocks[region_name] = self.divide_region_into_blocks(region_data['vertices'], NUM_BLOCKS)
        return blocks

    @staticmethod
    def is_point_in_polygon(x, y, vertices):
        return cv2.pointPolygonTest(vertices, (x, y), False) >= 0

    @staticmethod
    def divide_region_into_blocks(vertices, num_blocks):
        min_y = min(vertices[:, 1])
        max_y = max(vertices[:, 1])
        block_height = (max_y - min_y) / num_blocks
        return [(min_y + i * block_height, min_y + (i + 1) * block_height) for i in range(num_blocks)]

    def process_frame(self, frame):
        try:
            result = get_sliced_prediction(
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
            return result
        except Exception as e:
            print(f"Error processing frame for camera {self.cam_id}: {e}")
            return None

        # Add this constant at the top of your file with other configurations


    def analyze_region_detections(self, detections, region_name):
        region_data = REGIONS[f'cam{self.cam_id}'][region_name]
        vertices = region_data['vertices']
        blocks = self.region_blocks[region_name]
        filled_blocks = set()
        vehicles_in_region = 0

        try:
            for pred in detections.object_prediction_list:
                # Check if detected object is a vehicle
                if pred.category.name.lower() not in VEHICLE_CLASSES:
                    continue

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
        except Exception as e:
            print(f"Error analyzing region {region_name} for camera {self.cam_id}: {e}")
            return 0, 0, set()


    def draw_region_visualization(self, frame, region_name, vehicles, density, filled_blocks):
        region_data = REGIONS[f'cam{self.cam_id}'][region_name]
        vertices = region_data['vertices']
        color = region_data['color']
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

        # Add metrics text
        text_position = (vertices[0][0], vertices[0][1] - 10)
        cv2.putText(frame,
                  f'{region_name}: {vehicles} vehicles, {density:.1f}%',
                  text_position,
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def analyze_and_save(self, frame, result):
        frame_start_time = time.time()
        processed_frame = frame.copy()
        current_time = datetime.now()

        if result is not None:
            try:
                # Draw bounding boxes for all detected objects
                for pred in result.object_prediction_list:
                    # Only process vehicle classes
                    if pred.category.name.lower() not in VEHICLE_CLASSES:
                        continue

                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, pred.bbox.to_xyxy())

                    # Get score and category from SAHI prediction
                    score = pred.score.value
                    category = pred.category.name

                    # Draw bounding box
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                    # Draw label with score
                    label = f'{category}: {score:.2f}'
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + 10
                    cv2.putText(processed_frame, label, (x1, label_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Process R1
                r1_vehicles, r1_density, r1_blocks = self.analyze_region_detections(result, 'R1')
                self.draw_region_visualization(processed_frame, 'R1', r1_vehicles, r1_density, r1_blocks)

                # Process R2
                r2_vehicles, r2_density, r2_blocks = self.analyze_region_detections(result, 'R2')
                self.draw_region_visualization(processed_frame, 'R2', r2_vehicles, r2_density, r2_blocks)

                # Calculate weighted metrics
                weighted_vehicles = r1_vehicles + r2_vehicles
                weighted_density = (r1_density * REGIONS[f'cam{self.cam_id}']['R1']['weight'] +
                                r2_density * REGIONS[f'cam{self.cam_id}']['R2']['weight'])

                # Add total vehicle count to the frame
                cv2.putText(processed_frame,
                          f'Total Vehicles: {weighted_vehicles}',
                          (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          1,
                          (255, 255, 255),
                          2)

                # Set VDC flag based on weighted density
                vdc = weighted_density > 50

                # Save to database
                processing_time = time.time() - frame_start_time
                self.save_to_database(r1_vehicles, r1_density,
                                    r2_vehicles, r2_density,
                                    weighted_vehicles, weighted_density,
                                    vdc, processing_time)

                # Save processed frame periodically
                if time.time() - self.last_save_time >= IMAGE_SAVE_INTERVAL:
                    filename = f"{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    save_path = os.path.join(PROCESSED_FOLDER, f'cam{self.cam_id}', filename)
                    cv2.imwrite(save_path, processed_frame)
                    self.last_save_time = time.time()

                return processed_frame, weighted_vehicles, weighted_density, vdc

            except Exception as e:
                print(f"Error in analyze_and_save for camera {self.cam_id}: {e}")

        return frame, 0, 0, False

    def save_to_database(self, r1_vehicles, r1_density, r2_vehicles, r2_density,
                        weighted_vehicles, weighted_density, vdc, processing_time):
        try:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            cursor.execute(f'''INSERT INTO traffic_data_cam{self.cam_id}
                           (timestamp, r1_vehicle_count, r1_density, 
                            r2_vehicle_count, r2_density,
                            weighted_vehicle_count, weighted_density, 
                            vdc{self.cam_id}, processing_time)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (timestamp, r1_vehicles, r1_density,
                         r2_vehicles, r2_density,
                         weighted_vehicles, weighted_density,
                         vdc, processing_time))

            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"Database error for camera {self.cam_id}: {e}")

def process_camera_stream(cam_id, duration):
    processor = CameraProcessor(cam_id)
    cap = cv2.VideoCapture(RTSP_URLS[f'cam{cam_id}'])

    if not cap.isOpened():
        print(f"Failed to open camera {cam_id}")
        return

    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        result = processor.process_frame(frame)
        processed_frame, vehicles, density, vdc = processor.analyze_and_save(frame, result)

    cap.release()
    processor.thread_pool.shutdown()

def cyclic_processing():
    while True:
        for cam_id in range(1, 5):
            process_camera_stream(cam_id, PROCESS_DURATION)
            time.sleep(0.1)  # Small delay between cameras

@app.route('/')
def index():
    # Get latest processed images for all cameras
    latest_images = {}
    for cam_id in range(1, 5):
        cam_folder = os.path.join(PROCESSED_FOLDER, f'cam{cam_id}')
        if os.path.exists(cam_folder):
            files = [f for f in os.listdir(cam_folder) if f.endswith('.jpg')]
            if files:
                latest_images[f'cam{cam_id}'] = files[-1]
            else:
                latest_images[f'cam{cam_id}'] = None
        else:
            latest_images[f'cam{cam_id}'] = None

    return render_template('index.html', latest_images=latest_images)

@app.route('/analytics')
def analytics():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        data = {}
        for cam_id in range(1, 5):
            try:
                cursor.execute(f'''SELECT 
                    id, timestamp, r1_vehicle_count, r1_density,
                    r2_vehicle_count, r2_density, weighted_vehicle_count,
                    weighted_density, vdc{cam_id}, processing_time
                    FROM traffic_data_cam{cam_id} 
                    ORDER BY timestamp DESC LIMIT 100''')
                rows = cursor.fetchall()
                # If no data exists yet, provide empty list
                data[f'cam{cam_id}'] = rows if rows else []
            except sqlite3.Error as e:
                print(f"Error fetching data for camera {cam_id}: {e}")
                data[f'cam{cam_id}'] = []  # Provide empty list on error

        conn.close()

        # Calculate aggregates for each camera
        aggregates = {}
        for cam_id in range(1, 5):
            cam_data = data[f'cam{cam_id}']
            if cam_data:
                weighted_densities = [row[7] for row in cam_data]  # weighted_density is at index 7
                vehicle_counts = [row[6] for row in cam_data]      # weighted_vehicle_count is at index 6
                aggregates[f'cam{cam_id}'] = {
                    'avg_density': sum(weighted_densities) / len(weighted_densities) if weighted_densities else 0,
                    'peak_count': max(vehicle_counts) if vehicle_counts else 0
                }
            else:
                aggregates[f'cam{cam_id}'] = {
                    'avg_density': 0,
                    'peak_count': 0
                }

        return render_template('analytics.html', data=data, aggregates=aggregates)

    except Exception as e:
        print(f"Error in analytics route: {e}")
        return f"Error loading analytics: {str(e)}", 500

@app.route('/CameraStats', methods=['GET'])
def camera_stats():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        response = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cameras': {}
        }

        # Query the latest data for all cameras
        for camera_id in range(1, 5):
            try:
                cursor.execute(f'''SELECT 
                                timestamp,
                                r1_vehicle_count,
                                r1_density,
                                r2_vehicle_count,
                                r2_density,
                                weighted_vehicle_count,
                                weighted_density
                             FROM traffic_data_cam{camera_id} 
                             ORDER BY timestamp DESC LIMIT 1''')
                result = cursor.fetchone()

                if result:
                    response['cameras'][f'cam{camera_id}'] = {
                        'timestamp': result[0],
                        'r1': {
                            'vehicle_count': result[1],
                            'density': float(result[2])
                        },
                        'r2': {
                            'vehicle_count': result[3],
                            'density': float(result[4])
                        },
                        'total': {
                            'vehicle_count': result[5],
                            'density': float(result[6])
                        }
                    }
                else:
                    response['cameras'][f'cam{camera_id}'] = {
                        'status': 'No data available'
                    }

            except sqlite3.Error as e:
                print(f"Database error for camera {camera_id}: {e}")
                response['cameras'][f'cam{camera_id}'] = {
                    'error': str(e),
                    'status': 'Database error'
                }

        conn.close()
        return jsonify(response)

    except Exception as e:
        print(f"Server error in camera_stats: {e}")
        return jsonify({
            'error': str(e),
            'status': 'Server error',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@app.route('/VehicleDetect', methods=['GET'])
def vehicle_detect():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()

        response = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Query the latest VDC status for all cameras
        for camera_id in range(1, 5):
            try:
                cursor.execute(f'''SELECT vdc{camera_id}
                             FROM traffic_data_cam{camera_id} 
                             ORDER BY timestamp DESC LIMIT 1''')
                result = cursor.fetchone()

                # Convert boolean to 1 or 0
                response[f'vdc{camera_id}'] = 1 if result and result[0] else 0

            except sqlite3.Error as e:
                print(f"Database error for camera {camera_id}: {e}")
                response[f'vdc{camera_id}'] = 0

        conn.close()
        return jsonify(response)

    except Exception as e:
        print(f"Server error in vehicle_detect: {e}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'vdc1': 0,
            'vdc2': 0,
            'vdc3': 0,
            'vdc4': 0
        }), 500

if __name__ == "__main__":
    # Start cyclic processing in a separate thread
    processing_thread = Thread(target=cyclic_processing, daemon=True)
    processing_thread.start()
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
