import cv2
import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
from ultralytics import YOLO
import sqlite3
from datetime import datetime
from threading import Thread, Lock
import json

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
DATABASE = 'traffic00.db'
RTSP_URL = "http://109.236.111.203/mjpg/video.mjpg"
TRAFFIC_DATA_FILE = 'static/traffic_data.json'

# Add a database lock for thread safety
db_lock = Lock()
json_lock = Lock()

# Ensure required directories exist
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(TRAFFIC_DATA_FILE), exist_ok=True)

# Initialize the YOLOv8n model
try:
    best_model = YOLO('models/yolov8n.pt')
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    best_model = None

def init_db(database_name):
    with db_lock:
        conn = sqlite3.connect(database_name)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                vehicle_count INTEGER NOT NULL,
                density_percentage REAL NOT NULL,
                vdc1 INTEGER NOT NULL
            )
        ''')
        # Add index for better query performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON traffic_data(timestamp)
        ''')
        conn.commit()
        conn.close()

# Initialize database
init_db(DATABASE)

def save_to_json(data):
    """Thread-safe JSON file saving"""
    try:
        with json_lock:
            with open(TRAFFIC_DATA_FILE, "w") as f:
                json.dump(data, f)
    except Exception as e:
        print(f"Error saving JSON file: {e}")

def save_to_db(timestamp, vehicles_in_region, density_percentage, vdc1):
    """Thread-safe database saving"""
    try:
        with db_lock:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO traffic_data 
                (timestamp, vehicle_count, density_percentage, vdc1) 
                VALUES (?, ?, ?, ?)
            ''', (timestamp, vehicles_in_region, density_percentage, vdc1))
            conn.commit()
            conn.close()
    except sqlite3.Error as e:
        print(f"Error inserting into database: {e}")

def divide_region_into_blocks(vertices, num_blocks):
    """Divide the defined region into horizontal blocks"""
    min_y = min(vertices[:, 1])
    max_y = max(vertices[:, 1])
    block_height = (max_y - min_y) / num_blocks
    blocks = [(min_y + i * block_height, min_y + (i + 1) * block_height) 
             for i in range(num_blocks)]
    return blocks

def get_filled_blocks_and_count(detection_frame, boxes, blocks, vertices):
    """Determine which blocks contain vehicles and count vehicles inside the polygon"""
    filled_blocks = set()
    vehicles_in_region = 0
    bounding_boxes = []

    for box in boxes:
        x1, y1, x2, y2 = box
        x_center = int((x1 + x2) / 2.0)
        y_center = int((y1 + y2) / 2.0)

        if is_point_in_polygon(x_center, y_center, vertices):
            vehicles_in_region += 1
            for idx, (y_min, y_max) in enumerate(blocks):
                if y_min <= y_center <= y_max:
                    filled_blocks.add(idx)
                    bounding_boxes.append((x1, y1, x2, y2))
                    break

    return len(filled_blocks), vehicles_in_region, bounding_boxes

def is_point_in_polygon(x, y, vertices):
    """Check if a point is inside the polygon defined by vertices"""
    return cv2.pointPolygonTest(vertices, (x, y), False) >= 0

def process_video():
    if best_model is None:
        print("YOLO model is not loaded. Exiting video processing.")
        return

    cap = cv2.VideoCapture(RTSP_URL)
    frame_count = 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the region quadrilateral
    #vertices1 = np.array([(219,499), (287,191), (501,195), (639,483)], dtype=np.int32)
    vertices1 = np.array([(115,403), (561,353), (213,171), (94,218)], dtype=np.int32)

    # Number of horizontal blocks
    NUM_BLOCKS = 5
    blocks = divide_region_into_blocks(vertices1, NUM_BLOCKS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detection_frame = frame.copy()
        try:
            results = best_model.predict(detection_frame, imgsz=640, conf=0.4, 
                                       classes=[2, 3, 5, 7])
        except Exception as e:
            print(f"Error during model prediction: {e}")
            break

        # Get the bounding boxes of detected vehicles
        boxes = results[0].boxes.xyxy.cpu().numpy()

        # Get the number of filled blocks, vehicles in region, and bounding boxes
        num_filled_blocks, vehicles_in_region, bounding_boxes = get_filled_blocks_and_count(
            detection_frame, boxes, blocks, vertices1)

        # Calculate traffic density based on filled blocks (percentage)
        density_percentage = (num_filled_blocks / NUM_BLOCKS) * 100

        # Determine VDC1 value based on density
        vdc1 = 1 if density_percentage > 30 else 0

        # Draw the region and blocks
        cv2.polylines(detection_frame, [vertices1], True, (0, 255, 0), 2)
        for y_min, y_max in blocks:
            cv2.line(detection_frame, (0, int(y_min)), 
                    (frame_width, int(y_min)), (255, 0, 0), 1)

        # Draw the bounding boxes
        for box in bounding_boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Save data for every frame
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        
        # Save to database in a separate thread
        Thread(target=save_to_db, 
               args=(timestamp, vehicles_in_region, density_percentage, vdc1)).start()

        # Save to JSON in a separate thread
        data = {
            "weighted_density": density_percentage,
            "vdc1": vdc1,
            "timestamp": timestamp,
            "vehicle_count": vehicles_in_region
        }
        Thread(target=save_to_json, args=(data,)).start()

        ret, buffer = cv2.imencode('.jpg', detection_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analytics')
def analytics():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = 100
        offset = (page - 1) * per_page
        
        with db_lock:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            
            # Get total count
            cursor.execute('SELECT COUNT(*) FROM traffic_data')
            total_records = cursor.fetchone()[0]
            
            # Get paginated data
            cursor.execute('''
                SELECT timestamp, vehicle_count, density_percentage, vdc1 
                FROM traffic_data 
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            ''', (per_page, offset))
            data = cursor.fetchall()
            conn.close()

        total_pages = (total_records + per_page - 1) // per_page
        return render_template('analytics.html', 
                             data=data, 
                             page=page, 
                             total_pages=total_pages)
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return f"Database error: {e}", 500

@app.route('/VehicleDetect/<string:device_id>', methods=['GET'])
def vehicle_detect(device_id):
    try:
        with json_lock:
            with open(TRAFFIC_DATA_FILE, 'r') as f:
                data = json.load(f)
        response = {'DeviceID': device_id, 'VDC1': data['vdc1']}
        return jsonify(response)
    except FileNotFoundError:
        return jsonify({'error': 'JSON file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    return Response(process_video(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
