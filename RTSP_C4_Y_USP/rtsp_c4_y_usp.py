import cv2
import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, Response, jsonify, send_from_directory
from ultralytics import YOLO
import sqlite3
from datetime import datetime
from threading import Thread, Lock
import json
import time
from collections import defaultdict

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
DATABASE = 'traffic_multi.db'
TRAFFIC_DATA_FILE = 'static/traffic_data.json'
FRAME_SKIP = 5
PROCESS_DURATION = 2  # Process each stream for 2 seconds
SAVE_INTERVAL = 2  # Save images every 10 seconds

# RTSP URLs for each camera
RTSP_URLS = {
    'CAM1': "http://109.236.111.203/mjpg/video.mjpg",
    'CAM2': "http://109.236.111.203/mjpg/video.mjpg",
    'CAM3': "http://109.236.111.203/mjpg/video.mjpg",
    'CAM4': "http://camera.buffalotrace.com/mjpg/video.mjpg"
}

CAMERA_FOLDERS = {
    'CAM1': os.path.join(PROCESSED_FOLDER, 'camera1'),
    'CAM2': os.path.join(PROCESSED_FOLDER, 'camera2'),
    'CAM3': os.path.join(PROCESSED_FOLDER, 'camera3'),
    'CAM4': os.path.join(PROCESSED_FOLDER, 'camera4')
}
# Define regions for each camera
REGIONS = {
    'CAM1': {
        'R1': {
            'vertices': np.array([(86,431), (94,187), (315,191), (323,430)], dtype=np.int32),
            'weight': 0.5,
            'color': (0, 255, 0)
        },
        'R2': {
            'vertices': np.array([(339,422), (324,170), (553,173), (586,434)], dtype=np.int32),
            'weight': 0.5,
            'color': (255, 0, 0)
        }
    },
    'CAM2': {
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
    },
    'CAM3': {
        'R1': {
            'vertices': np.array([(86,431), (94,187), (315,191), (323,430)], dtype=np.int32),
            'weight': 0.5,
            'color': (0, 255, 0)
        },
        'R2': {
            'vertices': np.array([(339,422), (324,170), (553,173), (586,434)], dtype=np.int32),
            'weight': 0.5,
            'color': (255, 0, 0)
        }
    },
    'CAM4': {
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


# Locks
db_lock = Lock()
json_lock = Lock()
processed_images_lock = Lock()

# Ensure directories exist
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(TRAFFIC_DATA_FILE), exist_ok=True)

# Initialize YOLO model
try:
    model = YOLO('models/yolov8n.pt')
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

def init_db():
    """Initialize database tables for each camera"""
    with db_lock:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        
        for cam_id in RTSP_URLS.keys():
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {cam_id.lower()}_traffic_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    r1_vehicle_count INTEGER NOT NULL,
                    r1_density REAL NOT NULL,
                    r2_vehicle_count INTEGER NOT NULL,
                    r2_density REAL NOT NULL,
                    weighted_vehicle_count REAL NOT NULL,
                    weighted_density REAL NOT NULL,
                    vdc INTEGER NOT NULL,
                    processing_time REAL NOT NULL
                )
            ''')
            cursor.execute(f'''
                CREATE INDEX IF NOT EXISTS idx_{cam_id.lower()}_timestamp 
                ON {cam_id.lower()}_traffic_data(timestamp)
            ''')
        conn.commit()
        conn.close()

# Initialize database
init_db()

def save_to_db(cam_id, data):
    """Save traffic data to the database"""
    try:
        with db_lock:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            cursor.execute(f'''
                INSERT INTO {cam_id.lower()}_traffic_data 
                (timestamp, r1_vehicle_count, r1_density, r2_vehicle_count, 
                r2_density, weighted_vehicle_count, weighted_density, vdc, 
                processing_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['timestamp'],
                data['r1_vehicle_count'],
                data['r1_density'],
                data['r2_vehicle_count'],
                data['r2_density'],
                data['weighted_vehicle_count'],
                data['weighted_density'],
                data['vdc'],
                data['processing_time']
            ))
            conn.commit()
            conn.close()
    except sqlite3.Error as e:
        print(f"Database error for {cam_id}: {e}")

def save_processed_image(cam_id, frame, timestamp):
    """Save processed frame as image in camera-specific folder"""
    try:
        # Get camera folder path
        folder_path = CAMERA_FOLDERS[cam_id]
        if not folder_path:
            raise Exception(f"Invalid camera folder path for {cam_id}")

        # Create filename with timestamp
        current_time = timestamp.strftime('%Y%m%d_%H%M%S')
        filename = f"{current_time}.jpg"
        filepath = os.path.join(folder_path, filename)

        print(f"Saving image for {cam_id} to {filepath}")  # Debug log

        # Save image
        with processed_images_lock:
            success = cv2.imwrite(filepath, frame)
            if not success:
                raise Exception(f"Failed to write image to {filepath}")

            print(f"Successfully saved image for {cam_id}")  # Debug log

            # Clean up old files
            try:
                image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
                image_files.sort(key=lambda x: os.path.getctime(os.path.join(folder_path, x)))

                # Keep only the 10 most recent images
                if len(image_files) > 10:
                    for old_file in image_files[:-10]:
                        old_path = os.path.join(folder_path, old_file)
                        try:
                            os.remove(old_path)
                            print(f"Removed old file: {old_file}")  # Debug log
                        except Exception as e:
                            print(f"Failed to remove old file {old_file}: {e}")

            except Exception as e:
                print(f"Error during cleanup for {cam_id}: {e}")

        # Return relative path for URL (e.g., "1/filename.jpg" for CAM1)
        cam_num = cam_id.replace('CAM', '')
        return f"{cam_num}/{filename}"

    except Exception as e:
        print(f"Error saving image for {cam_id}: {e}")
        return None

def process_frame(frame, cam_id, regions):
    """Process a single frame for vehicle detection"""
    if frame is None:
        print(f"Received empty frame for {cam_id}")
        return None, None
    if model is None:
        return frame, None
    
    detection_frame = frame.copy()
    results = model.predict(detection_frame, imgsz=640, conf=0.4, classes=[2, 3, 5, 7])
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    region_data = defaultdict(dict)
    
    # Process each region
    for region_name, region_info in regions.items():
        vehicles = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            if cv2.pointPolygonTest(region_info['vertices'], (center_x, center_y), False) >= 0:
                vehicles.append((x1, y1, x2, y2))
                # Draw bounding box
                cv2.rectangle(detection_frame, (x1, y1), (x2, y2), region_info['color'], 2)
        
        # Draw region polygon
        cv2.polylines(detection_frame, [region_info['vertices']], True, region_info['color'], 2)
        
        # Calculate metrics
        region_data[region_name] = {
            'vehicle_count': len(vehicles),
            'density': (len(vehicles) / 10) * 100  # Simplified density calculation
        }
    
    return detection_frame, region_data

def camera_processor(cam_id, url, duration=2):
    """Process individual camera stream for specified duration"""
    print(f"Starting processing of {cam_id}")

    try:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            print(f"Failed to open camera {cam_id}")
            return

        process_start_time = time.time()
        last_save_time = time.time()
        frame_count = 0

        while time.time() - process_start_time < duration:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame from {cam_id}")
                break

            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue

            # Process frame
            processed_frame, region_data = process_frame(frame, cam_id, REGIONS[cam_id])

            if region_data:
                # Calculate weighted metrics
                weighted_count = sum(data['vehicle_count'] for data in region_data.values())
                weighted_density = sum(
                    data['density'] * REGIONS[cam_id][region]['weight']
                    for region, data in region_data.items()
                )

                # Determine VDC value based on weighted density
                vdc = 1 if weighted_density > 30 else 0

                # Save data
                timestamp = datetime.now()
                processing_time = time.time() - process_start_time

                data = {
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'r1_vehicle_count': region_data['R1']['vehicle_count'],
                    'r1_density': region_data['R1']['density'],
                    'r2_vehicle_count': region_data['R2']['vehicle_count'],
                    'r2_density': region_data['R2']['density'],
                    'weighted_vehicle_count': weighted_count,
                    'weighted_density': weighted_density,
                    'vdc': vdc,
                    'processing_time': processing_time
                }

                # Save to database
                save_to_db(cam_id, data)

                # Save processed image every SAVE_INTERVAL seconds
                current_time = time.time()
                if current_time - last_save_time >= SAVE_INTERVAL:
                    save_processed_image(cam_id, processed_frame, timestamp)
                    last_save_time = current_time

    except Exception as e:
        print(f"Error processing {cam_id}: {e}")
    finally:
        if 'cap' in locals():
            cap.release()
        print(f"Finished processing {cam_id}")

def cyclic_processing():
    """Cycle through cameras, processing each for 2 seconds"""
    camera_order = list(RTSP_URLS.keys())  # ['CAM1', 'CAM2', 'CAM3', 'CAM4']
    current_index = 0

    while True:
        try:
            # Get current camera
            cam_id = camera_order[current_index]
            url = RTSP_URLS[cam_id]

            print(f"Processing {cam_id}")

            # Process current camera
            camera_processor(cam_id, url, duration=PROCESS_DURATION)

            # Move to next camera
            current_index = (current_index + 1) % len(camera_order)

            # Small delay between cameras
            time.sleep(0.1)

        except Exception as e:
            print(f"Error in cyclic processing: {e}")
            # Move to next camera even if current one fails
            current_index = (current_index + 1) % len(camera_order)
            time.sleep(1)  # Add delay before retrying

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analytics')
def analytics():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = 50
        offset = (page - 1) * per_page
        
        data = {}
        total_records = {}
        
        with db_lock:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            
            for cam_id in RTSP_URLS.keys():
                # Get total count
                cursor.execute(f'SELECT COUNT(*) FROM {cam_id.lower()}_traffic_data')
                total_records[cam_id] = cursor.fetchone()[0]
                
                # Get paginated data
                cursor.execute(f'''
                    SELECT * FROM {cam_id.lower()}_traffic_data 
                    ORDER BY timestamp DESC 
                    LIMIT ? OFFSET ?
                ''', (per_page, offset))
                data[cam_id] = cursor.fetchall()
            
            conn.close()
        
        total_pages = max((max(total_records.values()) + per_page - 1) // per_page, 1)
        return render_template('analytics.html',
                             data=data,
                             page=page,
                             total_pages=total_pages,
                             cameras=RTSP_URLS.keys())
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return f"Database error: {e}", 500

@app.route('/processed_images/<string:camera>/<path:filename>')
def processed_images(camera, filename):
    """Serve images from camera-specific folders"""
    try:
        # Sanitize inputs
        camera = str(camera).strip()
        filename = str(filename).strip()
        
        if not camera.isdigit() or not filename.endswith('.jpg'):
            return "Invalid request", 400
            
        folder_path = os.path.join(PROCESSED_FOLDER, f'camera{camera}')
        
        if not os.path.exists(folder_path):
            return f"Camera folder not found: camera{camera}", 404
            
        return send_from_directory(folder_path, filename)
    except Exception as e:
        print(f"Error serving image {filename} from camera {camera}: {e}")
        return "Error serving image", 500

@app.route('/get_latest_data')
def get_latest_data():
    """Get the latest data and images for all cameras"""
    try:
        data = {}
        for cam_id in RTSP_URLS.keys():
            try:
                # Get camera folder path
                folder_path = CAMERA_FOLDERS[cam_id]

                # Get latest image
                files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
                latest_image = None
                if files:
                    latest_image = max(files, key=lambda x: os.path.getctime(
                        os.path.join(folder_path, x)))

                # Get latest metrics from database
                with db_lock:
                    conn = sqlite3.connect(DATABASE)
                    cursor = conn.cursor()
                    cursor.execute(f'''
                        SELECT * FROM {cam_id.lower()}_traffic_data 
                        ORDER BY timestamp DESC LIMIT 1
                    ''')
                    row = cursor.fetchone()
                    conn.close()

                if row:
                    data[cam_id] = {
                        'image': latest_image,
                        'timestamp': row[1],
                        'r1_vehicle_count': row[2],
                        'r1_density': row[3],
                        'r2_vehicle_count': row[4],
                        'r2_density': row[5],
                        'weighted_vehicle_count': row[6],
                        'weighted_density': row[7],
                        'vdc': row[8],
                        'processing_time': row[9]
                    }

            except Exception as e:
                print(f"Error getting data for {cam_id}: {e}")
                continue

        return jsonify(data)

    except Exception as e:
        print(f"Error in get_latest_data: {e}")
        return jsonify({'error': str(e)}), 500

def ensure_camera_folders():
    """Ensure all camera folders exist"""
    for folder in CAMERA_FOLDERS.values():
        os.makedirs(folder, exist_ok=True)

# Call this when starting the app
ensure_camera_folders()
@app.route('/VehicleDetect/<int:camera_id>', methods=['GET'])
def vehicle_detect(camera_id):
    """
    Get the latest VDC (Vehicle Density Classification) status for a specific camera
    camera_id should be 1-4 corresponding to CAM1-CAM4
    """
    try:
        # Map camera_id to camera name
        cam_name = f'CAM{camera_id}'

        # Verify valid camera_id
        if camera_id < 1 or camera_id > 4:
            return jsonify({
                'error': 'Invalid camera ID. Must be between 1 and 4'
            }), 400

        # Connect to database with lock to prevent concurrent access
        with db_lock:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()

            # Query the latest VDC value for the specified camera
            cursor.execute(f'''
                SELECT vdc 
                FROM {cam_name.lower()}_traffic_data 
                ORDER BY timestamp DESC 
                LIMIT 1
            ''')
            result = cursor.fetchone()
            conn.close()

        if result is not None:
            return jsonify({
                'CameraID': camera_id,
                f'VDC{camera_id}': int(result[0])
            })

        return jsonify({
            'error': f'No data available for camera {camera_id}'
        }), 404

    except Exception as e:
        print(f"Error in vehicle_detect for camera {camera_id}: {e}")
        return jsonify({
            'error': f'Error retrieving data: {str(e)}'
        }), 500
if __name__ == "__main__":
    try:
        # Initialize model globally
        if model is None:
            print("Warning: YOLO model failed to load")

        # Ensure camera folders exist
        ensure_camera_folders()

        # Start cyclic processing in a separate thread
        processing_thread = Thread(target=cyclic_processing, daemon=True)
        processing_thread.start()

        # Start Flask app (only once, with threading enabled)
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error in main: {e}")
        cv2.destroyAllWindows()
