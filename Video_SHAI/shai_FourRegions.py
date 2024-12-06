import cv2
import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, Response
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import sqlite3
from datetime import datetime
from threading import Thread
import supervision as sv
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
DATABASE = 'traffic.db'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Ensure upload and processed folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define three regions with their vertices and weightages
REGIONS = {
    'R1': {
        'vertices': np.array([(301,751), (883,792), (1218,312), (967,306)], dtype=np.int32),
        'weight': 0.25,
        'color': (0, 255, 0)  # Green
    },
    'R2': {
        'vertices': np.array([(910,780), (1438,793), (1420,320), (1230,312)], dtype=np.int32),
        'weight': 0.25,
        'color': (255, 0, 0)  # Blue
    },
    'R3': {
        'vertices': np.array([(998,296), (1411,306), (1425,170), (1354,167)], dtype=np.int32),
        'weight': 0.50,
        'color': (0, 0, 255)  # Red
    }
}

NUM_BLOCKS = 5

# Initialize YOLO and SAHI models
try:
    local_model_path = 'models/yolov8n.pt'
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=local_model_path,
        confidence_threshold=0.3,
        device='cuda'
    )
    yolo_model = YOLO(local_model_path)
except Exception as e:
    print(f"Error loading models: {e}")
    detection_model = None
    yolo_model = None

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
            weighted_density REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db(DATABASE)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_point_in_polygon(x, y, vertices):
    return cv2.pointPolygonTest(vertices, (x, y), False) >= 0

def divide_region_into_blocks(vertices, num_blocks):
    min_y = min(vertices[:, 1])
    max_y = max(vertices[:, 1])
    block_height = (max_y - min_y) / num_blocks
    blocks = [(min_y + i * block_height, min_y + (i + 1) * block_height) for i in range(num_blocks)]
    return blocks

def analyze_region(boxes, blocks, vertices):
    filled_blocks = set()
    vehicles_in_region = 0

    for box in boxes:
        x1, y1, x2, y2 = box
        x_center = int((x1 + x2) / 2.0)
        y_center = int((y1 + y2) / 2.0)

        if is_point_in_polygon(x_center, y_center, vertices):
            vehicles_in_region += 1
            for idx, (y_min, y_max) in enumerate(blocks):
                if y_min <= y_center <= y_max:
                    filled_blocks.add(idx)
                    break

    density = (len(filled_blocks) / NUM_BLOCKS) * 100
    return vehicles_in_region, density

def process_frame_with_sahi(frame):
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

        boxes = []
        confidences = []
        class_ids = []

        # Filter for vehicle classes (2: car, 3: motorcycle, 5: bus, 7: truck)
        vehicle_classes = [2, 3, 5, 7]

        for pred in result.object_prediction_list:
            if pred.category.id in vehicle_classes:
                bbox = pred.bbox.to_xyxy()
                boxes.append(bbox)
                confidences.append(pred.score.value)
                class_ids.append(int(pred.category.id))

        if boxes:
            boxes = np.array(boxes, dtype=np.float32)
            confidences = np.array(confidences, dtype=np.float32)
            class_ids = np.array(class_ids, dtype=int)

            # Draw bounding boxes on the frame
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 100), 2)  # Yellow box
                label = f'Vehicle {conf:.2f}'
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255, 255), 2)

            return sv.Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=class_ids
            )
        return None

    except Exception as e:
        print(f"Error in SAHI processing: {e}")
        return None

def process_video(video_path, processed_path):
    if detection_model is None and yolo_model is None:
        print("No detection models loaded. Exiting video processing.")
        return

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_path, fourcc, fps, (frame_width, frame_height))

    region_blocks = {
        region: divide_region_into_blocks(data['vertices'], NUM_BLOCKS)
        for region, data in REGIONS.items()
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = process_frame_with_sahi(frame)
        processed_frame = frame.copy()

        if detections is not None:
            try:
                boxes = detections.xyxy
                region_metrics = {}
                total_vehicles = 0  # Total count across all regions

                # Process each region
                for region_name, region_data in REGIONS.items():
                    vertices = region_data['vertices']
                    weight = region_data['weight']
                    color = region_data['color']
                    blocks = region_blocks[region_name]

                    # Analyze region
                    vehicles, density = analyze_region(boxes, blocks, vertices)
                    region_metrics[region_name] = {
                        'vehicles': vehicles,
                        'density': density
                    }
                    total_vehicles += vehicles

                    # Draw region and its metrics
                    cv2.polylines(processed_frame, [vertices], isClosed=True, color=color, thickness=2)

                    # Draw block boundaries
                    for y_min, y_max in blocks:
                        cv2.line(processed_frame,
                                (vertices[0][0], int(y_min)),
                                (vertices[-1][0], int(y_min)),
                                color, 1)

                    # Add region metrics to frame
                    text_position = vertices[0]
                    cv2.putText(processed_frame,
                              f'{region_name}: {vehicles} vehicles, {density:.1f}%',
                              (text_position[0], text_position[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Calculate weighted density
                weighted_density = sum(
                    region_metrics[region]['density'] * REGIONS[region]['weight']
                    for region in REGIONS.keys()
                )

                # Add overall metrics to frame
                cv2.putText(processed_frame,
                          f'Total Vehicles: {total_vehicles}, Weighted Density: {weighted_density:.1f}%',
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Store data in database once per second
                if frame_count % int(fps) == 0:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    try:
                        conn = sqlite3.connect(DATABASE)
                        cursor = conn.cursor()
                        cursor.execute('''INSERT INTO traffic_data 
                                       (timestamp, r1_vehicle_count, r1_density,
                                        r2_vehicle_count, r2_density,
                                        r3_vehicle_count, r3_density,
                                        weighted_vehicle_count, weighted_density)
                                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                    (timestamp,
                                     region_metrics['R1']['vehicles'],
                                     region_metrics['R1']['density'],
                                     region_metrics['R2']['vehicles'],
                                     region_metrics['R2']['density'],
                                     region_metrics['R3']['vehicles'],
                                     region_metrics['R3']['density'],
                                     total_vehicles,
                                     weighted_density))
                        conn.commit()
                        conn.close()
                    except sqlite3.Error as e:
                        print(f"Database error: {e}")

            except Exception as e:
                print(f"Error in frame processing: {e}")
                processed_frame = frame.copy()

        out.write(processed_frame)
        frame_count += 1

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    out.release()

# Flask routes
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
