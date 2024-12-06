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

# Define the region quadrilateral
#vertices1 = np.array([(2027, 2149), (1969, 946), (2461, 949), (3786, 2129)], dtype=np.int32) for Test1
vertices1 = np.array([(276,783), (1322,133), (1481,145), (1430,814)], dtype=np.int32)
NUM_BLOCKS = 5

# Initialize YOLO and SAHI models
try:
    # Path to your local YOLOv8x model
    local_model_path = 'models/yolov8n.pt'
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=local_model_path,
        confidence_threshold=0.3,
        device='cuda'  # Use 'cpu' if no GPU available
    )
    # Also initialize regular YOLO model for backup/comparison if needed
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
            vehicle_count INTEGER NOT NULL,
            density_status REAL NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database
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

def get_filled_blocks_and_count(detection_frame, boxes, blocks, vertices):
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

    return len(filled_blocks), vehicles_in_region

def process_frame_with_sahi(frame):
    """
    Process a single frame using SAHI technique with corrected parameters
    """
    try:
        # SAHI prediction with slicing - removed class_agnostic parameter
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

        # Convert SAHI results to supervision Detections format
        boxes = []
        confidences = []
        class_ids = []

        for pred in result.object_prediction_list:
            # Filter for vehicle classes (car=2, motorcycle=3, bus=5, truck=7)
            if pred.category.id in [2, 3, 5, 7]:
                bbox = pred.bbox.to_xyxy()
                boxes.append(bbox)
                confidences.append(pred.score.value)
                class_ids.append(int(pred.category.id))

        if boxes:
            boxes = np.array(boxes, dtype=np.float32)
            confidences = np.array(confidences, dtype=np.float32)
            class_ids = np.array(class_ids, dtype=int)

            return sv.Detections(
                xyxy=boxes,
                confidence=confidences,
                class_id=class_ids
            )
        return None

    except Exception as e:
        print(f"Error in SAHI processing: {e}")
        # Fallback to regular YOLO detection if SAHI fails
        try:
            results = yolo_model.predict(
                frame,
                conf=0.4,
                classes=[2, 3, 5, 7],  # Only detect vehicles
                verbose=False  # Reduce console output
            )

            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()

                return sv.Detections(
                    xyxy=boxes.astype(np.float32),
                    confidence=confidences.astype(np.float32),
                    class_id=class_ids.astype(int)
                )
        except Exception as e:
            print(f"Error in fallback YOLO detection: {e}")
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

    blocks = divide_region_into_blocks(vertices1, NUM_BLOCKS)

    # Initialize annotators with explicit colors
    colors = sv.ColorPalette.from_hex(['#FF0000', '#00FF00', '#0000FF', '#FFFF00'])
    box_annotator = sv.BoxAnnotator(
        color=colors,
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=colors,
        text_thickness=2,
        text_scale=0.5
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame with SAHI
        detections = process_frame_with_sahi(frame)
        processed_frame = frame.copy()

        if detections is not None:
            try:
                # Draw detections with error handling
                processed_frame = box_annotator.annotate(
                    scene=processed_frame.copy(),
                    detections=detections
                )

                # Create labels for detected objects
                labels = [f"Vehicle {i}" for i in range(len(detections))]
                processed_frame = label_annotator.annotate(
                    scene=processed_frame,
                    detections=detections,
                    labels=labels
                )

                # Get boxes for traffic analysis
                boxes = detections.xyxy

                # Get the number of filled blocks and vehicles in region
                num_filled_blocks, vehicles_in_region = get_filled_blocks_and_count(frame, boxes, blocks, vertices1)

                # Calculate traffic density
                density_percentage = (num_filled_blocks / NUM_BLOCKS) * 100
                density_status = f"Density: {density_percentage:.2f}%"

                # Draw region of interest and annotations
                cv2.polylines(processed_frame, [vertices1], isClosed=True, color=(0, 255, 0), thickness=2)

                # Draw block boundaries
                for idx, (y_min, y_max) in enumerate(blocks):
                    cv2.line(processed_frame, (0, int(y_min)), (frame_width, int(y_min)), (255, 0, 0), 2)
                    cv2.putText(processed_frame, f'Block {idx + 1}', (10, int(y_min) + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

                # Add metrics with improved visibility
                metrics_color = (0, 0, 255)  # Red color for metrics
                cv2.putText(processed_frame, f'Vehicles in Region: {vehicles_in_region}',
                           (frame_width - 400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, metrics_color, 2, cv2.LINE_AA)
                cv2.putText(processed_frame, f'Blocks Filled: {num_filled_blocks}',
                           (frame_width - 400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, metrics_color, 2, cv2.LINE_AA)
                cv2.putText(processed_frame, density_status,
                           (frame_width - 400, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, metrics_color, 2, cv2.LINE_AA)

                # Store data in database
                if frame_count % int(fps) == 0:  # Store data once per second
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    try:
                        conn = sqlite3.connect(DATABASE)
                        cursor = conn.cursor()
                        cursor.execute('''INSERT INTO traffic_data 
                                       (timestamp, vehicle_count, density_status) 
                                       VALUES (?, ?, ?)''',
                                    (timestamp, vehicles_in_region, density_percentage))
                        conn.commit()
                        conn.close()
                    except sqlite3.Error as e:
                        print(f"Database error: {e}")

            except Exception as e:
                print(f"Error in frame processing: {e}")
                processed_frame = frame.copy()  # Use original frame if processing fails

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
        cursor.execute('SELECT timestamp, vehicle_count, density_status FROM traffic_data ORDER BY id DESC LIMIT 100')
        data = cursor.fetchall()
        conn.close()
        data = data[::-1]  # Reverse to have chronological order
        return render_template('analytics.html', data=data)
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
