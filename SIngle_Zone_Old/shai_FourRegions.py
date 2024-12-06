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
DATABASE = 'traffic_data_1.db'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define regions as quadrilateral boundaries for R1, R2, R3, and R4
regions = {
    "R1": np.array([(387, 777), (977, 420), (1186, 435), (863, 814)], dtype=np.int32),
    "R2": np.array([(1074, 778), (1704, 784), (1558, 2130), (1062, 2136)], dtype=np.int32),
    "R3": np.array([(1716, 790), (2384, 798), (2042, 2116), (1572, 2130)], dtype=np.int32),
    "R4": np.array([(2398, 804), (3056, 810), (2562, 2104), (2058, 2116)], dtype=np.int32)
}

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
            total_vehicle_count INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db(DATABASE)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_point_in_polygon(x, y, vertices):
    return cv2.pointPolygonTest(vertices, (x, y), False) >= 0

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

        boxes, confidences, class_ids = [], [], []

        for pred in result.object_prediction_list:
            if pred.category.id in [2, 3, 5, 7]:
                bbox = pred.bbox.to_xyxy()
                boxes.append(bbox)
                confidences.append(pred.score.value)
                class_ids.append(int(pred.category.id))

        if boxes:
            return sv.Detections(
                xyxy=np.array(boxes, dtype=np.float32),
                confidence=np.array(confidences, dtype=np.float32),
                class_id=np.array(class_ids, dtype=int)
            )
        return None

    except Exception as e:
        print(f"Error in SAHI processing: {e}")
        try:
            results = yolo_model.predict(
                frame,
                conf=0.4,
                classes=[2, 3, 5, 7],
                verbose=False
            )

            if results and results[0].boxes:
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

def process_video(video_path):
    if detection_model is None and yolo_model is None:
        print("No detection models loaded. Exiting video processing.")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = process_frame_with_sahi(frame)
        processed_frame = frame.copy()

        if detections:
            region_counts = {region: 0 for region in regions.keys()}

            for box in detections.xyxy:
                x_center, y_center = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                # Draw bounding box
                cv2.rectangle(processed_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

                for region_name, vertices in regions.items():
                    if is_point_in_polygon(x_center, y_center, vertices):
                        region_counts[region_name] += 1

            total_count = sum(region_counts.values())

            for region_name, vertices in regions.items():
                cv2.polylines(processed_frame, [vertices], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.putText(processed_frame, f'{region_name} Count: {region_counts[region_name]}',
                            (vertices[0][0], vertices[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            cv2.putText(processed_frame, f'Total Vehicles: {total_count}',
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % int(fps) == 0:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                try:
                    conn = sqlite3.connect(DATABASE)
                    cursor = conn.cursor()
                    cursor.execute('INSERT INTO traffic_data (timestamp, total_vehicle_count) VALUES (?, ?)',
                                   (timestamp, total_count))
                    conn.commit()
                    conn.close()
                except sqlite3.Error as e:
                    print(f"Database error: {e}")

        # Encode frame to JPEG for live video feed
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.route('/')
def index():
    filename = request.args.get('filename')
    return render_template('index.html', filename=filename)

@app.route('/analytics')
def analytics():
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('SELECT timestamp, total_vehicle_count FROM traffic_data ORDER BY id DESC LIMIT 100')
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
        return redirect(url_for('video_feed', filename=filename))

@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(process_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
