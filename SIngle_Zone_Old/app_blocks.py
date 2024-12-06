import cv2
import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, Response
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import sqlite3
from datetime import datetime
from threading import Thread

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
DATABASE = 'traffic.db'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Ensure upload and processed folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Set the upload folder in the Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the single region quadrilateral (vertices1)

vertices1 = np.array([(2027,2149), (1969,946), (2461,949), (3786,2129)], dtype=np.int32)

# Number of horizontal blocks (10 blocks)
NUM_BLOCKS = 5

# Initialize the YOLOv8 model
try:
    best_model = YOLO('models/best.pt')  # Ensure this path is correct
except AttributeError as e:
    print(f"Error loading YOLO model: {e}")
    best_model = None  # Prevent processing if the model fails to load

# Import the database initialization function
from init_db import init_db
init_db(DATABASE)

# Check for valid file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def divide_region_into_blocks(vertices, num_blocks):
    """
    Divide the defined region into horizontal blocks.
    """
    min_y = min(vertices[:, 1])
    max_y = max(vertices[:, 1])
    block_height = (max_y - min_y) / num_blocks
    blocks = [(min_y + i * block_height, min_y + (i + 1) * block_height) for i in range(num_blocks)]
    return blocks

def get_filled_blocks(detection_frame, boxes, blocks, vertices):
    """
    Determine which blocks contain vehicles based on bounding boxes.
    """
    filled_blocks = set()
    for box in boxes:
        x1, y1, x2, y2 = box
        x_center = int((x1 + x2) / 2.0)  # Convert to int for cv2 function
        y_center = int((y1 + y2) / 2.0)  # Convert to int for cv2 function

        if cv2.pointPolygonTest(vertices, (x_center, y_center), False) >= 0:
            for idx, (y_min, y_max) in enumerate(blocks):
                if y_min <= y_center <= y_max:
                    filled_blocks.add(idx)
                    break
    return len(filled_blocks)

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
    except sqlite3.OperationalError as e:
        print(f"Database error: {e}")
        return f"Database error: {e}", 500

def process_video(video_path, processed_path):
    if best_model is None:
        print("YOLO model is not loaded. Exiting video processing.")
        return

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default to 30 if FPS cannot be retrieved

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print("Error: Could not open VideoWriter.")
        cap.release()
        return

    blocks = divide_region_into_blocks(vertices1, NUM_BLOCKS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detection_frame = frame.copy()
        try:
            results = best_model.predict(detection_frame, imgsz=640, conf=0.4)
        except AttributeError as e:
            print(f"Error during model prediction: {e}")
            break
        except Exception as e:
            print(f"Unexpected error during model prediction: {e}")
            break

        processed_frame = results[0].plot(line_width=1)
        cv2.polylines(processed_frame, [vertices1], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw block boundaries and labels
        for idx, (y_min, y_max) in enumerate(blocks):
            cv2.line(processed_frame, (0, int(y_min)), (frame_width, int(y_min)), (255, 0, 0), 2)
            cv2.putText(processed_frame, f'Block {idx + 1}', (10, int(y_min) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Get the bounding boxes of detected vehicles
        boxes = results[0].boxes.xyxy

        # Count the number of detected vehicles
        num_vehicles = len(boxes)

        # Get the number of filled blocks
        num_filled_blocks = get_filled_blocks(detection_frame, boxes, blocks, vertices1)

        # Calculate traffic density based on filled blocks (percentage)
        density_percentage = (num_filled_blocks / NUM_BLOCKS) * 100
        density_status = f"Density: {density_percentage:.2f}%"

        # Add text annotations for number of vehicles, blocks filled, and density
        cv2.putText(processed_frame, f'Vehicles: {num_vehicles}', (frame_width - 300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(processed_frame, f'Blocks Filled: {num_filled_blocks}', (frame_width - 300, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(processed_frame, density_status, (frame_width - 300, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        frame_count += 1
        if frame_count % int(fps) == 0:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            try:
                conn = sqlite3.connect(DATABASE)
                cursor = conn.cursor()
                cursor.execute('INSERT INTO traffic_data (timestamp, vehicle_count, density_status) VALUES (?, ?, ?)',
                               (timestamp, num_vehicles, density_percentage))
                conn.commit()
                conn.close()
            except sqlite3.Error as e:
                print(f"Error inserting into database: {e}")

        out.write(processed_frame)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    out.release()


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

        thread = Thread(target=process_and_save, args=(filepath, processed_path))
        thread.start()

        return redirect(url_for('index', filename=filename))
    return redirect(request.url)

def process_and_save(video_path, processed_path):
    for _ in process_video(video_path, processed_path):
        pass

@app.route('/video_feed')
def video_feed():
    filename = request.args.get('filename')
    if filename:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        processed_filename = f'processed_{filename}'
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
        return Response(process_video(video_path, processed_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    return "No video found.", 404

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
