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
#vertices1 = np.array([(43,324), (200,15), (279,18), (531,320)], dtype=np.int32) for hikevision
vertices1 = np.array([(478,2049), (1508,431), (2613,418), (3820,2093)], dtype=np.int32)

# Define traffic thresholds
HEAVY_TRAFFIC_THRESHOLD = 10
DENSITY_THRESHOLD = 8  # Threshold for density classification

# Initialize the YOLOv8 model
try:
    best_model = YOLO('models/best.pt')  # Ensure this path is correct
except AttributeError as e:
    print(f"Error loading YOLO model: {e}")
    best_model = None  # Prevent processing if the model fails to load

# Import the database initialization function
from init_db import init_db

# Run database initialization at startup
init_db(DATABASE)

# Check for valid file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """
    Renders the home page with an option to upload a video.
    Displays the video stream if a filename is provided.
    """
    filename = request.args.get('filename')
    return render_template('index.html', filename=filename)

@app.route('/analytics')
def analytics():
    """
    Renders the analytics page displaying vehicle counts and density statuses.
    """
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, vehicle_count, density_status 
            FROM traffic_data 
            ORDER BY id DESC 
            LIMIT 100
        ''')
        data = cursor.fetchall()
        conn.close()
        data = data[::-1]  # Reverse to have chronological order
        return render_template('analytics.html', data=data)
    except sqlite3.OperationalError as e:
        print(f"Database error: {e}")
        return f"Database error: {e}", 500

def process_video(video_path, processed_path):
    """
    Processes the video frame by frame, detects vehicles within the defined region,
    counts them, determines density status, and streams the processed video.
    """
    if best_model is None:
        print("YOLO model is not loaded. Exiting video processing.")
        return

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default to 30 if FPS cannot be retrieved

    # Get frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_path, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print("Error: Could not open VideoWriter.")
        cap.release()
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detection_frame = frame.copy()

        # Perform inference using YOLOv8
        try:
            results = best_model.predict(detection_frame, imgsz=640, conf=0.4)
        except AttributeError as e:
            print(f"Error during model prediction: {e}")
            break
        except Exception as e:
            print(f"Unexpected error during model prediction: {e}")
            break

        # Plot the detection results on the frame
        processed_frame = results[0].plot(line_width=1)

        # Draw the defined region quadrilateral
        cv2.polylines(processed_frame, [vertices1], isClosed=True, color=(0, 255, 0), thickness=2)

        # Initialize vehicle count for this frame
        vehicle_count = 0

        # Iterate over detected bounding boxes
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0

            # Ensure coordinates are floats
            x_center = float(x_center)
            y_center = float(y_center)

            # Check if the vehicle center is within the defined region
            if cv2.pointPolygonTest(vertices1, (x_center, y_center), False) >= 0:
                vehicle_count += 1

        # Determine traffic density status
        density_status = "Heavy" if vehicle_count > DENSITY_THRESHOLD else "Smooth"

        # Add text annotations for vehicle count and density status
        cv2.putText(processed_frame, f'Vehicles: {vehicle_count}', (frame_width - 250, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  # Red color
        cv2.putText(processed_frame, f'Density: {density_status}', (frame_width - 250, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  # Red color

        # Insert data into the database periodically (e.g., every second)
        frame_count += 1
        if frame_count % int(fps) == 0:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            try:
                conn = sqlite3.connect(DATABASE)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO traffic_data 
                    (timestamp, vehicle_count, density_status)
                    VALUES (?, ?, ?)
                ''', (timestamp, vehicle_count, density_status))
                conn.commit()
                conn.close()
            except sqlite3.Error as e:
                print(f"Error inserting into database: {e}")

        # Write the processed frame to the output video
        out.write(processed_frame)

        # Convert processed frame to JPEG and yield for streaming
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
    """
    Handles video file uploads, saves them, clears existing database data,
    and starts processing the video in a separate thread.
    """
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        # Secure the filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save the uploaded file
        file.save(filepath)

        # Clear existing database data
        try:
            conn = sqlite3.connect(DATABASE)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM traffic_data')
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            print(f"Error clearing database: {e}")

        # Define the processed video path
        processed_filename = f'processed_{filename}'
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)

        # Remove existing processed video if it exists
        if os.path.exists(processed_path):
            os.remove(processed_path)

        # Start processing the video in the background
        thread = Thread(target=process_and_save, args=(filepath, processed_path))
        thread.start()

        return redirect(url_for('index', filename=filename))
    return redirect(request.url)

def process_and_save(video_path, processed_path):
    """
    Wrapper function to process the video. This is used to run process_video in a separate thread.
    """
    for _ in process_video(video_path, processed_path):
        pass  # All processing is handled in process_video

@app.route('/video_feed')
def video_feed():
    """
    Streams the processed video to the client.
    """
    filename = request.args.get('filename')
    if filename:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        processed_filename = f'processed_{filename}'
        processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
        return Response(process_video(video_path, processed_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    return "No video found.", 404

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
