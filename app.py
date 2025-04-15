from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

app = Flask(__name__)

# Load model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: model.p not found!")
    model = None

# Get label mapping
try:
    data_dict = pickle.load(open('./data.pickle', 'rb'))
    unique_labels = sorted(set(data_dict['labels']))
    labels_dict = {i: label for i, label in enumerate(unique_labels)}
    print(f"Labels loaded: {labels_dict}")
except FileNotFoundError:
    print("Error: data.pickle not found!")
    labels_dict = {}

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Constants
NUM_LANDMARKS = 21
FEATURES_PER_LANDMARK = 2
MAX_HANDS = 2
EXPECTED_FEATURE_LENGTH = NUM_LANDMARKS * FEATURES_PER_LANDMARK * MAX_HANDS  # 84

# For rate limiting predictions (avoid flickering)
last_prediction_time = 0
prediction_interval = 0.5  # seconds
current_prediction = "None"

camera = None

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def process_frame(frame):
    global current_prediction, last_prediction_time
    
    # Process with MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = []
    x_ = []
    y_ = []

    # Draw hand landmarks on frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks[:MAX_HANDS]:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x_.append(hand_landmarks.landmark[i].x)
                y_.append(hand_landmarks.landmark[i].y)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(hand_landmarks.landmark[i].x - min(x_))
                data_aux.append(hand_landmarks.landmark[i].y - min(y_))

    # Make prediction if we have features and enough time has passed
    current_time = time.time()
    if len(data_aux) > 0 and (current_time - last_prediction_time) >= prediction_interval:
        # Ensure correct feature length
        while len(data_aux) < EXPECTED_FEATURE_LENGTH:
            data_aux.append(0.0)  # Pad missing values
            
        # Make prediction if model is loaded and features are correct
        if model is not None and len(data_aux) == EXPECTED_FEATURE_LENGTH:
            try:
                prediction = model.predict([np.asarray(data_aux)])[0]
                if isinstance(prediction, str):
                    current_prediction = prediction
                else:
                    predicted_label = int(prediction)
                    current_prediction = labels_dict.get(predicted_label, "Unknown")
                last_prediction_time = current_time
            except Exception as e:
                print(f"Prediction error: {e}")
    
    # Draw prediction on frame
    cv2.putText(frame, f"Sign: {current_prediction}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    return frame

def generate_frames():
    camera = get_camera()
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Process the frame
            processed_frame = process_frame(frame)
            
            # Convert to JPEG for streaming
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    return jsonify({'prediction': current_prediction})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    release_camera()
    return jsonify({'status': 'Camera released'})

if __name__ == '__main__':
    app.run(debug=True)