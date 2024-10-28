from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import pyautogui  # For simulating keypress

app = Flask(__name__)

# Load the pre-trained model
with open('models/final_model_3dcnn_lstm.json', 'r') as json_file:
    loaded_json_model = json_file.read()
model = model_from_json(loaded_json_model)
model.load_weights("models/final_model_3dcnn_lstm.h5")
print("Model loaded successfully")

# Gesture classes and actions
classes = ['No gesture', 'Swiping Left', 'Swiping Right', 'Thumb Down', 'Thumb Up', 'Zooming In With Full Hand', 'Zooming Out With Full Hand']
gesture_actions = {"Swiping Left": 'right', "Swiping Right": 'left', "Zooming In With Full Hand": 'esc', "Zooming Out With Full Hand": 'esc'}

# Helper variables
num_frames = 0
frame_buffer = []
cooldown = 0
key_pressed = None
current_prediction = "No gesture"  # To store the latest predicted gesture

def predict_gesture(frame_sequence):
    inp_video = np.asarray(frame_sequence)[None, :]
    output = model.predict(inp_video, verbose=0)[0][1:]
    index = np.argmax(output)
    return classes[index], output

def handle_key_action(label):
    global key_pressed
    if label in gesture_actions:
        action_key = gesture_actions[label]
        if key_pressed != action_key:
            if key_pressed:  # Release previously pressed key
                pyautogui.keyUp(key_pressed)
            pyautogui.keyDown(action_key)
            key_pressed = action_key
    else:
        if key_pressed:
            pyautogui.keyUp(key_pressed)
            key_pressed = None

def generate_frames():
    global num_frames, frame_buffer, cooldown, current_prediction

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Camera could not be opened.")
        return
    
    while True:
        grabbed, frame = camera.read()
        if not grabbed:
            break
        frame = cv2.flip(frame, 1)
        resized_frame = cv2.resize(frame, (32, 32))

        if num_frames < 36:
            frame_buffer.append(resized_frame)
        else:
            frame_buffer.pop(0)
            frame_buffer.append(resized_frame)

        if num_frames >= 36 and (num_frames - 36) % 18 == 0 and cooldown == 0:
            current_prediction, output = predict_gesture(frame_buffer)
            handle_key_action(current_prediction)
            cooldown = 15

        if cooldown > 0:
            cooldown -= 1
        num_frames += 1

        # Encode frame for live video streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    return jsonify(prediction=current_prediction)

if __name__ == '__main__':
    app.run(debug=True)
