# video_break_app.py

import os
import cv2
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from werkzeug.utils import secure_filename

# Set Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'  # where uploaded videos will go
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load MobileNetV2 to extract features from frames
cnn_base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
cnn_model = Sequential([
    cnn_base,
    GlobalAveragePooling2D()  # this flattens the CNN output
])

# Loading my trained LSTM model that uses those features
lstm_model = load_model("cnn_lstm_anger_classifier.h5")

# Function pulls out 30 evenly spaced frames from the video
def extract_frames(video_path, count=30):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, total_frames - 1, count).astype(int)
    frames = []
    for idx in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))  # resize to MobileNet input
        frame = frame.astype("float32") / 255.0  # normalize
        frames.append(frame)
    cap.release()
    return np.array(frames)

# Decide if the user should take a break or not
def get_break_suggestion(video_path):
    frames = extract_frames(video_path)
    if len(frames) != 30:
        return "❌ Could not process enough frames. Try a longer clip."

    # Get CNN features for each frame
    features = cnn_model.predict(frames, verbose=0)
    features = np.expand_dims(features, axis=0)  # shape becomes (1, 30, feature_size)

    # Predict with LSTM model
    prediction = lstm_model.predict(features)[0][0]
    
    # Based on prediction, I return a helpful suggestion
    if prediction > 0.5:
        return f"😡 Detected elevated anger ({prediction:.2f}). You should take a break."
    else:
        return f"😌 Low anger detected ({prediction:.2f}). You're good to keep playing!"

# This is the main route where people upload their videos
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.mp4'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            suggestion = get_break_suggestion(filepath)
            return render_template('result.html', suggestion=suggestion)
    return render_template('index.html')

# run the app
if __name__ == '__main__':
    app.run(debug=True)
