1. 
FILE PATHS
```
FILE PATHS
├── CMPT419 Dataset/
│   ├── Anger/anger mp4 files
│   ├── Neutral/neutral mp4 files
│   └── openface_output/ openface csv annotations
├── static/
│   └── styles/
│       └── SFU.jpg
├── templates/
│   ├── index.html
│   └── result.html
├── uploads/
│   ├── Anger1.mp4
│   ├── Anger1.5.mp4
│   ├── Anger1m.mp4
│   ├── Neutral1.mp4
│   ├── Neutral1.4.mp4
│   └── Neutral1m.mp4
├── app.py
├── baseline_model.ipynb
├── cnn_lstm_anger_classifier.h5
├── cnn_lstm_model.ipynb
├── readme
└── requirements.txt
```


EXPLANATION OF FILES: 

CMF419 Dataset/: Contains training and testing video data organized into Anger/ and Neutral/ folders.

openface_output/: Stores the extracted AU features from OpenFace for model training.

static/: Holds static assets for the web app, including images and stylesheets.

templates/: Contains HTML templates for the web interface (index.html, result.html).

uploads/: Sample .mp4 videos provided for demo/testing in the web app — these can be dragged into the app to see results.

app.py: Main script to run the web app.

baseline_model.ipynb: Jupyter notebook for training and testing the OpenFace AU + LSTM baseline model.

cnn_lstm_model.ipynb: Jupyter notebook for the CNN → LSTM model pipeline.

cnn_lstm_anger_classifier.h5: Saved model weights for the CNN → LSTM classifier.

readme: The project README file.

requirements.txt: List of required Python packages.

2.
Our aim was to build an emotion-based break recommendation system that analyzes gameplay footage to detect moments of frustration or anger and recommend breaks when needed. Early on, we intended to use DeepFace’s frame-by-frame emotion classification, specifically its anger confidence scores, as a quick baseline to determine emotional intensity throughout a video. The idea was to track these scores over time and suggest a break if sustained anger was detected. However, we quickly realized DeepFace is designed for static image analysis — it doesn't consider temporal context or how emotions evolve across frames — so it wasn’t a fair or meaningful comparison for full video-based emotion analysis. Because of this limitation, we moved away from using DeepFace as a baseline and instead focused on developing two full video pipelines that could be compared fairly in an ablation study.

The first pipeline was our baseline model, which used OpenFace to extract facial action units (AUs) from each frame. These AU features were passed into an LSTM to capture how facial muscle movements associated with emotions evolved over time. The second pipeline — our main model — used a CNN (MobileNetV2) to extract spatial features directly from each frame of the video, which were then fed into an LSTM to learn temporal patterns of emotion. This allowed us to directly compare the impact of feature extraction methods: hand-crafted AUs from OpenFace versus learned features from raw video frames. Through our ablation study, we found that the CNN → LSTM model performed slightly better overall, likely due to its ability to learn richer and more flexible visual representations.

In terms of following our proposal, we stuck to our core goal and implemented the frustration detection system based on gameplay recordings. We made some changes to our initial plan: we originally considered using OpenCV and MediaPipe for expression detection but switched to OpenFace for more accurate AU annotations. We also didn’t reach the 200 sample target due to time and data availability, so we focused on a smaller, curated dataset with Anger and Neutral clips to keep things manageable and consistent. One thing to note for grading is that there wasn’t really an existing solution that was directly comparable to what we were trying to build, so we had to figure out a lot of things manually and make decisions on the fly throughout the project. Despite that, we were able to build two working pipelines, run a fair comparison between them, and stay aligned with our original goal. We also bundled OpenFace into the project to make everything easier to run without extra setup.

3.
Steps to set up our app
   1. Make sure python version is 3.10

   2. Install the required packages using requirements.txt.

   3. Run app.py.

   4. Go to the URL shown in the terminal (usually http://127.0.0.1:5000).

   5. To demo the app, we added some .mp4 videos in the upload/ folder — you can drag and drop those directly into the interface.
