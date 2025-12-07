# Classical Face Recognition using MediaPipe + LBPH (AI Without Machine Learning)

This project implements a full face recognition pipeline using **only classical computer vision techniques** – no deep learning, no neural networks.

- **Face Detection**: MediaPipe Face Detection  
- **Face Recognition**: OpenCV LBPHFaceRecognizer (Local Binary Patterns Histograms)  

The system successfully detects and correctly identifies **at least two different people** in real time.

## Demo Video
See `demo_video.mp4` (20–40 seconds)  
The video clearly shows:
- Real-time face detection with bounding boxes
- Correct recognition of Person 1
- Correct recognition of Person 2
- Unknown faces labeled as "Unknown"

## Project Structure
face-recognition-lbph-mediapipe/
├── capture.py              # Collect face samples from webcam
├── train.py                # Train LBPH model and save it
├── predict.py              # Real-time recognition (use this to record demo)
├── models/
│   ├── lbph_model.yml      # Trained LBPH model
│   └── label_map.json      # ID → Name mapping
├── dataset/                # One folder per person with grayscale face images
├── demo_video.mp4          # Assignment submission video
├── requirements.txt
└── README.md               # This file
text## How to Run

### 1. Install requirements
```bash
pip install -r requirements.txt
2. Capture training images
Bashpython capture.py

Type the person's name (use underscores, e.g. john_doe)
Press SPACE to capture each face
Capture 30–50 images per person
Press q to quit

3. Train the model
Bashpython train.py
4. Run live recognition (use this to record your submission video)
Bashpython predict.py
Assignment Requirements – All Completed

MediaPipe detection + classical LBPH recognition
Recognizes ≥2 people correctly
Includes trained model and label map
Public GitHub repo with clear instructions
20–40 second demo video included

Runs completely offline on CPU · Zero deep learning used
Submitted: December 2025
Student: TUYISHIME Christian