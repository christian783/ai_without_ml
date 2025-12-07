import cv2
import mediapipe as mp
import json
import numpy as np

mp_face_detection = mp.solutions.face_detection

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/lbph_model.yml")

with open("models/label_map.json") as f:
    label_map = json.load(f)
label_map = {int(k): v for k, v in label_map.items()}

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                face_crop = frame[y:y+h, x:x+w]
                gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (200, 200))

                label_id, confidence = recognizer.predict(gray)

                if confidence < 100:  # good match
                    name = label_map.get(label_id, "Unknown")
                    text = f"{name} ({confidence:.1f})"
                    color = (0, 255, 0)
                else:
                    text = f"Unknown ({confidence:.1f})"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow('Face Recognition - LBPH + MediaPipe', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()