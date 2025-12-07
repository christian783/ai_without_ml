import cv2
import mediapipe as mp
import os
import sys

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Get person's name from user
person_name = input("Enter the person's name (use underscores for spaces): ").strip()
folder = f"dataset/{person_name}"
os.makedirs(folder, exist_ok=True)

# Initialize video capture
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
) as face_detection:
    count = 0
    print("Press SPACE to capture a face, 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape

                # Convert relative coordinates to absolute pixel values
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                # Draw bounding box around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Capture face when SPACE is pressed
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    face_crop = frame[y:y + h, x:x + w]
                    face_crop = cv2.resize(face_crop, (200, 200))
                    face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(f"{folder}/{count:04d}.jpg", face_gray)
                    count += 1
                    print(f"Saved {count} images")

        # Display the video feed
        cv2.imshow('Capture faces - Press SPACE', frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print(f"Captured {count} images for {person_name}")