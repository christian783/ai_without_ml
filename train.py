import cv2
import os
import numpy as np
import json
from pathlib import Path

dataset_path = "dataset"
faces = []
labels = []
label_map = {}
current_label = 0

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person_name.replace("_", " ")
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(current_label)
    current_label += 1

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

os.makedirs("models", exist_ok=True)
recognizer.save("models/lbph_model.yml")

with open("models/label_map.json", "w") as f:
    json.dump(label_map, f)

print("Training finished!")
print("Labels:", label_map)