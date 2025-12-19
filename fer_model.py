from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO("./best.onnx", task="detect")

# Emotion classes expected from the model
emotion_classes = {0: "angry", 1: "fear", 2: "happy", 3: "neutral", 4: "sad"}

def predict_emotion(image):
    # Ensure 3 channels 
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    results = model(image)
    if not results or len(results) == 0:
        return {"emotion": "none", "confidence": 0.0}

    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return {"emotion": "none", "confidence": 0.0}

    # Get best prediction (highest confidence)
    confs = boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))

    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confidences = confs

    emotion_idx = cls_ids[best_idx]
    confidence = round(float(confidences[best_idx]), 2)
    emotion = emotion_classes.get(emotion_idx, "unknown")

    # Save image with bounding box for debugging
    xyxy = boxes.xyxy[best_idx].cpu().numpy().astype(int)
    x1, y1, x2, y2 = xyxy
    annotated = image.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(annotated, f"{emotion} ({confidence:.2f})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    os.makedirs("face_data/annotated", exist_ok=True)
    cv2.imwrite("face_data/annotated/latest_detected.jpg", annotated)

    return {
        "emotion": emotion,
        "confidence": confidence
    }
