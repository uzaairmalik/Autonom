from ultralytics import YOLO
import cv2
import os
from collections import Counter

MODEL_PATH = "models/best.pt"

model = YOLO(MODEL_PATH)


def detect_image(image_path, output_path="outputs/result.jpg", conf=0.25):
    """
    Run YOLO object detection on an aerial image.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the annotated output image.
        conf (float): Confidence threshold for detections.

    Returns:
        tuple: (output_path, counts_dict) where counts_dict maps class name -> count.
               Returns (None, {}) if no results produced.
    """
    results = model(image_path, conf=conf)

    for r in results:
        annotated = r.plot()
        names = r.names
        classes = r.boxes.cls.tolist() if r.boxes is not None else []
        counts = Counter([names[int(c)] for c in classes])

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated)

        return output_path, dict(counts)

    return None, {}
