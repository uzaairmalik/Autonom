from ultralytics import YOLO
import cv2
import os
from collections import Counter
from typing import Generator

import time

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
        tuple: (output_path, counts_dict, inference_time) where counts_dict maps class name -> count.
               Returns (None, {}, 0.0) if no results produced.
    """
    start_time = time.time()
    results = model(image_path, conf=conf)
    end_time = time.time()
    inference_time = end_time - start_time

    for r in results:
        annotated = r.plot()
        names = r.names
        classes = r.boxes.cls.tolist() if r.boxes is not None else []
        counts = Counter([names[int(c)] for c in classes])

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, annotated)

        return output_path, dict(counts), inference_time

    return None, {}, 0.0


def detect_video(
    video_path: str,
    output_path: str = "outputs/result.mp4",
    conf: float = 0.25,
) -> Generator[tuple[int, int, dict, float], None, tuple[str, dict, list]]:
    """
    Run YOLO object detection on every frame of a video and write annotated output.

    Yields (current_frame, total_frames, frame_counts, fps) for progress reporting.
    Returns (output_path, aggregate_counts, frame_counts_history) when complete.

    Args:
        video_path  : Path to the input video file.
        output_path : Where to save the annotated output video.
        conf        : YOLO confidence threshold.

    Yields:
        (current_frame: int, total_frames: int, frame_counts: dict, fps: float)

    Returns:
        (output_path: str, aggregate_counts: dict, frame_counts_history: list)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Use mp4v codec — universally compatible with Streamlit's st.video()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))

    aggregate: Counter = Counter()
    frame_counts_history = []
    frame_idx = 0

    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, conf=conf, verbose=False)

        for r in results:
            annotated = r.plot()
            writer.write(annotated)

            names = r.names
            classes = r.boxes.cls.tolist() if r.boxes is not None else []
            frame_counts = Counter([names[int(c)] for c in classes])
            aggregate.update(frame_counts)
            
            frame_counts_history.append(dict(frame_counts))

            elapsed = time.time() - start_time
            current_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0

            yield frame_idx, total_frames, dict(frame_counts), current_fps

        frame_idx += 1

    cap.release()
    writer.release()

    return output_path, dict(aggregate), frame_counts_history
