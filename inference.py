from ultralytics import YOLO
import cv2
import os
from collections import Counter
from typing import Generator

import time

MODEL_PATH = "models/newbest.pt"

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
    iou: float = 0.45,
) -> Generator[tuple[int, int, dict, float], None, tuple[str, dict, list]]:
    """
    Run YOLO object detection + ByteTrack tracking on every frame of a video
    and write an annotated output video.

    Uses model.track(persist=True) rather than plain per-frame model() calls —
    this gives each object a stable track ID across frames, which is what makes
    the boxes hold steady instead of flickering/jittering frame to frame. Plain
    per-frame detection has no memory between frames, so a single missed frame
    (very common with a modest-mAP model) makes a box vanish and reappear.

    Yields (current_frame, total_frames, frame_counts, fps) for progress reporting.
    Returns (output_path, aggregate_counts, frame_counts_history) when complete.

    Args:
        video_path  : Path to the input video file.
        output_path : Where to save the annotated output video.
        conf        : YOLO confidence threshold.
        iou         : NMS IoU threshold.

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

    # Write with mp4v first (safe, always available); re-encoded to H.264 below
    # for actual browser playback — mp4v itself is NOT reliably browser-playable.
    raw_path = output_path + ".raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_path, fourcc, video_fps, (width, height))

    # Reset any tracker state left over from a previous call on this model instance.
    model.predictor = None

    aggregate: Counter = Counter()
    frame_counts_history = []
    frame_idx = 0

    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(
            frame,
            persist=True,
            conf=conf,
            iou=iou,
            tracker="bytetrack.yaml",
            verbose=False,
        )

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

    final_path = _reencode_for_browser(raw_path, output_path)

    return final_path, dict(aggregate), frame_counts_history


def _reencode_for_browser(raw_path: str, output_path: str) -> str:
    """
    Re-encode with ffmpeg to H.264/yuv420p, which browsers (and Streamlit's
    st.video) reliably play back. Falls back to the raw mp4v file — with a
    printed warning — if ffmpeg isn't available on the host.
    """
    import subprocess

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", raw_path,
                "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                output_path,
            ],
            check=True, capture_output=True,
        )
        if os.path.exists(raw_path):
            os.remove(raw_path)
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(
            "[WARN] ffmpeg re-encode failed or ffmpeg not installed — "
            "output video may not play in all browsers. Add `ffmpeg` to "
            "packages.txt on Streamlit Cloud, or `apt install ffmpeg` locally."
        )
        return raw_path
