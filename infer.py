#!/usr/bin/env python3
"""
infer.py — Local CLI inference script for VisDrone YOLO11n model.
For personal use only. Run detections on images or whole folders.

Usage examples:
  # Single image
  python infer.py --input sample_images/drone1.jpg

  # Whole folder of images
  python infer.py --input sample_images/

  # Custom confidence threshold + custom output folder
  python infer.py --input sample_images/ --conf 0.35 --output my_results/

  # Show result window (requires a display)
  python infer.py --input sample_images/drone1.jpg --show
"""

import argparse
import os
import sys
from pathlib import Path
import time

# ── imports ──────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics not installed. Run:  pip install ultralytics")

try:
    import cv2
except ImportError:
    sys.exit("opencv-python not installed. Run:  pip install opencv-python")

# ── VisDrone class names ──────────────────────────────────────────────────────
CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

# ── accepted extensions ───────────────────────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args():
    p = argparse.ArgumentParser(description="Local YOLO11n inference — VisDrone")
    p.add_argument("--input",  "-i", required=True,
                   help="Path to an image, video, or folder of images/videos.")
    p.add_argument("--output", "-o", default="outputs/cli",
                   help="Folder to save annotated results. Default: outputs/cli/")
    p.add_argument("--model",  "-m", default="models/best.pt",
                   help="Path to the YOLO .pt model. Default: models/best.pt")
    p.add_argument("--conf",   "-c", type=float, default=0.25,
                   help="Confidence threshold (0–1). Default: 0.25")
    p.add_argument("--iou",          type=float, default=0.45,
                   help="NMS IoU threshold. Default: 0.45")
    p.add_argument("--imgsz",        type=int,   default=640,
                   help="Inference image size. Default: 640")
    p.add_argument("--show",  action="store_true",
                   help="Display annotated result in a window (requires display).")
    p.add_argument("--no-save", action="store_true",
                   help="Do NOT save annotated output files.")
    return p.parse_args()


def collect_files(input_path: Path):
    """Return a sorted list of image/video paths from a file or directory."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = []
        for f in sorted(input_path.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS:
                files.append(f)
        return files
    sys.exit(f"[ERROR] Input path not found: {input_path}")


def print_summary(results, elapsed: float):
    """Pretty-print per-class detection counts from a YOLO Results object."""
    counts = {}
    for box in results.boxes:
        cls_id = int(box.cls[0])
        name = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
        counts[name] = counts.get(name, 0) + 1

    total = sum(counts.values())
    print(f"  ├─ Detections : {total}")
    print(f"  ├─ Inference  : {elapsed*1000:.1f} ms")
    if counts:
        for cls_name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            bar = "█" * min(cnt, 40)
            print(f"  │   {cls_name:<18} {bar} {cnt}")
    else:
        print("  │   (no objects above confidence threshold)")


def run_image(model, src: Path, out_dir: Path, args):
    print(f"\n[IMAGE] {src.name}")
    t0 = time.perf_counter()
    results = model(str(src), conf=args.conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
    elapsed = time.perf_counter() - t0

    print_summary(results, elapsed)

    if not args.no_save:
        out_path = out_dir / src.name
        annotated = results.plot()         # numpy array (BGR)
        cv2.imwrite(str(out_path), annotated)
        print(f"  └─ Saved → {out_path}")

    if args.show:
        cv2.imshow(src.name, results.plot())
        print("     Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video(model, src: Path, out_dir: Path, args):
    print(f"\n[VIDEO] {src.name}")
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"  [WARN] Could not open video: {src}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    raw_path = None
    out_path = None
    if not args.no_save:
        out_path = out_dir / (src.stem + "_detected.mp4")
        raw_path = out_dir / (src.stem + "_detected.raw.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(str(raw_path), fourcc, fps, (width, height))

    # Reset any tracker state left over from a previous call on this model instance.
    model.predictor = None

    frame_idx  = 0
    total_dets = 0
    t_start    = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # model.track(persist=True) keeps a per-object track ID alive across
        # frames (ByteTrack), which is what stops boxes from flickering —
        # plain per-frame model() calls have no memory between frames.
        results  = model.track(
            frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
            persist=True, tracker="bytetrack.yaml", verbose=False,
        )[0]
        annotated = results.plot()
        total_dets += len(results.boxes)
        frame_idx  += 1

        if writer:
            writer.write(annotated)

        if args.show:
            cv2.imshow(src.name, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("  [INFO] Stopped by user (q pressed).")
                break

        # Progress every 30 frames
        if frame_idx % 30 == 0:
            pct = frame_idx / max(total, 1) * 100
            sys.stdout.write(f"\r  Processing frame {frame_idx}/{total} ({pct:.0f}%)  ")
            sys.stdout.flush()

    elapsed = time.perf_counter() - t_start
    cap.release()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"\n  ├─ Frames     : {frame_idx}")
    print(f"  ├─ Total dets : {total_dets}")
    print(f"  ├─ Avg FPS    : {frame_idx/elapsed:.1f}")
    if not args.no_save:
        final_path = _reencode_for_browser(str(raw_path), str(out_path))
        print(f"  └─ Saved → {final_path}")


def _reencode_for_browser(raw_path: str, output_path: str) -> str:
    """Re-encode with ffmpeg to H.264/yuv420p for reliable playback. Falls
    back to the raw mp4v file if ffmpeg isn't installed."""
    import subprocess
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_path, "-vcodec", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "23", output_path],
            check=True, capture_output=True,
        )
        if os.path.exists(raw_path):
            os.remove(raw_path)
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  [WARN] ffmpeg not found — output kept as mp4v, may not play in all players.")
        return raw_path


def main():
    args     = parse_args()
    src      = Path(args.input)
    out_dir  = Path(args.output)
    files    = collect_files(src)

    if not files:
        sys.exit(f"[ERROR] No image or video files found in: {src}")

    # Load model once
    print(f"\nLoading model: {args.model}")
    model = YOLO(args.model)
    print(f"Model loaded. Classes: {', '.join(CLASSES)}")
    print(f"Settings: conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}")
    print(f"Files to process: {len(files)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        ext = f.suffix.lower()
        if ext in IMAGE_EXTS:
            run_image(model, f, out_dir, args)
        elif ext in VIDEO_EXTS:
            run_video(model, f, out_dir, args)
        else:
            print(f"[SKIP] Unsupported file type: {f.name}")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
