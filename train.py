#!/usr/bin/env python3
"""
YOLO11 / YOLOv8 VisDrone Dataset Training Script
Optimized for the semester competition, featuring advanced data augmentations,
high-resolution training config, and automated ONNX model export for web deployment.

Usage examples:
  # Standard training using YOLO11m on GPU 0
  python train.py --model yolo11m.pt --epochs 100 --imgsz 640 --batch 16 --device 0

  # High-resolution training (better for tiny aerial objects, requires more VRAM)
  python train.py --model yolo11m.pt --epochs 150 --imgsz 1024 --batch 8 --device 0

  # Resume training from a checkpoint
  python train.py --resume --model runs/detect/train/weights/last.pt
"""

import argparse
import sys
import os
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERROR] ultralytics package not found. Run: pip install ultralytics")

# Standard VisDrone classes for reference
CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="🚀 YOLO VisDrone Training & Optimization for Semester Competition"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11m.pt",
        help="Pretrained YOLO model weights or path to checkpoint (e.g., yolo11n.pt, yolo11m.pt, yolo11l.pt).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="VisDrone.yaml",
        help="Path to the dataset .yaml configuration file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train. (Default: 100)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference/Training image size. (640 or 1024 recommended for VisDrone).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for training. Adjust based on VRAM (e.g., 4, 8, 16, 32).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader CPU workers. (Default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="CUDA device index or 'cpu' (e.g., '0', '0,1', 'cpu').",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Directory to save experimental training runs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="visdrone_comp",
        help="Name of the training run folder.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint of the specified --model path.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience (epochs with no improvement). (Default: 30)",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Disable automatic export of the best weights to ONNX format upon completion.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("🛸 AUTONOMOUS DRONE PROJECT - YOLO VISDRONE TRAINING HUB 🛸")
    print(f"   Model Weights:       {args.model}")
    print(f"   Dataset Config:      {args.data}")
    print(f"   Total Epochs:        {args.epochs}")
    print(f"   Image Resolution:    {args.imgsz}x{args.imgsz}")
    print(f"   Batch Size:          {args.batch}")
    print(f"   Device Selected:     {args.device}")
    print(f"   Saving To:           {args.project}/{args.name}")
    print("=" * 70)

    # Load model
    if args.resume:
        print(f"🔁 Resuming training from checkpoint: {args.model}")
        if not os.path.exists(args.model):
            sys.exit(f"[ERROR] Checkpoint path does not exist: {args.model}")
        model = YOLO(args.model)
    else:
        print(f"🚀 Loading pretrained model weights: {args.model}")
        model = YOLO(args.model)

    # Advanced Training Hyperparameters optimized for small, dense VisDrone objects
    training_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "resume": args.resume,
        "patience": args.patience,
        "save": True,
        "save_period": 5,
        "cos_lr": True,          # Cosine learning rate scheduler
        # High quality data augmentations to prevent overfitting on busy aerial scenes
        "mosaic": 1.0,           # Full mosaic augmentation
        "mixup": 0.15,           # Blend images for robust representations
        "copy_paste": 0.2,       # Copy objects between frames for small target learning
        "degrees": 10.0,         # Slight rotations
        "translate": 0.15,       # Slight shifts
        "scale": 0.5,            # Scale variance
        "fliplr": 0.5,           # Flip horizontal
    }

    try:
        # Launch model training
        print("\nStarting training pipeline... Grab a coffee, this will take some time! ☕")
        results = model.train(**training_kwargs)
        print("\n🏆 Model training complete successfully!")

    except Exception as e:
        print(f"\n[ERROR] An error occurred during training: {e}", file=sys.stderr)
        sys.exit(1)

    # Automatic Export of best checkpoint to ONNX format
    if not args.no_export:
        print("\n" + "-" * 70)
        print("📦 AUTOMATED POST-TRAINING DEPLOYMENT EXPORT")
        print("-" * 70)

        # Determine paths
        best_pt_path = Path(args.project) / args.name / "weights" / "best.pt"
        if not best_pt_path.exists():
            # Try finding the actual final path in ultralytics' output
            best_pt_path = Path(model.trainer.save_dir) / "weights" / "best.pt"

        if best_pt_path.exists():
            print(f"Found best PyTorch weights at: {best_pt_path}")
            print("Exporting model to ONNX format for browser-side web deployment (imgsz=640)...")
            try:
                # Load the newly trained best weights
                trained_model = YOLO(str(best_pt_path))
                # Export with imgsz=640 to match expectations in app.js
                onnx_path = trained_model.export(format="onnx", imgsz=640, simplify=True)
                print(f"✅ Success! ONNX model exported to: {onnx_path}")
                print("Copy this ONNX file to your 'models/' folder to power your web interface (app.js)!")
            except Exception as e:
                print(f"[WARN] Automatic ONNX export failed: {e}")
                print("You can manually export the weights using: yolo export model=path/to/best.pt format=onnx imgsz=640")
        else:
            print("[WARN] Best PyTorch weights (best.pt) file could not be found for ONNX export.")

    print("\n✓ Done. Best of luck in the semester competition! 🛸🥇")


if __name__ == "__main__":
    main()
