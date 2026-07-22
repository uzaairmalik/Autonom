import os
import streamlit as st
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Configuration for the supported models - strictly the requested 5 models
MODEL_CONFIG = {
    "My Model": {
        "type": "local",
        "path": "models/best10classes.pt",
        "architecture": "YOLOv11",
        "dataset": "VisDrone2019",
        "params": "3.0M",
        "imgsz": "640x640",
        "desc": "Custom 10-class model trained on VisDrone aerial dataset.",
        "metrics": {
            "mAP50": "32.4%",
            "Precision": "44.9%",
            "Recall": "34.7%"
        }
    },
    "YOLOv8x": {
        "type": "standard",
        "path": "yolov8x.pt",
        "architecture": "YOLOv8",
        "dataset": "COCO",
        "params": "68.2M",
        "imgsz": "640x640",
        "desc": "YOLOv8 Extra Large - Standard highly-accurate general object detector.",
        "metrics": {
            "mAP50": "36.5%",
            "Precision": "49.8%",
            "Recall": "38.5%"
        }
    },
    "YOLOv9e": {
        "type": "standard",
        "path": "yolov9e.pt",
        "architecture": "YOLOv9",
        "dataset": "COCO",
        "params": "58.1M",
        "imgsz": "640x640",
        "desc": "YOLOv9 Extended - Advanced programmable relation/information-rich model.",
        "metrics": {
            "mAP50": "37.8%",
            "Precision": "51.5%",
            "Recall": "40.2%"
        }
    },
    "YOLOv10x": {
        "type": "standard",
        "path": "yolov10x.pt",
        "architecture": "YOLOv10",
        "dataset": "COCO",
        "params": "31.6M",
        "imgsz": "640x640",
        "desc": "YOLOv10 Extra Large - Real-time end-to-end detector with dual label assignment.",
        "metrics": {
            "mAP50": "37.2%",
            "Precision": "50.9%",
            "Recall": "39.7%"
        }
    },
    "YOLOv26x": {
        "type": "hf",
        "repo_id": "openvision/yolo26-x",
        "filename": "model.pt",
        "architecture": "YOLOv26",
        "dataset": "VisDrone2019",
        "params": "55.7M",
        "imgsz": "640x640",
        "desc": "YOLOv26 Extra Large - NMS-free state-of-the-art accuracy variant.",
        "metrics": {
            "mAP50": "38.3%",
            "Precision": "52.9%",
            "Recall": "41.1%"
        }
    }
}

@st.cache_resource(show_spinner="Loading YOLO model weights...")
def load_model(model_name: str) -> YOLO:
    """
    Downloads and caches a YOLO model based on the selected model name.

    Args:
        model_name (str): Identifier from MODEL_CONFIG keys.

    Returns:
        YOLO: Loaded Ultralytics YOLO model instance.
    """
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Unknown model identifier: {model_name}. Must be one of {list(MODEL_CONFIG.keys())}")

    config = MODEL_CONFIG[model_name]

    if config["type"] == "local":
        path = config["path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Local model weights not found at: {path}")
        return YOLO(path)

    elif config["type"] == "standard":
        # Standard ultralytics models are downloaded automatically by ultralytics YOLO()
        return YOLO(config["path"])

    elif config["type"] == "hf":
        repo_id = config["repo_id"]
        filename = config["filename"]
        # Download weights from Hugging Face Hub
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return YOLO(model_path)

    else:
        raise ValueError(f"Invalid model type in configuration for {model_name}")
