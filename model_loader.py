import os
import streamlit as st
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Configuration for the supported models
MODEL_CONFIG = {
    "Local (best10classes.pt)": {
        "type": "local",
        "path": "models/best10classes.pt",
        "desc": "Custom 10-class model trained on VisDrone dataset.",
        "params": "3.0M",
        "imgsz": "640x640",
        "mAP": "32.4%",
        "Precision": "44.9%",
        "Recall": "34.7%"
    },
    "YOLOv26n": {
        "type": "hf",
        "repo_id": "openvision/yolo26-n",
        "filename": "model.pt",
        "desc": "YOLOv26 Nano - NMS-free, ultra-fast, optimized for edge CPU deployment.",
        "params": "2.4M",
        "imgsz": "640x640",
        "mAP": "40.9%",
        "Precision": "51.2%",
        "Recall": "39.8%"
    },
    "YOLOv26s": {
        "type": "hf",
        "repo_id": "openvision/yolo26-s",
        "filename": "model.pt",
        "desc": "YOLOv26 Small - NMS-free, excellent balance between speed and accuracy.",
        "params": "9.5M",
        "imgsz": "640x640",
        "mAP": "48.6%",
        "Precision": "59.1%",
        "Recall": "46.5%"
    },
    "YOLOv26m": {
        "type": "hf",
        "repo_id": "openvision/yolo26-m",
        "filename": "model.pt",
        "desc": "YOLOv26 Medium - NMS-free, powerful real-time detector.",
        "params": "20.4M",
        "imgsz": "640x640",
        "mAP": "53.1%",
        "Precision": "63.4%",
        "Recall": "51.0%"
    },
    "YOLOv26l": {
        "type": "hf",
        "repo_id": "openvision/yolo26-l",
        "filename": "model.pt",
        "desc": "YOLOv26 Large - NMS-free, high-capacity model for complex scenarios.",
        "params": "24.8M",
        "imgsz": "640x640",
        "mAP": "55.0%",
        "Precision": "65.2%",
        "Recall": "53.2%"
    },
    "YOLOv26x": {
        "type": "hf",
        "repo_id": "openvision/yolo26-x",
        "filename": "model.pt",
        "desc": "YOLOv26 Extra Large - NMS-free, maximum accuracy variant.",
        "params": "55.7M",
        "imgsz": "640x640",
        "mAP": "57.5%",
        "Precision": "68.0%",
        "Recall": "55.8%"
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

    elif config["type"] == "hf":
        repo_id = config["repo_id"]
        filename = config["filename"]
        # Download weights from Hugging Face Hub
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        return YOLO(model_path)

    else:
        raise ValueError(f"Invalid model type in configuration for {model_name}")
