"""
Aerial Object Detection — YOLO11n on VisDrone2019 / Multi-YOLO Upgraded App
Single Streamlit app for image and video inference.
"""

import os
import tempfile

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from inference import detect_image, detect_video
from model_loader import load_model, MODEL_CONFIG

st.set_page_config(
    page_title="Autonom · Streamlit Demo",
    page_icon="□",
    layout="wide",
    initial_sidebar_state="expanded",
)

SITE_URL = "https://uzaairmalik.github.io/Autonom/"

if "counts" not in st.session_state:
    st.session_state.counts = {}
if "frame_history" not in st.session_state:
    st.session_state.frame_history = []
if "inf_time" not in st.session_state:
    st.session_state.inf_time = 0.0
if "video_fps" not in st.session_state:
    st.session_state.video_fps = 0.0

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
      html, body, [class*='css'] { font-family: 'IBM Plex Sans', sans-serif; }
      .hero {
        background: linear-gradient(135deg, rgba(15,24,56,0.95), rgba(20,35,80,0.95));
        border: 1px solid rgba(99,179,237,0.25); border-radius: 20px; padding: 2rem 2.2rem; margin-bottom: 1.5rem;
      }
      .eyebrow { font-family: 'IBM Plex Mono', monospace; letter-spacing: .12em; text-transform: uppercase; color: #e8a33d; font-size: .72rem; }
      .title { font-size: clamp(2rem, 4vw, 2.7rem); font-weight: 800; margin: .35rem 0 .6rem; }
      .sub { color: #8b98b8; margin: 0; }
      .card {
        background: rgba(15,24,56,0.75); border: 1px solid rgba(233,237,247,0.10); border-radius: 16px; padding: 1rem; text-align: center;
      }
      .footer { text-align:center; color:#576082; font-family:'IBM Plex Mono', monospace; font-size:.78rem; margin-top:2rem; padding-top:1rem; border-top:1px solid rgba(233,237,247,0.08); }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:0.75rem 0 0.5rem;'><div style='font-size:2rem;'>□</div><div style='font-family:IBM Plex Mono, monospace;font-weight:700;color:#e8a33d;'>VISDRONE // MULTI-YOLO</div><div style='font-size:.75rem;color:#8b98b8;'>Upgraded Streamlit demo</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Model Selection")
    selected_model_name = st.selectbox(
        "Choose YOLO Model",
        options=list(MODEL_CONFIG.keys()),
        index=0,
        help="Select between the local trained model or Hugging Face YOLOv26 model sizes."
    )

    # Show selected model information
    cfg = MODEL_CONFIG[selected_model_name]
    st.markdown(
        f"""
        <div style="background-color: rgba(232,163,61,0.08); border: 1px solid rgba(232,163,61,0.3); border-radius: 8px; padding: 8px; margin-bottom: 15px; font-size: 0.85rem;">
          <strong>Type:</strong> {cfg['type'].upper()}<br/>
          <strong>Parameters:</strong> {cfg['params']}<br/>
          <strong>Input Size:</strong> {cfg['imgsz']}<br/>
          <strong>Description:</strong> {cfg['desc']}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Settings")
    conf = st.slider("Confidence", 0.10, 0.90, 0.25, 0.05)
    iou = st.slider("IoU (video NMS)", 0.10, 0.90, 0.45, 0.05)
    st.markdown("### Links")
    st.markdown(f"[Project overview]({SITE_URL})")

# Dynamically load the selected YOLO model (cached using Streamlit cache_resource)
yolo_model = load_model(selected_model_name)

st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">Live Demo // Multi-Model Support</div>
      <div class="title">Aerial Object Detection Platform</div>
      <p class="sub">Currently Running: <strong>{selected_model_name}</strong>. Support for local model and Hugging Face YOLOv26 variants (n, s, m, l, x) on VisDrone categories.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

CLASS_ICONS = {
    "car": "🚗",
    "truck": "🚛",
    "bus": "🚌",
    "pedestrian": "🚶",
    "people": "🚶",
    "bicycle": "🚲",
    "motor": "🏍️",
    "van": "🚐",
    "tricycle": "🛺",
    "awning-tricycle": "🛺",
}


def get_icon(name: str) -> str:
    return CLASS_ICONS.get(name.lower(), "📦")


tab_image, tab_video, tab_analytics, tab_eval = st.tabs(["Image Demo", "Video Demo", "Analytics", "Model Evaluation"])

with tab_image:
    uploaded_image = st.file_uploader(
        "Drop an aerial image here",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )

    if uploaded_image is None:
        st.info("Upload an aerial image to get started.")
    else:
        image = Image.open(uploaded_image)
        source_name = uploaded_image.name
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown(f"**Input · {source_name}**")
            st.image(image, use_container_width=True)

        suffix = os.path.splitext(source_name)[1].lower() or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        result_path = None
        counts = {}
        try:
            result_path, counts, inf_time = detect_image(yolo_model, tmp_path, "outputs/result.jpg", conf)
            st.session_state.counts = counts
            st.session_state.frame_history = []
            st.session_state.inf_time = inf_time
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        with col2:
            st.markdown("**Annotated output**")
            if result_path and os.path.exists(result_path):
                st.image(result_path, use_container_width=True)
                with open(result_path, "rb") as file_handle:
                    st.download_button(
                        "Download annotated image",
                        file_handle,
                        file_name="aerial_detection_result.jpg",
                        mime="image/jpeg",
                    )
            else:
                st.warning("No annotated image was generated.")

        st.markdown("#### Detections")
        if counts:
            total = sum(counts.values())
            cols = st.columns(min(len(counts), 6))
            for index, (cls_name, count) in enumerate(sorted(counts.items(), key=lambda item: -item[1])):
                with cols[index % len(cols)]:
                    st.markdown(
                        f'<div class="card"><div>{get_icon(cls_name)}</div><div style="font-size:1.6rem;font-weight:700;">{count}</div><div style="text-transform:capitalize;">{cls_name}</div></div>',
                        unsafe_allow_html=True,
                    )
            st.caption(f"{total} detected object-instances · conf ≥ {conf:.0%}")
        else:
            st.info("No detections above the current confidence threshold — try lowering it in the sidebar.")

with tab_video:
    uploaded_video = st.file_uploader(
        "Drop an aerial video here",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
    )

    if uploaded_video is None:
        st.info("Upload an aerial video to start tracking objects across frames.")
    else:
        video_bytes = uploaded_video.read()
        source_name = uploaded_video.name
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown(f"**Input · {source_name}**")
            st.video(video_bytes)

        suffix = os.path.splitext(source_name)[1].lower() or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        progress_bar = st.progress(0, text="Starting…")
        output_path = "outputs/result.mp4"

        try:
            generator = detect_video(yolo_model, tmp_path, output_path, conf, iou)
            last_fps = 0.0
            while True:
                frame_idx, total_frames, frame_counts, current_fps = next(generator)
                st.session_state.counts = frame_counts
                last_fps = current_fps
                pct = int((frame_idx + 1) / max(total_frames, 1) * 100)
                progress_bar.progress(
                    min(pct, 100),
                    text=f"Frame {frame_idx + 1}/{total_frames} · {current_fps:.1f} fps",
                )
        except StopIteration as stop:
            if stop.value:
                output_path, counts, frame_history = stop.value
                st.session_state.counts = counts
                st.session_state.frame_history = frame_history
                st.session_state.video_fps = last_fps
        except RuntimeError as error:
            st.error(f"Video processing failed: {error}")
        finally:
            progress_bar.empty()
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        with col2:
            if os.path.exists(output_path):
                st.markdown("**Annotated output**")
                st.video(output_path)
                with open(output_path, "rb") as file_handle:
                    st.download_button(
                        "Download annotated video",
                        file_handle,
                        file_name="aerial_detection_result.mp4",
                        mime="video/mp4",
                    )
            else:
                st.warning("No annotated video was generated.")

        st.markdown("#### Detections")
        if st.session_state.counts:
            total = sum(st.session_state.counts.values())
            cols = st.columns(min(len(st.session_state.counts), 6))
            for index, (cls_name, count) in enumerate(sorted(st.session_state.counts.items(), key=lambda item: -item[1])):
                with cols[index % len(cols)]:
                    st.markdown(
                        f'<div class="card"><div>{get_icon(cls_name)}</div><div style="font-size:1.6rem;font-weight:700;">{count}</div><div style="text-transform:capitalize;">{cls_name}</div></div>',
                        unsafe_allow_html=True,
                    )
            st.caption(f"{total} tracked object-instances across all frames · conf ≥ {conf:.0%}")
        else:
            st.info("No detections above the current confidence threshold — try lowering it in the sidebar.")

with tab_analytics:
    st.subheader("Analytics Dashboard")

    # Add real-time performance indicator section
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    with perf_col1:
        st.metric("Selected Model", selected_model_name)
    with perf_col2:
        if st.session_state.get("inf_time", 0.0) > 0:
            st.metric("Image Inference Time", f"{st.session_state.inf_time:.3f} seconds")
        else:
            st.metric("Image Inference Time", "N/A")
    with perf_col3:
        if st.session_state.get("video_fps", 0.0) > 0:
            st.metric("Video Inference Speed", f"{st.session_state.video_fps:.1f} FPS")
        else:
            st.metric("Video Inference Speed", "N/A")

    st.markdown("---")

    if st.session_state.counts:
        series = pd.Series(st.session_state.counts).sort_values(ascending=False)
        fig = px.bar(
            series,
            x=series.index,
            y=series.values,
            labels={"x": "Class", "y": "Count"},
            title="Current detection counts",
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        if st.session_state.frame_history:
            df = pd.DataFrame(st.session_state.frame_history).fillna(0)
            df["Frame"] = df.index
            df_melt = df.melt(id_vars=["Frame"], var_name="Class", value_name="Count")
            fig_time = px.area(df_melt, x="Frame", y="Count", color="Class", title="Cumulative object counts over time")
            st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Run an image or video demo first to populate analytics.")

with tab_eval:
    st.subheader("Model Evaluation")
    st.write("Use the Streamlit demo to inspect qualitative results. For offline metrics, run the evaluation scripts in the repository.")
    metrics = st.columns(3)
    metrics[0].metric("mAP@50", "32.4%")
    metrics[1].metric("Precision", "44.9%")
    metrics[2].metric("Recall", "34.7%")

st.markdown("<div class='footer'>Autonom · Streamlit-only demo</div>", unsafe_allow_html=True)
