"""
Autonom
Multi-YOLO Object Detection Platform
VisDrone2019
"""

import os
import tempfile
import time

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

# Session state initialization
if "counts" not in st.session_state:
    st.session_state.counts = {}
if "frame_history" not in st.session_state:
    st.session_state.frame_history = []
if "inf_time" not in st.session_state:
    st.session_state.inf_time = 0.0
if "video_fps" not in st.session_state:
    st.session_state.video_fps = 0.0
if "avg_conf" not in st.session_state:
    st.session_state.avg_conf = 0.0
if "total_objects" not in st.session_state:
    st.session_state.total_objects = 0

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
        "<div style='text-align:center;padding:0.75rem 0 0.5rem;'><div style='font-size:2rem;'>□</div><div style='font-family:IBM Plex Mono, monospace;font-weight:700;color:#e8a33d;'>AUTONOM // MULTI-YOLO</div><div style='font-size:.75rem;color:#8b98b8;'>VisDrone2019 Platform</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Model Selection")
    selected_model_name = st.selectbox(
        "Choose YOLO Model",
        options=list(MODEL_CONFIG.keys()),
        index=0,
        help="Select between the local trained model or Hugging Face YOLOv26 model sizes."
    )

    cfg = MODEL_CONFIG[selected_model_name]
    st.markdown(
        f"""
        <div style="background-color: rgba(232,163,61,0.08); border: 1px solid rgba(232,163,61,0.3); border-radius: 8px; padding: 8px; margin-bottom: 15px; font-size: 0.85rem;">
          <strong>Type:</strong> {cfg['type'].upper()}<br/>
          <strong>Parameters:</strong> {cfg['params']}<br/>
          <strong>Input Size:</strong> {cfg['imgsz']}<br/>
          <strong>mAP@50:</strong> {cfg['mAP']}<br/>
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
      <div class="eyebrow">Autonom // Multi-YOLO Object Detection Platform</div>
      <div class="title">VisDrone2019 Aerial Scene Understanding</div>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-top: 1.5rem;">
        <div style="background: rgba(255,255,255,0.06); padding: 0.8rem 1.2rem; border-radius: 12px; border-left: 4px solid #e8a33d;">
          <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #8b98b8; text-transform: uppercase;">Current Model</div>
          <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-top: 2px;">{selected_model_name}</div>
        </div>
        <div style="background: rgba(255,255,255,0.06); padding: 0.8rem 1.2rem; border-radius: 12px; border-left: 4px solid #48bb78;">
          <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #8b98b8; text-transform: uppercase;">Model Parameters</div>
          <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-top: 2px;">{cfg['params']}</div>
        </div>
        <div style="background: rgba(255,255,255,0.06); padding: 0.8rem 1.2rem; border-radius: 12px; border-left: 4px solid #3182ce;">
          <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #8b98b8; text-transform: uppercase;">Input Resolution</div>
          <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-top: 2px;">{cfg['imgsz']}</div>
        </div>
        <div style="background: rgba(255,255,255,0.06); padding: 0.8rem 1.2rem; border-radius: 12px; border-left: 4px solid #9f7aea;">
          <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #8b98b8; text-transform: uppercase;">Dataset Target</div>
          <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff; margin-top: 2px;">VisDrone2019</div>
        </div>
      </div>
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
            result_path, counts, inf_time, avg_conf, total_objects = detect_image(yolo_model, tmp_path, "outputs/result.jpg", conf)
            st.session_state.counts = counts
            st.session_state.frame_history = []
            st.session_state.inf_time = inf_time
            st.session_state.avg_conf = avg_conf
            st.session_state.total_objects = total_objects
        finally:
            # We don't remove tmp_path yet if we want to run benchmarking on the same file!
            # Let's save tmp_path into session_state so the benchmarking function can reuse it safely.
            st.session_state.last_image_path = tmp_path

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

        # Presentation-ready dynamic metrics container
        st.markdown("### 📈 Active Detection Metrics")
        m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
        m_col1.metric("Inference Time", f"{st.session_state.inf_time:.3f} s")
        m_col2.metric("Detected Objects", f"{st.session_state.total_objects}")
        m_col3.metric("Total Classes", f"{len(st.session_state.counts)}")
        m_col4.metric("Avg Confidence", f"{st.session_state.avg_conf:.1%}")
        m_col5.metric("Model Size", f"{cfg['params']}")
        m_col6.metric("Input Resolution", f"{cfg['imgsz']}")

        st.markdown("---")

        # Benchmark All Models Option
        st.markdown("### 🏆 Multi-YOLO Performance Benchmarking")
        st.write("Compare all supported models side-by-side using the uploaded image. This evaluates accuracy and speed in real-time.")

        if st.button("🚀 Run Benchmark All Models", use_container_width=True):
            if "last_image_path" in st.session_state and os.path.exists(st.session_state.last_image_path):
                with st.spinner("Executing real-time inference benchmark across all 6 models..."):
                    benchmark_rows = []
                    for model_name, model_cfg in MODEL_CONFIG.items():
                        m_instance = load_model(model_name)
                        start_bt = time.time()
                        # Run single-image detection via Ultralytics
                        res_bt = m_instance(st.session_state.last_image_path, conf=conf, verbose=False)
                        elapsed_bt = time.time() - start_bt

                        num_objs = 0
                        avg_cf = 0.0
                        if res_bt and len(res_bt) > 0:
                            r_bt = res_bt[0]
                            if r_bt.boxes is not None:
                                num_objs = len(r_bt.boxes)
                                confs_bt = r_bt.boxes.conf.tolist()
                                avg_cf = sum(confs_bt) / len(confs_bt) if confs_bt else 0.0

                        benchmark_rows.append({
                            "Model": model_name,
                            "Time (s)": round(elapsed_bt, 3),
                            "Objects Detected": num_objs,
                            "Avg Confidence": f"{avg_cf:.1%}",
                            "Winner": ""
                        })

                    # Highlight the model with the highest object count
                    max_objs = -1
                    winner_i = -1
                    for idx, row in enumerate(benchmark_rows):
                        if row["Objects Detected"] > max_objs:
                            max_objs = row["Objects Detected"]
                            winner_i = idx

                    if winner_i != -1:
                        benchmark_rows[winner_i]["Winner"] = "🏆 Winner"

                    st.session_state.benchmark_data = pd.DataFrame(benchmark_rows)
            else:
                st.error("Please ensure the uploaded image has loaded successfully before benchmarking.")

        if "benchmark_data" in st.session_state:
            st.dataframe(st.session_state.benchmark_data, use_container_width=True, hide_index=True)
            w_row = st.session_state.benchmark_data[st.session_state.benchmark_data["Winner"] == "🏆 Winner"]
            if not w_row.empty:
                st.success(f"🎉 **{w_row.iloc[0]['Model']}** detected the highest number of objects ({w_row.iloc[0]['Objects Detected']}) and is crowned the Benchmark Winner!")

        # Clean up temporary path if session ends or is reset
        if "last_image_path" in st.session_state and not uploaded_image:
            if os.path.exists(st.session_state.last_image_path):
                os.remove(st.session_state.last_image_path)

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
                st.session_state.total_objects = sum(counts.values())
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

        # Presentation-ready dynamic video metrics container
        st.markdown("### 📈 Active Video Metrics")
        v_col1, v_col2, v_col3, v_col4 = st.columns(4)
        v_col1.metric("Video FPS", f"{st.session_state.video_fps:.1f} FPS")
        v_col2.metric("Tracked Instances", f"{st.session_state.total_objects}")
        v_col3.metric("Model Size", f"{cfg['params']}")
        v_col4.metric("Input Resolution", f"{cfg['imgsz']}")

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

    # Advanced stats calculation for the current run
    most_common_class = "N/A"
    detected_classes_count = 0
    if st.session_state.counts:
        detected_classes_count = len(st.session_state.counts)
        most_common_class = max(st.session_state.counts, key=st.session_state.counts.get).capitalize()

    # Dynamic metrics display
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    perf_col1.metric("Selected Model", selected_model_name)
    perf_col2.metric("Detected Classes", f"{detected_classes_count}")
    perf_col3.metric("Average Confidence", f"{st.session_state.avg_conf:.1%}" if st.session_state.avg_conf > 0 else "N/A")
    perf_col4.metric("Most Common Object", most_common_class)

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
    st.subheader("Model Evaluation Metrics")
    st.write(f"Showing validated offline benchmarks for **{selected_model_name}** on the VisDrone2019 validation set.")
    metrics_cols = st.columns(3)
    metrics_cols[0].metric("mAP@50", cfg["mAP"])
    metrics_cols[1].metric("Precision", cfg["Precision"])
    metrics_cols[2].metric("Recall", cfg["Recall"])

st.markdown("<div class='footer'>Autonom · Streamlit-only demo</div>", unsafe_allow_html=True)
