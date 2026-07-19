"""
Aerial Object Detection — YOLO11n on VisDrone2019
Streamlit demo: video detection with multi-object tracking.

Images are handled by the static site (demo.html) — this app focuses on
video, where server-side inference + ByteTrack tracking actually earns its
keep over a browser-only pipeline.

CSC354 — Machine Learning semester project.
"""

import os
import tempfile
import time
from collections import Counter

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

from inference import detect_video

st.set_page_config(
    page_title="Aerial Detection — Video Demo",
    page_icon="\u25a2",
    layout="wide",
    initial_sidebar_state="expanded",
)

SITE_URL = "https://uzaairmalik.github.io/Autonom/"
IMAGE_DEMO_URL = "https://uzaairmalik.github.io/Autonom/demo.html"

# ==========================================================================
# Theme system — light / dark, toggled at runtime via session_state
# ==========================================================================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

THEMES = {
    "dark": {
        "bg": "#0a0f1c", "panel": "#111a2e", "panel_alt": "#0d1526",
        "ink": "#e9edf7", "muted": "#8b98b8", "faint": "#576082",
        "accent": "#e8a33d", "accent_soft": "rgba(232,163,61,0.14)",
        "line": "rgba(233,237,247,0.10)", "danger": "#c1554a",
        "good": "#5ba087", "sidebar_bg": "#0d1526",
    },
    "light": {
        "bg": "#f7f5f0", "panel": "#ffffff", "panel_alt": "#f0ede4",
        "ink": "#1a1a2e", "muted": "#5b6073", "faint": "#93968f",
        "accent": "#c07d2e", "accent_soft": "rgba(192,125,46,0.12)",
        "line": "rgba(26,26,46,0.10)", "danger": "#a8443a",
        "good": "#3f7d63", "sidebar_bg": "#efece3",
    },
}
T = THEMES[st.session_state.theme]

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}
    .stApp {{ background: {T['bg']}; }}
    [data-testid="stSidebar"] {{ background: {T['sidebar_bg']}; border-right: 1px solid {T['line']}; }}
    [data-testid="stSidebar"] * {{ color: {T['ink']} !important; }}
    h1, h2, h3, h4 {{ font-family: 'IBM Plex Mono', monospace !important; color: {T['ink']} !important; }}
    p, span, label, .stMarkdown {{ color: {T['ink']}; }}
    .stCaption, [data-testid="stCaptionContainer"] {{ color: {T['muted']} !important; }}

    .eyebrow {{
        font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; letter-spacing: 0.14em;
        text-transform: uppercase; color: {T['accent']}; margin-bottom: 0.4rem;
    }}
    .eyebrow::before {{ content: "// "; color: {T['faint']}; }}

    .hero-box {{
        background: {T['panel']}; border: 1px solid {T['line']}; border-radius: 4px;
        padding: 2rem 2.2rem; margin-bottom: 1.6rem; position: relative;
    }}
    .hero-box::before {{ content:""; position:absolute; top:-1px; left:-1px; width:16px; height:16px;
        border-top:2px solid {T['accent']}; border-left:2px solid {T['accent']}; }}
    .hero-box::after {{ content:""; position:absolute; bottom:-1px; right:-1px; width:16px; height:16px;
        border-bottom:2px solid {T['accent']}; border-right:2px solid {T['accent']}; }}
    .hero-title {{ font-size: 1.9rem; font-weight: 700; color: {T['ink']}; margin: 0 0 0.4rem; font-family: 'IBM Plex Mono', monospace; }}
    .hero-sub {{ color: {T['muted']}; font-size: 0.98rem; margin: 0; max-width: 640px; }}
    .back-link {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: {T['muted']}; text-decoration: none; }}

    [data-testid="stFileUploader"] {{
        background: {T['panel']} !important; border: 2px dashed {T['line']} !important;
        border-radius: 4px !important; padding: 1.4rem !important;
    }}

    .stat-card {{
        background: {T['panel']}; border: 1px solid {T['line']}; border-radius: 3px;
        padding: 1rem 1rem; text-align: center;
    }}
    .stat-num {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.7rem; font-weight: 700; color: {T['accent']}; line-height: 1.1; }}
    .stat-lbl {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: {T['muted']}; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 0.25rem; }}

    .info-card {{
        background: {T['accent_soft']}; border: 1px solid {T['accent']}; border-radius: 4px;
        padding: 1.1rem 1.4rem; margin: 1rem 0;
    }}
    .info-card a {{ color: {T['accent']}; font-weight: 600; text-decoration: none; }}

    .section-divider {{ height: 1px; background: {T['line']}; margin: 1.8rem 0; border: none; }}
    .footer {{ text-align: center; color: {T['faint']}; font-size: 0.78rem; padding: 1.6rem 0 0.6rem;
        border-top: 1px solid {T['line']}; margin-top: 2.4rem; font-family: 'IBM Plex Mono', monospace; }}
    .footer span {{ color: {T['accent']}; }}

    .stButton > button, .stDownloadButton > button {{
        font-family: 'IBM Plex Mono', monospace; border-radius: 2px; border: 1px solid {T['accent']};
    }}
    .stButton > button[kind="primary"], .stDownloadButton > button {{
        background: {T['accent']}; color: {T['bg']}; font-weight: 600;
    }}
    .stTabs [data-baseweb="tab"] {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================================
# Sidebar
# ==========================================================================
with st.sidebar:
    st.markdown(
        f'<div style="text-align:center;padding:0.6rem 0 0.8rem;">'
        f'<div style="font-size:1.6rem;">\u25a2</div>'
        f'<div style="font-size:1rem;font-weight:700;color:{T["accent"]};margin-top:0.3rem;font-family:\'IBM Plex Mono\',monospace;">VISDRONE // YOLO11n</div>'
        f'<div style="font-size:0.72rem;color:{T["muted"]};margin-top:0.2rem;">Video Detection Demo</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    theme_choice = st.radio("Appearance", ["Dark", "Light"], horizontal=True,
                             index=0 if st.session_state.theme == "dark" else 1)
    new_theme = theme_choice.lower()
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

    st.markdown('<div class="eyebrow">Detection settings</div>', unsafe_allow_html=True)
    conf = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)
    iou = st.slider("IoU threshold (NMS)", 0.10, 0.90, 0.45, 0.05)

    st.markdown('<div class="eyebrow">Model</div>', unsafe_allow_html=True)
    st.caption("YOLO11n fine-tuned on VisDrone2019-DET")
    st.caption("mAP@50: 32.4% \u00b7 mAP@50-95: 18.4%")
    st.caption("Precision: 44.9% \u00b7 Recall: 34.7%")

    st.markdown('<div class="eyebrow">Elsewhere</div>', unsafe_allow_html=True)
    st.markdown(f'[\u2190 Project overview]({SITE_URL})')
    st.markdown(f'[Image detection (instant, browser-only)]({IMAGE_DEMO_URL})')

# ==========================================================================
# Header
# ==========================================================================
st.markdown(f'<a class="back-link" href="{SITE_URL}" target="_blank">&larr; Back to overview</a>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="hero-box">
        <div class="eyebrow">Live Demo // Server-Side Inference + ByteTrack</div>
        <div class="hero-title">Video detection with stable, tracked bounding boxes</div>
        <p class="hero-sub">Upload aerial footage and run the trained YOLO11n detector with ByteTrack multi-object
        tracking — this is what keeps boxes steady across frames instead of flickering. Need a quick single-image
        check instead? <a href="{IMAGE_DEMO_URL}" target="_blank" style="color:{T['accent']};">Use the instant browser demo</a>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

CLASS_ICONS = {
    "car": "\U0001F697", "truck": "\U0001F69B", "bus": "\U0001F68C", "pedestrian": "\U0001F6B6",
    "people": "\U0001F6B6", "bicycle": "\U0001F6B2", "motor": "\U0001F3CD\uFE0F",
    "van": "\U0001F690", "tricycle": "\U0001F6FA", "awning-tricycle": "\U0001F6FA",
}
def get_icon(name: str) -> str:
    return CLASS_ICONS.get(name.lower(), "\U0001F4E6")

tab_demo, tab_analytics, tab_eval = st.tabs(["Video Demo", "Analytics", "Model Evaluation"])

# ==========================================================================
# TAB: Video Demo
# ==========================================================================
with tab_demo:
    uploaded_file = st.file_uploader(
        "Drop an aerial video here", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is not None:
        video_bytes = uploaded_file.read()
        source_name = uploaded_file.name

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown(f"**Input \u00b7 {source_name}**")
            st.video(video_bytes)

        ext_suffix = os.path.splitext(source_name)[1].lower() or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext_suffix) as tmp_vid:
            tmp_vid.write(video_bytes)
            tmp_vid_path = tmp_vid.name

        output_vid_path = "outputs/result.mp4"
        progress_bar = st.progress(0, text="Starting\u2026")
        counts = {}
        frame_history = []

        try:
            gen = detect_video(tmp_vid_path, output_vid_path, conf, iou)
            while True:
                frame_idx, total_frames, frame_counts, current_fps = next(gen)
                counts = frame_counts
                pct = int((frame_idx + 1) / max(total_frames, 1) * 100)
                progress_bar.progress(min(pct, 100), text=f"Frame {frame_idx+1}/{total_frames} \u00b7 {current_fps:.1f} fps")
        except StopIteration as e:
            if e.value:
                output_vid_path, counts, frame_history = e.value
                st.session_state.counts = counts
                st.session_state.frame_history = frame_history
        except RuntimeError as e:
            st.error(f"Video processing failed: {e}")
            counts = {}
        finally:
            progress_bar.empty()
            if os.path.exists(tmp_vid_path):
                os.remove(tmp_vid_path)

        with col2:
            st.markdown("**Annotated output (tracked)**")
            if os.path.exists(output_vid_path):
                with open(output_vid_path, "rb") as f:
                    out_bytes = f.read()
                st.video(out_bytes)
                st.download_button("Download annotated video", out_bytes, "aerial_detection_result.mp4", "video/mp4")
            else:
                st.warning("Could not generate annotated video.")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("#### Detections")
        if counts:
            total = sum(counts.values())
            cols = st.columns(min(len(counts), 6))
            for i, (cls_name, cnt) in enumerate(sorted(counts.items(), key=lambda x: -x[1])):
                with cols[i % len(cols)]:
                    st.markdown(
                        f'<div class="stat-card"><div>{get_icon(cls_name)}</div>'
                        f'<div class="stat-num">{cnt}</div><div class="stat-lbl">{cls_name}</div></div>',
                        unsafe_allow_html=True,
                    )
            st.caption(f"{total} tracked object-instances across all frames \u00b7 conf \u2265 {conf:.0%}")
        else:
            st.info("No detections above the current confidence threshold — try lowering it in the sidebar.")
    else:
        st.markdown(
            f"""
            <div style="text-align:center;padding:3.5rem 2rem;color:{T['faint']};">
                <div style="font-size:3rem;margin-bottom:0.8rem;">\u25a2</div>
                <div style="font-size:1.05rem;font-weight:500;color:{T['muted']};">Upload an aerial video to get started</div>
                <div style="font-size:0.85rem;margin-top:0.4rem;">MP4, AVI, MOV, or MKV \u00b7 processed frame-by-frame with tracking</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="info-card">Need a single image checked instead? '
            f'<a href="{IMAGE_DEMO_URL}" target="_blank">Use the instant browser-based demo \u2192</a></div>',
            unsafe_allow_html=True,
        )

# ==========================================================================
# TAB: Analytics
# ==========================================================================
with tab_analytics:
    st.markdown("#### Detection Analytics")
    if "counts" in st.session_state and st.session_state.counts:
        if st.session_state.get("frame_history"):
            st.markdown("**Object counts over time**")
            df = pd.DataFrame(st.session_state.frame_history).fillna(0)
            df["Frame"] = df.index
            df_melt = df.melt(id_vars=["Frame"], var_name="Class", value_name="Count")
            fig = px.area(df_melt, x="Frame", y="Count", color="Class")
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color=T["ink"])
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Class distribution**")
        df_dist = pd.DataFrame(list(st.session_state.counts.items()), columns=["Class", "Count"]).sort_values("Count", ascending=False)
        fig_bar = px.bar(df_dist, x="Class", y="Count", color="Class")
        fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color=T["ink"])
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Run a detection in the Video Demo tab first.")

# ==========================================================================
# TAB: Model Evaluation
# ==========================================================================
with tab_eval:
    st.markdown("#### Model Evaluation Metrics")
    st.caption("From the actual training run — 100 epochs, VisDrone2019-DET validation split (548 images, 38,759 instances).")

    c1, c2, c3 = st.columns(3)
    c1.metric("mAP@50", "32.4%")
    c2.metric("Precision", "44.9%")
    c3.metric("Recall", "34.7%")
    c4, c5, c6 = st.columns(3)
    c4.metric("mAP@50-95", "18.4%")
    c5.metric("F1-score", "39.1%")
    c6.metric("Inference speed", "65.0 FPS")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Per-class AP@50**")
    per_class_df = pd.DataFrame({
        "Class": ["car", "bus", "van", "motor", "pedestrian", "truck", "people", "tricycle", "awning-tricycle", "bicycle"],
        "AP@50 (%)": [75.0, 46.5, 36.8, 36.3, 34.7, 29.1, 26.2, 20.8, 10.5, 8.5],
    })
    fig_ap = px.bar(per_class_df, x="AP@50 (%)", y="Class", orientation="h",
                     color="AP@50 (%)", color_continuous_scale=[T["danger"], T["accent"]])
    fig_ap.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color=T["ink"], yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_ap, use_container_width=True)
    st.caption(
        "`car` (75.0%) and `bicycle` (8.5%) are the two extremes — the overall mean is pulled down "
        "by rare, visually small classes rather than uniform weakness across the model."
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    possible_paths = ["confusion_matrix.png", "models/confusion_matrix.png"]
    cm_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if cm_path:
        st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
    else:
        st.info("Place your training run's confusion matrix at `confusion_matrix.png` to display it here.")

st.markdown(
    f'<div class="footer">Runs server-side via Ultralytics YOLO + ByteTrack \u00b7 '
    f'<span>Semester Project</span> \u00b7 CSC354 Machine Learning</div>',
    unsafe_allow_html=True,
)
