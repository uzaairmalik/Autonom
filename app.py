"""
Aerial Object Detection — YOLO11n on VisDrone2019
Streamlit demo for image and video detection with multi-object tracking.

This app is the single demo surface for the project: quick image checks and
video inference both live here.

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

from inference import detect_image, detect_video

st.set_page_config(
    page_title="Autonom · Streamlit Demo",
    page_icon="\u25a2",
    layout="wide",
    initial_sidebar_state="expanded",
)

SITE_URL = "https://uzaairmalik.github.io/Autonom/"
tab_image, tab_video, tab_analytics, tab_eval = st.tabs(["Image Demo", "Video Demo", "Analytics", "Model Evaluation"])

# ==================================================================
# TAB: Image Demo
# ==================================================================
with tab_image:
    uploaded_image = st.file_uploader(
        "Drop an aerial image here", type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        source_name = uploaded_image.name

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown(f"**Input · {source_name}**")
            st.image(image, use_container_width=True)

        ext_suffix = os.path.splitext(source_name)[1].lower() or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext_suffix) as tmp_img:
            image.save(tmp_img.name)
            tmp_img_path = tmp_img.name

        output_img_path = "outputs/result.jpg"
        """
        Aerial Object Detection — YOLO11n on VisDrone2019
        Streamlit demo for image and video detection with multi-object tracking.

        This app is the single demo surface for the project: quick image checks and
        video inference both live here.

        CSC354 — Machine Learning semester project.
        """

        import os
        import tempfile
        from collections import Counter

        import pandas as pd
        import plotly.express as px
        import streamlit as st
        from PIL import Image

        from inference import detect_image, detect_video

        st.set_page_config(
            page_title="Autonom · Streamlit Demo",
            page_icon="□",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        SITE_URL = "https://uzaairmalik.github.io/Autonom/"

        if "theme" not in st.session_state:
            st.session_state.theme = "dark"

        THEMES = {
            "dark": {
                "bg": "#0a0f1c",
                "panel": "#111a2e",
                "panel_alt": "#0d1526",
                "ink": "#e9edf7",
                "muted": "#8b98b8",
                "faint": "#576082",
                "accent": "#e8a33d",
                "accent_soft": "rgba(232,163,61,0.14)",
                "line": "rgba(233,237,247,0.10)",
                "danger": "#c1554a",
                "good": "#5ba087",
                "sidebar_bg": "#0d1526",
            },
            "light": {
                "bg": "#f7f5f0",
                "panel": "#ffffff",
                "panel_alt": "#f0ede4",
                "ink": "#1a1a2e",
                "muted": "#5b6073",
                "faint": "#93968f",
                "accent": "#c07d2e",
                "accent_soft": "rgba(192,125,46,0.12)",
                "line": "rgba(26,26,46,0.10)",
                "danger": "#a8443a",
                "good": "#3f7d63",
                "sidebar_bg": "#efece3",
            },
        }
        T = THEMES[st.session_state.theme]

        st.markdown(
            f"""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
            html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}
            .stApp {{ background: linear-gradient(135deg, {T['bg']} 0%, {T['panel_alt']} 55%, {T['bg']} 100%); min-height: 100vh; color: {T['ink']}; }}
            [data-testid="stSidebar"] {{ background: {T['sidebar_bg']}; border-right: 1px solid {T['line']}; }}
            [data-testid="stSidebar"] * {{ color: {T['ink']} !important; }}
            .hero-box {{
                background: linear-gradient(135deg, rgba(15,24,56,0.9) 0%, rgba(20,35,80,0.9) 100%);
                border: 1px solid rgba(99,179,237,0.25); border-radius: 20px; padding: 2.2rem 2.5rem;
                margin-bottom: 1.8rem; backdrop-filter: blur(20px);
            }}
            .hero-title {{ font-size: 2rem; font-weight: 800; margin: 0 0 0.6rem 0; }}
            .hero-sub {{ color: {T['muted']}; margin: 0; font-size: 1rem; }}
            .eyebrow {{ font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: {T['accent']}; margin-bottom: 0.8rem; }}
            .section-divider {{ height: 1px; background: linear-gradient(90deg, transparent, {T['line']}, transparent); margin: 2rem 0; }}
            .stat-card {{ background: linear-gradient(135deg, rgba(15,24,56,0.9) 0%, rgba(20,35,80,0.9) 100%); border: 1px solid {T['line']}; border-radius: 14px; padding: 1rem; text-align: center; margin-bottom: 0.7rem; }}
            .stat-num {{ font-size: 1.7rem; font-weight: 800; color: {T['accent']}; line-height: 1; }}
            .stat-lbl {{ font-size: 0.74rem; color: {T['muted']}; margin-top: 0.3rem; text-transform: capitalize; }}
            .footer {{ text-align: center; color: {T['faint']}; font-size: 0.78rem; padding: 1.5rem 0 0.8rem; border-top: 1px solid {T['line']}; margin-top: 2rem; }}
            .info-card {{ background: rgba(99,179,237,0.08); border: 1px solid rgba(99,179,237,0.25); border-radius: 14px; padding: 1rem 1.1rem; color: {T['ink']}; }}
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

        with st.sidebar:
            st.markdown(
                f'<div style="text-align:center;padding:0.6rem 0 0.8rem;">'
                f'<div style="font-size:1.6rem;">□</div>'
                f'<div style="font-size:1rem;font-weight:700;color:{T["accent"]};margin-top:0.3rem;font-family:\'IBM Plex Mono\',monospace;">VISDRONE // YOLO11n</div>'
                f'<div style="font-size:0.72rem;color:{T["muted"]};margin-top:0.2rem;">Image + video demo</div>'
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
            st.caption("mAP@50: 32.4% · mAP@50-95: 18.4%")
            st.caption("Precision: 44.9% · Recall: 34.7%")

            st.markdown('<div class="eyebrow">Elsewhere</div>', unsafe_allow_html=True)
            st.markdown(f'[← Project overview]({SITE_URL})')

        st.markdown(f'<a class="back-link" href="{SITE_URL}" target="_blank">&larr; Back to overview</a>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="hero-box">
                <div class="eyebrow">Live Demo // Server-Side Inference + ByteTrack</div>
                <div class="hero-title">Image and video detection with stable, tracked bounding boxes</div>
                <p class="hero-sub">Upload aerial images for quick detections or aerial footage for ByteTrack tracking. Everything now lives in this Streamlit app.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        CLASS_ICONS = {
            "car": "🚗", "truck": "🚛", "bus": "🚌", "pedestrian": "🚶",
            "people": "🚶", "bicycle": "🚲", "motor": "🏍️",
            "van": "🚐", "tricycle": "🛺", "awning-tricycle": "🛺",
        }


        def get_icon(name: str) -> str:
            return CLASS_ICONS.get(name.lower(), "📦")


        tab_image, tab_video, tab_analytics, tab_eval = st.tabs(["Image Demo", "Video Demo", "Analytics", "Model Evaluation"])

        with tab_image:
            uploaded_image = st.file_uploader("Drop an aerial image here", type=["jpg", "jpeg", "png"])

            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                source_name = uploaded_image.name

                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2, gap="large")

                with col1:
                    st.markdown(f"**Input · {source_name}**")
                    st.image(image, use_container_width=True)

                ext_suffix = os.path.splitext(source_name)[1].lower() or ".png"
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext_suffix) as tmp_img:
                    image.save(tmp_img.name)
                    tmp_img_path = tmp_img.name

                output_img_path = "outputs/result.jpg"
                counts = {}
                result_path = None

                try:
                    result_path, counts, inf_time = detect_image(tmp_img_path, output_img_path, conf)
                    st.session_state.counts = counts
                    st.session_state.inf_time = inf_time
                    st.session_state.frame_history = []
                finally:
                    if os.path.exists(tmp_img_path):
                        os.remove(tmp_img_path)

                with col2:
                    st.markdown("**Annotated output**")
                    if result_path and os.path.exists(result_path):
                        st.image(result_path, use_container_width=True)
                        with open(result_path, "rb") as f:
                            out_bytes = f.read()
                        st.download_button("Download annotated image", out_bytes, "aerial_detection_result.jpg", "image/jpeg")
                    else:
                        st.warning("Could not generate annotated image.")

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
                    st.caption(f"{total} detected object-instances · conf ≥ {conf:.0%}")
                else:
                    st.info("No detections above the current confidence threshold — try lowering it in the sidebar.")
            else:
                st.markdown(
                    f"""
                    <div style="text-align:center;padding:3.5rem 2rem;color:{T['faint']};">
                        <div style="font-size:3rem;margin-bottom:0.8rem;">□</div>
                        <div style="font-size:1.05rem;font-weight:500;color:{T['muted']};">Upload an aerial image to get started</div>
                        <div style="font-size:0.85rem;margin-top:0.4rem;">JPG or PNG · processed in the Streamlit app</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with tab_video:
            uploaded_file = st.file_uploader("Drop an aerial video here", type=["mp4", "avi", "mov", "mkv"])

            if uploaded_file is not None:
                video_bytes = uploaded_file.read()
                source_name = uploaded_file.name

                st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2, gap="large")

                with col1:
                    st.markdown(f"**Input · {source_name}**")
                    st.video(video_bytes)

                ext_suffix = os.path.splitext(source_name)[1].lower() or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext_suffix) as tmp_vid:
                    tmp_vid.write(video_bytes)
                    tmp_vid_path = tmp_vid.name

                output_vid_path = "outputs/result.mp4"
                progress_bar = st.progress(0, text="Starting…")
                counts = {}
                frame_history = []

                try:
                    gen = detect_video(tmp_vid_path, output_vid_path, conf, iou)
                    while True:
                        frame_idx, total_frames, frame_counts, current_fps = next(gen)
                        counts = frame_counts
                        pct = int((frame_idx + 1) / max(total_frames, 1) * 100)
                        progress_bar.progress(min(pct, 100), text=f"Frame {frame_idx+1}/{total_frames} · {current_fps:.1f} fps")
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
                    st.caption(f"{total} tracked object-instances across all frames · conf ≥ {conf:.0%}")
                else:
                    st.info("No detections above the current confidence threshold — try lowering it in the sidebar.")
            else:
                st.markdown(
                    f"""
                    <div style="text-align:center;padding:3.5rem 2rem;color:{T['faint']};">
                        <div style="font-size:3rem;margin-bottom:0.8rem;">□</div>
                        <div style="font-size:1.05rem;font-weight:500;color:{T['muted']};">Upload an aerial video to get started</div>
                        <div style="font-size:0.85rem;margin-top:0.4rem;">MP4, AVI, MOV, or MKV · processed frame-by-frame with tracking</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

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
                st.info("Run a detection in the Image Demo or Video Demo tab first.")

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
            f'<div class="footer">Runs server-side via Ultralytics YOLO + ByteTrack · '
            f'<span>Semester Project</span> · CSC354 Machine Learning</div>',
            unsafe_allow_html=True,
        )
