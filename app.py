import os
import tempfile
import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
from inference import detect_image, detect_video

st.set_page_config(
    page_title="Autonom · Streamlit Demo",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1530 50%, #0a1628 100%); min-height: 100vh; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f1829 0%, #111e35 100%); border-right: 1px solid rgba(99,179,237,0.15); }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.hero-header { background: linear-gradient(135deg, rgba(15,24,56,0.9) 0%, rgba(20,35,80,0.9) 100%); border: 1px solid rgba(99,179,237,0.25); border-radius: 20px; padding: 2.5rem 3rem; margin-bottom: 2rem; backdrop-filter: blur(20px); position: relative; overflow: hidden; }
.hero-title { font-size: 2.4rem; font-weight: 800; background: linear-gradient(135deg, #63b3ed 0%, #90cdf4 50%, #bee3f8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 0.5rem 0; }
.hero-sub { color: #90cdf4; font-size: 1rem; opacity: 0.85; margin: 0; }
.hero-badge { display: inline-block; background: rgba(99,179,237,0.15); border: 1px solid rgba(99,179,237,0.35); color: #90cdf4; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; padding: 0.3rem 0.8rem; border-radius: 50px; margin-bottom: 1rem; }
[data-testid="stFileUploader"] { background: rgba(15,24,56,0.6) !important; border: 2px dashed rgba(99,179,237,0.35) !important; border-radius: 16px !important; padding: 1.5rem !important; }
.image-panel { background: rgba(15,24,56,0.7); border: 1px solid rgba(99,179,237,0.15); border-radius: 16px; padding: 1.2rem; }
.panel-label { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #63b3ed; margin-bottom: 0.8rem; }
.metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 1rem; margin-top: 1rem; }
.metric-card { background: linear-gradient(135deg, rgba(15,24,56,0.9) 0%, rgba(20,35,80,0.9) 100%); border: 1px solid rgba(99,179,237,0.2); border-radius: 14px; padding: 1.1rem 1rem; text-align: center; }
.metric-count { font-size: 2rem; font-weight: 800; color: #90cdf4; line-height: 1; }
.metric-name { font-size: 0.75rem; color: #a0aec0; margin-top: 0.25rem; text-transform: capitalize; }
.total-banner { background: linear-gradient(135deg, rgba(99,179,237,0.12) 0%, rgba(144,205,244,0.08) 100%); border: 1px solid rgba(99,179,237,0.3); border-radius: 14px; padding: 1.2rem 1.8rem; display: flex; align-items: center; justify-content: space-between; margin-top: 1.5rem; }
.total-label { color: #90cdf4; font-size: 0.85rem; }
.total-value { font-size: 1.8rem; font-weight: 800; color: #bee3f8; }
.section-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(99,179,237,0.3), transparent); margin: 2rem 0; }
.footer { text-align: center; color: #4a5568; font-size: 0.78rem; padding: 2rem 0 1rem; border-top: 1px solid rgba(99,179,237,0.08); margin-top: 3rem; }
.footer span { color: #63b3ed; }
.sidebar-section-title { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #63b3ed; margin: 1.5rem 0 0.5rem; }
</style>
""", unsafe_allow_html=True)

CLASS_ICONS = {
    "car": "🚗", "truck": "🚛", "bus": "🚌", "pedestrian": "🚶",
    "person": "🚶", "bicycle": "🚲", "motorcycle": "🏍️", "van": "🚐", "drone": "🛸"
}
def get_icon(class_name: str) -> str:
    return CLASS_ICONS.get(class_name.lower(), "📦")

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.2rem 0 0.5rem;">
        <div style="font-size:2.5rem">🛸</div>
        <div style="font-size:1.1rem;font-weight:700;color:#90cdf4;margin-top:0.3rem;">Aerial Detection</div>
        <div style="font-size:0.72rem;color:#4a5568;margin-top:0.2rem;">Aerial Scene Understanding</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">⚙️ Detection Settings</div>', unsafe_allow_html=True)
    conf = st.slider("Confidence Threshold", 0.10, 0.90, 0.25, 0.05)

    st.markdown('<div class="sidebar-section-title">📂 Sample Media</div>', unsafe_allow_html=True)
    sample_dir = "sample_images and sample_videos"
    sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".mp4"))] if os.path.isdir(sample_dir) else []
    selected_sample = st.selectbox("Try a sample image or video", ["— none —"] + sample_files) if sample_files else "— none —"

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="margin:0 0 1rem 0;font-size:0.9rem;color:#90cdf4;">'
    '<a href="https://uzaairmalik.github.io/Autonom/" target="_blank" rel="noopener" style="color:#90cdf4;text-decoration:none;">← Back to overview</a>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("""
<div class="hero-header">
    <div class="hero-badge">🛸 FYP · Computer Vision · YOLOv11</div>
    <h1 class="hero-title">Autonom Streamlit Demo</h1>
    <p class="hero-sub">Upload a drone image or video to detect and count objects with a custom-trained YOLOv11 model. Video inference is handled here; the GitHub Pages site remains the landing page.</p>
</div>
""", unsafe_allow_html=True)

tab_demo, tab_analytics, tab_eval = st.tabs(["🛸 Live Demo", "📊 Analytics", "📈 Model Evaluation"])

with tab_demo:
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
    uploaded_file = st.file_uploader("📁 Drop an aerial image or video here", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

    input_image = None
    source_name = ""
    is_video = False
    video_bytes = None
    counts = {}

    if uploaded_file is not None:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        source_name = uploaded_file.name
        if ext in VIDEO_EXTS:
            is_video = True
            video_bytes = uploaded_file.read()
        else:
            input_image = Image.open(uploaded_file)
    elif selected_sample != "— none —":
        sample_path = os.path.join(sample_dir, selected_sample)
        ext = os.path.splitext(selected_sample)[1].lower()
        source_name = selected_sample
        if ext in VIDEO_EXTS:
            is_video = True
            with open(sample_path, "rb") as f:
                video_bytes = f.read()
        else:
            input_image = Image.open(sample_path)

    if input_image is not None:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown(f'<div class="image-panel"><div class="panel-label">📷 Input Image · {source_name}</div></div>', unsafe_allow_html=True)
            st.image(input_image, use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            input_image.save(temp.name)
            temp_path = temp.name

        output_path = "outputs/result.jpg"
        with st.spinner("🔍 Running YOLO detection…"):
            result_path, counts, inf_time = detect_image(temp_path, output_path, conf)
            st.session_state.counts = counts
            st.session_state.inf_time = inf_time
            st.session_state.is_video = False

        with col2:
            st.markdown('<div class="image-panel"><div class="panel-label">🎯 Detection Output</div></div>', unsafe_allow_html=True)
            if result_path and os.path.exists(result_path):
                st.image(result_path, use_container_width=True)
                with open(result_path, "rb") as f:
                    st.download_button("⬇️ Download Result", f, "aerial_detection_result.jpg", "image/jpeg")
            else:
                st.warning("No objects detected above threshold.")
        os.remove(temp_path)

    elif is_video and video_bytes is not None:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown(f'<div class="image-panel"><div class="panel-label">🎬 Input Video · {source_name}</div></div>', unsafe_allow_html=True)
            st.video(video_bytes)

        ext_suffix = os.path.splitext(source_name)[1].lower() or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext_suffix) as tmp_vid:
            tmp_vid.write(video_bytes)
            tmp_vid_path = tmp_vid.name

        output_vid_path = "outputs/result.mp4"
        progress_bar = st.progress(0, text="Starting…")
        frame_history = []

        try:
            gen = detect_video(tmp_vid_path, output_vid_path, conf)
            while True:
                frame_idx, total_frames, frame_counts, current_fps = next(gen)
                counts = frame_counts
                pct = int((frame_idx + 1) / max(total_frames, 1) * 100)
                progress_bar.progress(pct, text=f"Frame {frame_idx+1}/{total_frames} | FPS: {current_fps:.1f}")
        except StopIteration as e:
            if e.value:
                output_vid_path, counts, frame_history = e.value
                st.session_state.counts = counts
                st.session_state.frame_history = frame_history
                st.session_state.is_video = True
        except RuntimeError as e:
            st.error(f"Video processing failed: {e}")
            counts = {}
        finally:
            progress_bar.progress(100, text="Done")
            if os.path.exists(tmp_vid_path):
                os.remove(tmp_vid_path)

        with col2:
            st.markdown('<div class="image-panel"><div class="panel-label">🎯 Annotated Output</div></div>', unsafe_allow_html=True)
            if os.path.exists(output_vid_path):
                with open(output_vid_path, "rb") as f:
                    out_bytes = f.read()
                st.video(out_bytes)
                st.download_button("⬇️ Download Annotated Video", out_bytes, "aerial_detection_result.mp4", "video/mp4")
            else:
                st.warning("Could not generate annotated video.")

    if input_image is not None or (is_video and video_bytes is not None):
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 Detection Results")

        if counts:
            total = sum(counts.values())
            label = "across all frames" if is_video else f"in {st.session_state.get('inf_time', 0):.2f}s"
            cards_html = '<div class="metrics-grid">'
            for cls_name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                cards_html += f'<div class="metric-card"><div>{get_icon(cls_name)}</div><div class="metric-count">{cnt}</div><div class="metric-name">{cls_name}</div></div>'
            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)
            st.markdown(f'<div class="total-banner"><div><div class="total-label">Total Objects Detected</div><div style="color:#4a5568;font-size:0.72rem;">conf ≥ {conf:.0%} · {label}</div></div><div class="total-value">{total}</div></div>', unsafe_allow_html=True)
        else:
            st.info("No detections returned. Try lowering confidence.")
    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#4a5568;">
            <div style="font-size:4rem;margin-bottom:1rem;">🛸</div>
            <div style="font-size:1.1rem;font-weight:500;color:#718096;">Upload an aerial image or video to get started</div>
            <div style="font-size:0.85rem;margin-top:0.5rem;">or pick a sample from the sidebar</div>
        </div>
        """, unsafe_allow_html=True)

with tab_analytics:
    st.markdown("## 📊 Detection Analytics")
    if 'counts' in st.session_state and st.session_state.counts:
        if st.session_state.get('is_video', False) and st.session_state.get('frame_history'):
            st.markdown("### Object Detection Timeline (Video)")
            df = pd.DataFrame(st.session_state.frame_history).fillna(0)
            df['Frame'] = df.index
            df_melt = df.melt(id_vars=['Frame'], var_name='Class', value_name='Count')
            fig = px.area(df_melt, x='Frame', y='Count', color='Class', title="Cumulative Object Counts over Time")
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Class Distribution")
        df_dist = pd.DataFrame(list(st.session_state.counts.items()), columns=['Class', 'Count']).sort_values('Count', ascending=False)
        fig_bar = px.bar(df_dist, x='Class', y='Count', color='Class', title="Total Object Distribution")
        fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Run detection in Live Demo first.")

with tab_eval:
    st.markdown("## 📈 Model Evaluation Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("mAP@50", "0.925", "Target > 0.90")
    c2.metric("Precision", "0.89", "Target > 0.85")
    c3.metric("Recall", "0.87", "Target > 0.80")

    st.markdown("---")
    possible_paths = ["confusion_matrix.png", "models/confusion_matrix.png", "1783415382.png"]
    cm_path = next((p for p in possible_paths if os.path.exists(p)), None)
    if cm_path:
        st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
    else:
        st.info("Place confusion matrix file at confusion_matrix.png to display.")

st.markdown("""
<div class="footer">
    Built with <span>❤️</span> using <span>YOLOv11</span> & <span>Streamlit</span> · Semester Project · <span>Aerial Detection</span>
</div>
""", unsafe_allow_html=True)