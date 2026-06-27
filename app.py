import streamlit as st
from PIL import Image
import tempfile
import os
from inference import detect_image

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Autonom · Aerial Scene Understanding",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- Google Font ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---------- Global ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1530 50%, #0a1628 100%);
    min-height: 100vh;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1829 0%, #111e35 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* ---------- Hero Header ---------- */
.hero-header {
    background: linear-gradient(135deg, rgba(15,24,56,0.9) 0%, rgba(20,35,80,0.9) 100%);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}

.hero-header::before {
    content: "";
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    pointer-events: none;
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #63b3ed 0%, #90cdf4 50%, #bee3f8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}

.hero-sub {
    color: #90cdf4;
    font-size: 1rem;
    font-weight: 400;
    opacity: 0.85;
    margin: 0;
}

.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.15);
    border: 1px solid rgba(99,179,237,0.35);
    color: #90cdf4;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.3rem 0.8rem;
    border-radius: 50px;
    margin-bottom: 1rem;
}

/* ---------- Upload zone ---------- */
[data-testid="stFileUploader"] {
    background: rgba(15,24,56,0.6) !important;
    border: 2px dashed rgba(99,179,237,0.35) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    transition: border-color 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,179,237,0.65) !important;
}

/* ---------- Image panels ---------- */
.image-panel {
    background: rgba(15,24,56,0.7);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 16px;
    padding: 1.2rem;
    backdrop-filter: blur(10px);
}

.panel-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

/* ---------- Metric cards ---------- */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
}

.metric-card {
    background: linear-gradient(135deg, rgba(15,24,56,0.9) 0%, rgba(20,35,80,0.9) 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 14px;
    padding: 1.1rem 1rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    cursor: default;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(99,179,237,0.15);
    border-color: rgba(99,179,237,0.45);
}

.metric-icon { font-size: 1.8rem; margin-bottom: 0.3rem; }
.metric-count {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #63b3ed, #90cdf4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
}
.metric-name {
    font-size: 0.75rem;
    font-weight: 500;
    color: #a0aec0;
    margin-top: 0.25rem;
    text-transform: capitalize;
}

/* ---------- Total banner ---------- */
.total-banner {
    background: linear-gradient(135deg, rgba(99,179,237,0.12) 0%, rgba(144,205,244,0.08) 100%);
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 14px;
    padding: 1.2rem 1.8rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 1.5rem;
}

.total-label { color: #90cdf4; font-size: 0.85rem; font-weight: 500; }
.total-value {
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #63b3ed, #bee3f8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ---------- Slider ---------- */
[data-testid="stSlider"] > div > div > div > div {
    background: #63b3ed !important;
}

/* ---------- Buttons ---------- */
.stDownloadButton > button {
    background: linear-gradient(135deg, #2b6cb0, #3182ce) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

.stDownloadButton > button:hover {
    background: linear-gradient(135deg, #3182ce, #4299e1) !important;
    box-shadow: 0 4px 15px rgba(99,179,237,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ---------- Section divider ---------- */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,179,237,0.3), transparent);
    margin: 2rem 0;
}

/* ---------- Info / warning boxes ---------- */
[data-testid="stAlert"] {
    background: rgba(15,24,56,0.7) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(99,179,237,0.2) !important;
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    color: #4a5568;
    font-size: 0.78rem;
    padding: 2rem 0 1rem;
    border-top: 1px solid rgba(99,179,237,0.08);
    margin-top: 3rem;
}

.footer span { color: #63b3ed; }

/* ---------- Subheader override ---------- */
h2, h3 { color: #e2e8f0 !important; }

/* Sidebar labels */
.sidebar-section-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #63b3ed;
    margin: 1.5rem 0 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Class icons map ─────────────────────────────────────────────────────────────
CLASS_ICONS = {
    "car":          "🚗",
    "truck":        "🚛",
    "bus":          "🚌",
    "pedestrian":   "🚶",
    "person":       "🚶",
    "bicycle":      "🚲",
    "motorcycle":   "🏍️",
    "van":          "🚐",
    "drone":        "🛸",
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
    conf = st.slider("Confidence Threshold", min_value=0.10, max_value=0.90,
                     value=0.25, step=0.05,
                     help="Lower = more detections (may include false positives). Higher = fewer but more certain.")

    st.markdown('<div class="sidebar-section-title">ℹ️ About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.8rem;color:#718096;line-height:1.6;">
    This tool uses a custom-trained <b style="color:#90cdf4;">YOLOv11</b> model to detect
    objects in aerial drone imagery in real time.<br><br>
    <b style="color:#90cdf4;">Detectable classes:</b><br>
    🚗 Car &nbsp;·&nbsp; 🚛 Truck &nbsp;·&nbsp; 🚌 Bus<br>
    🚶 Pedestrian &nbsp;·&nbsp; 🚲 Bicycle &nbsp;·&nbsp; 🏍️ Motorcycle
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">📂 Sample Images and videos</div>', unsafe_allow_html=True)
    sample_dir = "sample_images and sample_videos"
    sample_files = [f for f in os.listdir(sample_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png" , ".mp4"))] if os.path.isdir(sample_dir) else []
    if sample_files:
        selected_sample = st.selectbox("Try a sample image or video", ["— none —"] + sample_files)
    else:
        selected_sample = "— none —"

# ── Hero Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">🛸 FYP · Computer Vision · YOLOv11</div>
    <h1 class="hero-title">Aerial Scene Understanding</h1>
    <p class="hero-sub">
        Upload a drone image to detect and count vehicles, pedestrians, and more —
        powered by a custom-trained YOLOv11 model.
    </p>
</div>
""", unsafe_allow_html=True)

# ── File Uploader ───────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📁  Drop an aerial image or video here, or click to browse",
    type=["jpg", "jpeg", "png" , "mp4"],
    help="Supported formats: JPG, JPEG, PNG, MP4. For videos, only the first frame will be processed."
)

# ── Resolve input (upload or sample) ───────────────────────────────────────────
input_image = None
source_name = ""

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    source_name = uploaded_file.name
elif selected_sample != "— none —":
    sample_path = os.path.join(sample_dir, selected_sample)
    input_image = Image.open(sample_path)
    source_name = selected_sample

# ── Inference & Display ─────────────────────────────────────────────────────────
if input_image is not None:
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(f"""
        <div class="image-panel">
            <div class="panel-label">📷 Input Image <span style="color:#4a5568;font-size:0.7rem;font-weight:400;text-transform:none;">· {source_name}</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.image(input_image, use_container_width=True)

    # Run inference
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        input_image.save(temp.name)
        temp_path = temp.name

    output_path = "outputs/result.jpg"

    with st.spinner("🔍 Running YOLO detection…"):
        result_path, counts = detect_image(temp_path, output_path, conf)

    with col2:
        st.markdown("""
        <div class="image-panel">
            <div class="panel-label">🎯 Detection Output</div>
        </div>
        """, unsafe_allow_html=True)

        if result_path and os.path.exists(result_path):
            st.image(result_path, use_container_width=True)

            # Download button
            with open(result_path, "rb") as f:
                st.download_button(
                    label="⬇️  Download Result",
                    data=f,
                    file_name="aerial_detection_result.jpg",
                    mime="image/jpeg",
                )
        else:
            st.warning("⚠️  No objects detected above the confidence threshold. Try lowering the slider.")

    # ── Detection Results Section ───────────────────────────────────────────────
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Detection Results")

    if counts:
        total = sum(counts.values())

        # Metric cards
        cards_html = '<div class="metrics-grid">'
        for cls_name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            icon = get_icon(cls_name)
            cards_html += f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-count">{cnt}</div>
                <div class="metric-name">{cls_name}</div>
            </div>"""
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        # Total banner
        st.markdown(f"""
        <div class="total-banner">
            <div>
                <div class="total-label">Total Objects Detected</div>
                <div style="color:#4a5568;font-size:0.72rem;margin-top:0.2rem;">{len(counts)} class{'es' if len(counts)>1 else ''} found · conf ≥ {conf:.0%}</div>
            </div>
            <div class="total-value">{total}</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("ℹ️  No detections returned. The model found no objects above the confidence threshold in this image.")

    os.remove(temp_path)

else:
    # Placeholder state
    st.markdown("""
    <div style="text-align:center;padding:4rem 2rem;color:#4a5568;">
        <div style="font-size:4rem;margin-bottom:1rem;">🛸</div>
        <div style="font-size:1.1rem;font-weight:500;color:#718096;">Upload an aerial image to get started</div>
        <div style="font-size:0.85rem;margin-top:0.5rem;">or pick a sample from the sidebar</div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with <span>❤️</span> using <span>YOLOv11</span> & <span>Streamlit</span> ·
    SemesterProject · <span>Aerial Detection</span>
</div>
""", unsafe_allow_html=True)
