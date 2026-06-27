import streamlit as st
from PIL import Image
import tempfile
import os
from inference import detect_image

st.set_page_config(
    page_title="Aerial Scene Understanding",
    page_icon="🛸",
    layout="wide"
)

st.title("🛸 Aerial Scene Understanding Using YOLO")
st.write(
    "Upload an aerial image and detect objects such as "
    "cars, pedestrians, trucks, buses, bicycles, and motorcycles."
)

conf = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)

uploaded_file = st.file_uploader("Upload Aerial Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(input_image, use_container_width=True)

    # Save upload to a temp file for inference
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        input_image.save(temp.name)
        temp_path = temp.name

    output_path = "outputs/result.jpg"
    result_path, counts = detect_image(temp_path, output_path, conf)

    with col2:
        st.subheader("Detection Output")
        if result_path and os.path.exists(result_path):
            st.image(result_path, use_container_width=True)
        else:
            st.warning("No detections found above the confidence threshold.")

    st.subheader("Detected Object Counts")
    if counts:
        st.write(counts)
    else:
        st.info("No objects detected.")

    # Clean up temp file
    os.remove(temp_path)
