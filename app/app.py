"""YOLO Object Detection Simulator  -  run with:  streamlit run app/app.py"""
import streamlit as st

from inference import load_model, run_detection
from utils import MODEL_PATH, detections_to_df, load_image

st.set_page_config(page_title="YOLO Object Detection Simulator", layout="wide")
st.title("YOLO Object Detection Simulator")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model(MODEL_PATH)  # loaded once, reused across reruns


try:
    model = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

st.caption(f"Model: `{MODEL_PATH.name}` | Classes: {', '.join(model.names.values())}")

conf = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)


def show_results(image, key):
    with st.spinner("Detecting..."):
        annotated, detections = run_detection(model, image, conf=conf)
    left, right = st.columns(2)
    left.subheader("Original Image")
    left.image(image)
    right.subheader("Detected Image")
    right.image(annotated)
    st.metric("Number of detections", len(detections))
    if detections:
        st.subheader("Detected Objects")
        st.dataframe(detections_to_df(detections), hide_index=True)
    else:
        st.info("No objects detected at this threshold. Try lowering it.")


tab_upload, tab_cam = st.tabs(["Upload image", "Webcam snapshot (optional)"])

with tab_upload:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        try:
            image = load_image(uploaded)
        except Exception as e:
            st.error(f"Could not read this image: {e}")
        else:
            if st.button("Run Detection", type="primary"):
                show_results(image, "upload")

with tab_cam:
    # Single-frame capture through the browser: simple and does not affect the upload tab.
    snap = st.camera_input("Take a picture")
    if snap is not None:
        show_results(load_image(snap), "cam")
