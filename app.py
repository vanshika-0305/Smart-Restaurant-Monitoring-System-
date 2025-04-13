import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import time
import os
import numpy as np
import torch

# Streamlit page configuration
st.set_page_config(
    page_title="Cafe Table Detection",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'detection_running' not in st.session_state:
    st.session_state.detection_running = False
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

def apply_custom_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #333;
    }
    .status-box {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .status-clean {
        background-color: rgba sareth(0, 255, 0, 0.2);
        border: 2px solid green;
    }
    .status-dirty {
        background-color: rgba(255, 0, 0, 0.2);
        border: 2px solid red;
    }
    .status-occupied {
        background-color: rgba(255, 165, 0, 0.2);
        border: 2px solid orange;
    }
    .stButton button {
        background-color: #0078d4;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# Main title and description
st.title("🍽️ Smart Cafe Table Detection System")
st.markdown("""
This system uses YOLOv8 to detect and classify tables in a cafe as:
- 🟢 **Clean Unoccupied Tables** - Available for customers
- 🟠 **Occupied Tables** - Currently in use
- 🔴 **Dirty Unoccupied Tables** - Need cleaning
""")

# Sidebar configuration
with st.sidebar:
    st.header("Settings & Stats")
    
    model_path = st.text_input("Model Path", "model/trained_model.pt", 
                              help="Enter the path to your YOLOv8 .pt model file")
    
    confidence = st.slider("Detection Confidence", 0.1, 1.0, 0.5, 0.05)
    
# Performance settings
    st.header("Performance Settings")
    frame_skip = st.slider("Frame Skip", 1, 10, 3, 
                           help="Process 1 frame out of every N frames. Higher values = faster but less smooth.")
    
    st.header("Statistics")
    col1, col2, col3 = st.columns(3)
    
    clean_count = col1.metric("Clean Tables", "0", "Available")
    occupied_count = col2.metric("Occupied", "0", "In Use")
    dirty_count = col3.metric("Dirty Tables", "0", "Need Cleaning")
    
    st.divider()
    st.subheader("Table Status")
    
    clean_tables = st.container()
    occupied_tables = st.container()
    dirty_tables = st.container()

# Tabs for Live Detection and About
tab1, tab2 = st.tabs(["Live Detection", "About"])

with tab1:
    # Detection settings
    col1, col2 = st.columns(2)
    
    with col1:
        source_radio = st.radio("Select Input Source", ["Webcam", "Upload Video"])
    
    with col2:
        draw_mode = st.radio("Visualization Style", ["Circles", "Bounding Boxes"])
    
    # File uploader for video (visible when Upload Video is selected)
    if source_radio == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="video_uploader")
        if uploaded_file is not None:
            st.session_state.uploaded_file = uploaded_file
        elif st.session_state.uploaded_file is not None and uploaded_file is None:
            st.session_state.uploaded_file = None
    
    # Start/Stop button
    if not st.session_state.detection_running:
        start_button = st.button("Start Detection", use_container_width=True)
        if start_button:
            if source_radio == "Upload Video" and st.session_state.uploaded_file is None:
                status_text.error("Please upload a video file first.")
            else:
                st.session_state.detection_running = True
    else:
        stop_button = st.button("Stop Detection", use_container_width=True)
        if stop_button:
            st.session_state.detection_running = False
    
    # Video display placeholder
    video_placeholder = st.empty()
    
    # Status text
    status_text = st.empty()

    def load_model(model_path):
        try:
            if not os.path.exists(model_path):
                status_text.error(f"Model not found at {model_path}. Please check the path.")
                return None
            model = YOLO(model_path)
            return model
        except Exception as e:
            status_text.error(f"Error loading model: {str(e)}")
            return None
    
    def draw_detection(frame, results, draw_mode="Circles"):
        annotated_frame = frame.copy()
        counts = {"unoccupied-clean": 0, "unoccupied-dirty": 0, "occupied": 0}
        colors = {
            "unoccupied-clean": (0, 255, 0),
            "unoccupied-dirty": (0, 0, 255),
            "occupied": (0, 165, 255)
        }
        
        class_names = ["occupied", "unoccupied-clean", "unoccupied-dirty"]
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if conf < confidence:
                    continue
                
                class_name = class_names[cls]
                color = colors[class_name]
                counts[class_name] += 1
                
                if draw_mode == "Circles":
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    radius = int(max(x2 - x1, y2 - y1) // 2)
                    cv2.circle(annotated_frame, (center_x, center_y), radius, color, 3)
                    cv2.putText(annotated_frame, f"{class_name} {conf:.2f}", 
                               (center_x - radius//2, center_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                else:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{class_name} {conf:.2f}", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        clean_count.metric("Clean Tables", str(counts["unoccupied-clean"]), "Available")
        occupied_count.metric("Occupied", str(counts["occupied"]), "In Use")
        dirty_count.metric("Dirty Tables", str(counts["unoccupied-dirty"]), "Need Cleaning")
        
        return annotated_frame, counts
    
    def process_webcam():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            status_text.error("Error: Could not open webcam.")
            return
        
        model = load_model(model_path)
        if model is None:
            return
        
        status_text.info("Webcam detection started.")

        try:
            while st.session_state.detection_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only process every frame_skip frames
                if cap.get(cv2.CAP_PROP_POS_FRAMES) % frame_skip != 0:
                    continue
                    
                results = model(frame)
                annotated_frame, counts = draw_detection(frame, results, draw_mode)
                    
                update_status_containers(counts)
                video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                
                
                
        finally:
            cap.release()
            status_text.info("Webcam detection stopped.")
            st.session_state.detection_running = False
    
    def process_uploaded_video(video_file):
        try:
            status_text.info(f"Processing video: {video_file.name}")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(video_file.read())
                tfile_name = tfile.name
            
            if not os.path.exists(tfile_name) or os.path.getsize(tfile_name) == 0:
                status_text.error("Error: Failed to save video file.")
                return
            
            model = load_model(model_path)
            if model is None:
                return
            
            cap = cv2.VideoCapture(tfile_name)
            if not cap.isOpened():
                status_text.error("Error: Could not open video file. Ensure the format is supported (MP4, AVI, MOV).")
                return
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            status_text.info(f"Video details: {total_frames} frames at {fps:.2f} FPS")
            
            # Create a progress bar
            progress_bar = st.progress(0)
            frame_counter = 0
            
            try:
                while cap.isOpened() and st.session_state.detection_running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_counter += 1
                    # Update progress bar
                    progress_bar.progress(min(frame_counter / total_frames, 1.0))
                    
                    # Only process every frame_skip frames
                    if frame_counter % frame_skip != 0:
                        continue
                    
                    results = model(frame)
                    annotated_frame, counts = draw_detection(frame, results, draw_mode)
                    
                    update_status_containers(counts)
                    video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                    
                   
            
            finally:
                cap.release()
                try:
                    os.unlink(tfile_name)
                    status_text.info("Temporary file deleted.")
                except:
                    status_text.warning("Could not delete temporary file.")
                status_text.info("Video processing completed.")
                st.session_state.detection_running = False
                
        except Exception as e:
            status_text.error(f"Error processing video: {str(e)}")
            st.session_state.detection_running = False
    
    def update_status_containers(counts):
        with clean_tables:
            st.empty()
            if counts["unoccupied-clean"] > 0:
                st.markdown(f"""
                <div class="status-box status-clean">
                    🟢 <b>{counts["unoccupied-clean"]} Clean Tables</b> available for customers
                </div>
                """, unsafe_allow_html=True)
                
        with occupied_tables:
            st.empty()
            if counts["occupied"] > 0:
                st.markdown(f"""
                <div class="status-box status-occupied">
                    🟠 <b>{counts["occupied"]} Tables Occupied</b> by customers
                </div>
                """, unsafe_allow_html=True)
                
        with dirty_tables:
            st.empty()
            if counts["unoccupied-dirty"] > 0:
                st.markdown(f"""
                <div class="status-box status-dirty">
                    🔴 <b>{counts["unoccupied-dirty"]} Dirty Tables</b> need cleaning
                </div>
                """, unsafe_allow_html=True)
    
    # Run detection if started
    if st.session_state.detection_running:
        if source_radio == "Webcam":
            process_webcam()
        elif source_radio == "Upload Video" and st.session_state.uploaded_file is not None:
            process_uploaded_video(st.session_state.uploaded_file)

with tab2:
    st.header("About This System")
    st.markdown("""
    Smart Cafe Table Detection System for Cafe 96
    
    This application uses computer vision to help cafe staff monitor table status in real-time.
    """)


# Footer
st.markdown("""
---
*Developed for Cafe Table Management System | © 2025*
""", unsafe_allow_html=True)
