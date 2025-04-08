import streamlit as st
import cv2
import torch
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from models.common import DetectMultiBackend, AutoShape
from io import BytesIO
from PIL import Image
import tempfile
import os

# Khởi tạo DeepSort và YOLOv9
@st.cache_resource
def initialize_models():
    try:
        tracker = DeepSort(max_age=30)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = DetectMultiBackend(weights="weights/yolov9-c.pt", device=device, fuse=True)
        model = AutoShape(model)
        return tracker, model, device
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return None, None, None

tracker, model, device = initialize_models()

if tracker is None or model is None:
    st.stop()

# Load class names
try:
    with open("data_ext/classes.names") as f:
        class_names = f.read().strip().split('\n')
except FileNotFoundError:
    st.error("File 'data_ext/classes.names' not found!")
    st.stop()

colors = np.random.randint(0, 255, size=(len(class_names), 3))

def process_frame(frame, conf_threshold, tracking_class):
    try:
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            results = model(frame)
        
        detect = []
        for detect_object in results.pred[0]:
            label, confidence, bbox = detect_object[5], detect_object[4], detect_object[:4]
            x1, y1, x2, y2 = map(int, bbox)
            class_id = int(label)

            if tracking_class is None:
                if confidence < conf_threshold:
                    continue
            else:
                if class_id != tracking_class or confidence < conf_threshold:
                    continue

            detect.append([[x1, y1, x2-x1, y2-y1], confidence, class_id])

        tracks = tracker.update_tracks(detect, frame=frame)
        
        for track in tracks:
            if track.is_confirmed():
                track_id = track.track_id
                ltrb = track.to_ltrb()
                class_id = track.get_det_class()
                x1, y1, x2, y2 = map(int, ltrb)
                color = colors[class_id]
                B, G, R = map(int, color)

                label = f"{class_names[class_id]}-{track_id}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
                cv2.rectangle(frame, (x1-1, y1-20), (x1 + len(label)*12, y1), (B, G, R), -1)
                cv2.putText(frame, label, (x1+5, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    except Exception as e:
        st.error(f"Error processing frame: {str(e)}")
        return frame

# Streamlit interface
st.title("Object Tracking with YOLOv9 and DeepSort")

# Sidebar settings
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
tracking_class = st.sidebar.selectbox("Tracking Class", ["All"] + class_names)
tracking_class = None if tracking_class == "All" else class_names.index(tracking_class)

# Input options
input_option = st.radio("Select Input Type", ("Upload Image", "Upload Video", "Camera"))

# Khởi tạo session state để theo dõi trạng thái camera
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "cap" not in st.session_state:
    st.session_state.cap = None

# Hàm để dừng camera
def stop_camera():
    if st.session_state.camera_running and st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.camera_running = False
        st.session_state.cap = None

if input_option == "Upload Image":
    stop_camera()  # Dừng camera nếu đang chạy
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            processed_frame = process_frame(frame, conf_threshold, tracking_class)
            st.image(processed_frame, channels="BGR")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

elif input_option == "Upload Video":
    stop_camera()  # Dừng camera nếu đang chạy
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])
    if uploaded_file is not None:
        try:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()
            
            cap = cv2.VideoCapture(tfile.name)
            if not cap.isOpened():
                st.error("Error: Could not open video file.")
                os.unlink(tfile.name)
            else:
                stframe = st.empty()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    processed_frame = process_frame(frame, conf_threshold, tracking_class)
                    stframe.image(processed_frame, channels="BGR")
                cap.release()
                os.unlink(tfile.name)
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            if 'tfile' in locals():
                os.unlink(tfile.name)

elif input_option == "Camera":
    # Nếu camera chưa chạy, khởi tạo nó
    if not st.session_state.camera_running:
        stop_camera()  # Đảm bảo camera cũ được giải phóng
        st.session_state.cap = cv2.VideoCapture(0)
        if not st.session_state.cap.isOpened():
            st.error("Error: Could not open camera.")
            st.session_state.cap = None
        else:
            st.session_state.camera_running = True

    if st.session_state.camera_running and st.session_state.cap is not None:
        run = st.checkbox('Run Camera', value=True)
        FRAME_WINDOW = st.image([])
        
        if run:
            while st.session_state.camera_running and input_option == "Camera":
                ret, frame = st.session_state.cap.read()
                if not ret:
                    st.warning("Warning: Could not read frame from camera.")
                    break
                processed_frame = process_frame(frame, conf_threshold, tracking_class)
                FRAME_WINDOW.image(processed_frame, channels="BGR")
        else:
            stop_camera()  # Dừng camera nếu checkbox bị bỏ chọn
    else:
        st.write("Camera is not available.")

st.write(f"Using device: {device}")