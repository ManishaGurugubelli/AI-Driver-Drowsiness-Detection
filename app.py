import streamlit as st
import cv2
import mediapipe as mp
import av
import pygame
import numpy as np
import time

from scipy.spatial import distance
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Driver Drowsiness Detection",
    layout="wide"
)

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🚗 AI Driver Drowsiness Detection System")

st.markdown(
    """
    <h4 style='color:gray;'>
    Real-Time Driver Monitoring using
    MediaPipe + EAR + MAR
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("⚙ System Controls")

st.sidebar.markdown("---")

ear_threshold = st.sidebar.slider(
    "EAR Threshold",
    0.15,
    0.35,
    0.20,
    0.01
)

yawn_threshold = st.sidebar.slider(
    "Yawn Threshold",
    0.40,
    1.50,
    0.90,
    0.01
)

alarm_duration = st.sidebar.slider(
    "Drowsiness Time (sec)",
    1,
    5,
    3
)

st.sidebar.info(
    "Adjust EAR threshold based on eye shape and lighting."
)

st.sidebar.markdown("---")

st.sidebar.success("✅ Monitoring Active")

# ---------------------------------------------------
# EAR FUNCTION
# ---------------------------------------------------

def calculate_EAR(eye_points):

    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])

    ear = (A + B) / (2.0 * C)

    return ear

# ---------------------------------------------------
# MAR FUNCTION
# ---------------------------------------------------

MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

def calculate_MAR(landmarks, w, h):

    top = landmarks[MOUTH_TOP]
    bottom = landmarks[MOUTH_BOTTOM]
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]

    top_point = (
        int(top.x * w),
        int(top.y * h)
    )

    bottom_point = (
        int(bottom.x * w),
        int(bottom.y * h)
    )

    left_point = (
        int(left.x * w),
        int(left.y * h)
    )

    right_point = (
        int(right.x * w),
        int(right.y * h)
    )

    vertical = distance.euclidean(
        top_point,
        bottom_point
    )

    horizontal = distance.euclidean(
        left_point,
        right_point
    )

    mar = vertical / horizontal

    return mar

# ---------------------------------------------------
# LANDMARKS
# ---------------------------------------------------

LEFT_EYE = [33, 160, 158, 133, 153, 144]

RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ---------------------------------------------------
# MEDIAPIPE SETUP
# ---------------------------------------------------

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------------------------------------------
# ALARM SETUP
# ---------------------------------------------------

try:

    pygame.mixer.init()

    alarm_sound = pygame.mixer.Sound(
        "alarm.wav"
    )

except:

    alarm_sound = None

# ---------------------------------------------------
# VIDEO PROCESSOR
# ---------------------------------------------------

class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.eye_closed_start = None

        self.alarm_playing = False

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.resize(img, (640, 480))

        rgb_frame = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb_frame)

        status = "ALERT"

        closed_duration = 0

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                h, w, c = img.shape

                left_eye_points = []
                right_eye_points = []

                # ---------------------------------------------------
                # LEFT EYE
                # ---------------------------------------------------

                for idx in LEFT_EYE:

                    landmark = (
                        face_landmarks.landmark[idx]
                    )

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    left_eye_points.append((x, y))

                    cv2.circle(
                        img,
                        (x, y),
                        2,
                        (0,255,0),
                        -1
                    )

                # ---------------------------------------------------
                # RIGHT EYE
                # ---------------------------------------------------

                for idx in RIGHT_EYE:

                    landmark = (
                        face_landmarks.landmark[idx]
                    )

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    right_eye_points.append((x, y))

                    cv2.circle(
                        img,
                        (x, y),
                        2,
                        (0,0,255),
                        -1
                    )

                # ---------------------------------------------------
                # MOUTH LANDMARKS
                # ---------------------------------------------------

                mouth_indices = [
                    MOUTH_TOP,
                    MOUTH_BOTTOM,
                    MOUTH_LEFT,
                    MOUTH_RIGHT
                ]

                for idx in mouth_indices:

                    landmark = (
                        face_landmarks.landmark[idx]
                    )

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    cv2.circle(
                        img,
                        (x, y),
                        3,
                        (255,255,0),
                        -1
                    )

                # ---------------------------------------------------
                # EAR
                # ---------------------------------------------------

                left_EAR = calculate_EAR(
                    left_eye_points
                )

                right_EAR = calculate_EAR(
                    right_eye_points
                )

                avg_EAR = (
                    left_EAR + right_EAR
                ) / 2

                avg_EAR = round(avg_EAR, 2)

                # Prevent false detections

                if avg_EAR < 0.10:

                    avg_EAR = 0.25

                # ---------------------------------------------------
                # MAR
                # ---------------------------------------------------

                mar = calculate_MAR(
                    face_landmarks.landmark,
                    w,
                    h
                )

                mar = round(mar, 2)

                # ---------------------------------------------------
                # DROWSINESS LOGIC
                # ---------------------------------------------------

                eyes_closed = (
                    avg_EAR < ear_threshold
                )

                if eyes_closed:

                    if (
                        self.eye_closed_start
                        is None
                    ):

                        self.eye_closed_start = (
                            time.time()
                        )

                    closed_duration = (
                        time.time()
                        - self.eye_closed_start
                    )

                else:

                    self.eye_closed_start = None

                    closed_duration = 0

                # ---------------------------------------------------
                # DROWSINESS ALERT
                # ---------------------------------------------------

                if (
                    closed_duration >= alarm_duration
                    and mar < 0.75
                ):

                    status = "DROWSY"

                    if alarm_sound:

                        if not self.alarm_playing:

                            alarm_sound.play()

                            self.alarm_playing = True

                else:

                    status = "ALERT"

                    if alarm_sound:

                        alarm_sound.stop()

                    self.alarm_playing = False

                # ---------------------------------------------------
                # YAWN DETECTION
                # ---------------------------------------------------

                if mar > yawn_threshold:

                    cv2.putText(
                        img,
                        "YAWNING",
                        (20,240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255,0,255),
                        3
                    )

                # ---------------------------------------------------
                # DISPLAY TEXT
                # ---------------------------------------------------

                cv2.putText(
                    img,
                    f"EAR: {avg_EAR:.2f}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,255,255),
                    2
                )

                cv2.putText(
                    img,
                    f"MAR: {mar:.2f}",
                    (20,200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,255),
                    2
                )

                if status == "DROWSY":

                    color = (0,0,255)

                    cv2.putText(
                        img,
                        "WAKE UP!",
                        (120,300),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0,0,255),
                        5
                    )

                else:

                    color = (0,255,0)

                cv2.putText(
                    img,
                    status,
                    (20,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    3
                )

                cv2.putText(
                    img,
                    f"Closed Time: {closed_duration:.1f}s",
                    (20,160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,0),
                    2
                )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# ---------------------------------------------------
# STREAMLIT WEBCAM
# ---------------------------------------------------

st.subheader("📹 Live Driver Monitoring")

ctx = webrtc_streamer(
    key="driver-monitor",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {
            "width": 640,
            "height": 480
        },
        "audio": False
    },
    async_processing=True
)

st.markdown("---")

st.success(
    "🚀 Driver Drowsiness Detection Active"
)