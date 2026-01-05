import streamlit as st
# Perbaikan: Tambahkan WebRtcMode di import
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase, WebRtcMode
import cv2
import numpy as np
import pickle
import os
import av
from datetime import datetime
from keras_facenet import FaceNet

# --- SETUP HALAMAN ---
st.set_page_config(page_title="Absensi FaceNet", layout="centered")

# --- LOAD MODELS (DENGAN CACHE) ---
@st.cache_resource
def load_models():
    # 1. Load FaceNet
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    embedder = FaceNet()
    
    # 2. Load Haar Cascade
    # Menggunakan path bawaan cv2 agar aman di Cloud
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # 3. Load SVM & Encoder
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'best_face_model.pkl')
    encoder_path = os.path.join(base_dir, 'models', 'label_encoder.pkl')
    
    with open(model_path, 'rb') as f:
        svm_model = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
        
    return embedder, face_cascade, svm_model, encoder

try:
    embedder, face_cascade, svm_model, encoder = load_models()
    st.success("Sistem Siap! Silakan izinkan akses kamera.")
except Exception as e:
    st.error(f"Error loading models: {e}")

# --- LOGIKA PEMROSESAN VIDEO ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.status = "idle" 
        self.name = None
        self.confidence_display = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Mirror image
        img = cv2.flip(img, 1)
        h_frame, w_frame, _ = img.shape
        
        # Deteksi Wajah
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        overlay_text = "Mencari..."
        color = (255, 255, 255)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            
            # --- TAHAP 1: IDENTIFIKASI ---
            if self.status == "idle":
                face_img = img[y:y+h, x:x+w]
                if face_img.size > 0:
                    try:
                        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        face_resized = cv2.resize(face_rgb, (160, 160))
                        face_array = np.expand_dims(face_resized, axis=0)
                        
                        embedding = embedder.embeddings(face_array)
                        probs = svm_model.predict_proba(embedding)[0]
                        max_prob = np.max(probs)
                        
                        if max_prob > 0.60:
                            self.name = encoder.inverse_transform([np.argmax(probs)])[0]
                            self.confidence_display = max_prob
                            self.status = "challenge_left"
                        else:
                            overlay_text = "Wajah Tidak Dikenali"
                    except:
                        pass
            
            # --- TAHAP 2: TANTANGAN GESER ---
            center_x = x + (w // 2)
            
            if self.status == "challenge_left":
                overlay_text = f"Halo {self.name}! Geser KIRI >>"
                color = (0, 255, 255)
                # Kanan layar = Kiri user (mirror)
                if center_x > w_frame * 0.7: 
                    self.status = "challenge_right"
            
            elif self.status == "challenge_right":
                overlay_text = "Bagus! Geser KANAN <<"
                color = (0, 255, 255)
                # Kiri layar = Kanan user
                if center_x < w_frame * 0.3:
                    self.status = "verified"
            
            elif self.status == "verified":
                overlay_text = f"VERIFIED: {self.name}"
                color = (0, 255, 0)
            
            # Visualisasi
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, overlay_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        else:
            if self.status != "verified":
                self.status = "idle"
                self.name = None
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- TAMPILAN UTAMA ---
st.title("Sistem Absensi Online")

# Konfigurasi WebRTC (PERBAIKAN DI SINI)
# Konfigurasi WebRTC
ctx = webrtc_streamer(
    key="absensi-facenet",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
        ]
    }),
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
# --- PANEL PENCATATAN ---
if ctx.video_processor:
    status = ctx.video_processor.status
    name = ctx.video_processor.name
    
    if status == "verified" and name:
        st.success(f"Identitas Terverifikasi: **{name}**")
        
        with st.form("absen_form"):
            st.write("Catat absensi sekarang?")
            if st.form_submit_button("YA, Catat Absensi"):
                now = datetime.now()
                dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
                
                # Simpan CSV
                file_csv = 'absensi_cloud.csv'
                if not os.path.exists(file_csv):
                    with open(file_csv, 'w') as f:
                        f.write("Nama,Waktu\n")
                
                with open(file_csv, 'a') as f:
                    f.write(f"{name},{dt_string}\n")
                
                st.balloons()
                st.success(f"Absensi {name} sukses!")
                
                # Reset
                ctx.video_processor.status = "idle"

