import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase
import cv2
import numpy as np
import pickle
import os
import av
from datetime import datetime
from keras_facenet import FaceNet

# --- SETUP HALAMAN ---
st.set_page_config(page_title="Absensi FaceNet", layout="centered")

# --- LOAD MODELS (DENGAN CACHE AGAR CEPAT) ---
@st.cache_resource
def load_models():
    # Load FaceNet
    # Suppress TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    embedder = FaceNet()
    
    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load SVM & Encoder
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
    st.success("Model berhasil dimuat!")
except Exception as e:
    st.error(f"Error loading models: {e}")

# --- LOGIKA PEMROSESAN VIDEO ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.status = "idle" # idle, challenge_left, challenge_right, verified
        self.name = None
        self.frame_count = 0
        self.confidence_display = 0.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Mirror image biar natural
        img = cv2.flip(img, 1)
        h_frame, w_frame, _ = img.shape
        
        # 1. Deteksi Wajah
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        overlay_text = "Mencari Wajah..."
        color = (255, 255, 255)
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            
            # --- TAHAP 1: IDENTIFIKASI ---
            if self.status == "idle":
                face_img = img[y:y+h, x:x+w]
                if face_img.size > 0:
                    try:
                        # Preprocess
                        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        face_resized = cv2.resize(face_rgb, (160, 160))
                        face_array = np.expand_dims(face_resized, axis=0)
                        
                        # Predict
                        embedding = embedder.embeddings(face_array)
                        probs = svm_model.predict_proba(embedding)[0]
                        max_prob = np.max(probs)
                        
                        if max_prob > 0.60:
                            self.name = encoder.inverse_transform([np.argmax(probs)])[0]
                            self.confidence_display = max_prob
                            self.status = "challenge_left" # Masuk tantangan
                        else:
                            overlay_text = "Wajah Tidak Dikenali"
                    except:
                        pass
            
            # --- TAHAP 2: TANTANGAN (GESER POSISI) ---
            # Hitung posisi tengah wajah
            center_x = x + (w // 2)
            
            if self.status == "challenge_left":
                overlay_text = f"Halo {self.name}! Geser ke KIRI >>"
                color = (0, 255, 255)
                # Cek jika wajah ada di 30% area kanan layar (karena mirror, jadi kiri user)
                if center_x > w_frame * 0.7: 
                    self.status = "challenge_right"
            
            elif self.status == "challenge_right":
                overlay_text = "Bagus! Geser ke KANAN <<"
                color = (0, 255, 255)
                # Cek jika wajah ada di 30% area kiri layar
                if center_x < w_frame * 0.3:
                    self.status = "verified"
            
            elif self.status == "verified":
                overlay_text = f"VERIFIED: {self.name} ({self.confidence_display*100:.0f}%)"
                color = (0, 255, 0)
            
            # Gambar Kotak & Teks
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, overlay_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        else:
            # Jika wajah hilang, reset status (opsional, biar tidak curang)
            if self.status != "verified":
                self.status = "idle"
                self.name = None
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- TAMPILAN UTAMA ---
st.title("Sistem Absensi Online")
st.write("Silakan izinkan akses kamera. Ikuti instruksi di layar.")

# Konfigurasi WebRTC
ctx = webrtc_streamer(
    key="absensi-facenet",
    mode=f.WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# --- PANEL PENCATATAN ---
if ctx.video_processor:
    # Ambil status dari processor video
    status = ctx.video_processor.status
    name = ctx.video_processor.name
    
    if status == "verified" and name:
        st.success(f"Identitas Terverifikasi: **{name}**")
        
        # Form Konfirmasi Absen
        with st.form("absen_form"):
            st.write("Apakah data ini benar?")
            col1, col2 = st.columns(2)
            submit = col1.form_submit_button("YA, Catat Absensi")
            
            if submit:
                # Catat Waktu
                now = datetime.now()
                dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
                
                # Simpan ke CSV (Perhatian: di Cloud CSV ini bersifat sementara!)
                file_csv = 'absensi_cloud.csv'
                
                # Cek apakah file ada
                if not os.path.exists(file_csv):
                    with open(file_csv, 'w') as f:
                        f.write("Nama,Waktu\n")
                
                # Tulis data
                with open(file_csv, 'a') as f:
                    f.write(f"{name},{dt_string}\n")
                
                st.balloons()
                st.success(f"Absensi {name} tercatat pada {dt_string}")
                
                # Reset status (Hack kecil)
                ctx.video_processor.status = "idle"