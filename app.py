import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import io
import streamlit as st
import numpy as np
from PIL import Image
import json
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer
import av
import threading
import time

@st.cache_resource
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model("final_model.h5", compile=False)

def load_labels():
    p = os.path.join(os.path.dirname(__file__), "labels.txt")
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            return [x.strip() for x in f if x.strip()]
    return None

def preprocess(img, size):
    img = img.convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def speak(text, rate=1.0, lang="en-US"):
    components.html(
        f"""
        <script>
        const t = {json.dumps(text)};
        const u = new SpeechSynthesisUtterance(t);
        u.lang = {json.dumps(lang)};
        u.rate = {rate};
        speechSynthesis.cancel();
        speechSynthesis.speak(u);
        </script>
        """,
        height=0,
    )

def main():
    st.set_page_config(page_title="AI Based Safety Navigation System", page_icon="🧭", layout="centered")
    st.title("AI Based Safety Navigation System")
    audio_enabled = st.toggle("Audio feedback", value=True)
    try:
        model = load_model()
    except Exception:
        model = None
    labels = load_labels()
    tab1, tab2, tab3 = st.tabs(["Upload", "Camera", "Video"])
    with tab1:
        file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "bmp"])
    with tab2:
        camera_file = st.camera_input("Take a photo")
    with tab3:
        st.subheader("Live Video")
        record_enabled = st.checkbox("Record to GIF", value=False)
        lock = threading.Lock()
        shared = {"label": None, "frames": []}
        def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_image()
            if model is not None:
                x = preprocess(img, (224, 224))
                probs = model.predict(x)[0]
                idx = int(np.argmax(probs))
                name_v = labels[idx] if labels and idx < len(labels) else f"class_{idx}"
                with lock:
                    shared["label"] = name_v
                    if record_enabled:
                        shared["frames"].append(img.copy())
            return frame
        ctx = webrtc_streamer(
            key="live-video",
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
        )
        live_label = st.empty()
        while ctx.state.playing:
            with lock:
                name_v = shared.get("label")
            if name_v:
                live_label.write(f"Live: {name_v}")
                if audio_enabled:
                    if st.session_state.get("_last_live") != name_v:
                        speak(name_v)
                        st.session_state["_last_live"] = name_v
            time.sleep(0.1)
        if record_enabled and shared["frames"]:
            if st.button("Save GIF"):
                import imageio
                buf = io.BytesIO()
                imageio.mimsave(buf, [f.resize((320, 240)) for f in shared["frames"]], format="GIF", duration=0.1)
                st.download_button("Download recording.gif", data=buf.getvalue(), file_name="recording.gif", mime="image/gif")
    img = None
    if file is not None:
        img = Image.open(file)
    elif camera_file is not None:
        img = Image.open(io.BytesIO(camera_file.getvalue()))
    if img is not None:
        st.image(img, width=min(700, img.width))
        if model is not None:
            x = preprocess(img, (224, 224))
            probs = model.predict(x)[0]
            idx = int(np.argmax(probs))
            conf = float(np.max(probs))
            name = labels[idx] if labels and idx < len(labels) else f"class_{idx}"
            st.subheader("Prediction")
            st.write(name)
            if audio_enabled:
                speak(name)

if __name__ == "__main__":
    main()
