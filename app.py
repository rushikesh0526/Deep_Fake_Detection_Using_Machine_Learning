import streamlit as st
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import cv2
import tempfile
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="AI Deepfake Detection System",
    layout="wide"
)

# =========================================
# CUSTOM CSS
# =========================================
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
/* Main Heading */
h1 {
    font-size: 50px !important;
    font-weight: bold;
    color: #ff4b4b;
}
/* General Text */
p, label, div {
    font-size: 20px !important;
}
/* Sidebar */
section[data-testid="stSidebar"] {
    width: 260px !important;
    background-color: #f5f5f5;
}
/* Buttons */
.stButton>button {
    font-size: 20px !important;
    height: 3em;
    width: 220px;
    border-radius: 10px;
    background-color: #ff4b4b;
    color: white;
}
/* File uploader */
[data-testid="stFileUploader"] {
    font-size: 18px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# TITLE
# =========================================
st.title("🧠 AI Deepfake Detection System")

st.markdown("""
<h3 style='font-size:26px;'>
Upload an image or video to detect deepfake content.
</h3>
""", unsafe_allow_html=True)

# =========================================
# DEVICE
# =========================================
DEVICE = "cpu"

# =========================================
# LOAD FACE DETECTOR
# =========================================
@st.cache_resource
def load_mtcnn():

    return MTCNN(
        select_largest=False,
        post_process=False,
        device=DEVICE
    ).eval()

# =========================================
# LOAD MODEL
# =========================================
@st.cache_resource
def load_model():

    model = InceptionResnetV1(
        pretrained="vggface2",
        classify=True,
        num_classes=1
    )

    checkpoint = torch.load(
        "resnetinceptionv1_epoch_32.pth",
        map_location=torch.device("cpu")
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model

# =========================================
# INITIALIZE MODELS
# =========================================
mtcnn = load_mtcnn()
model = load_model()

# =========================================
# IMAGE PREDICTION
# =========================================
def predict_image(image):

    face = mtcnn(image)

    if face is None:
        return None, None

    face = F.interpolate(
        face.unsqueeze(0),
        size=(256, 256)
    )

    face = face.to(DEVICE).float() / 255.0

    with torch.no_grad():

        output = model(face).squeeze()

        fake_prob = torch.sigmoid(
            output
        ).item()

    label = (
        "REAL"
        if fake_prob < 0.5
        else "DEEPFAKE"
    )

    return label, fake_prob

# =========================================
# VIDEO FRAME EXTRACTION
# =========================================
def sample_frames(
    video_path,
    num_frames=20
):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames == 0:
        return []

    indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int
    )

    frames = []

    for idx in indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            idx
        )

        ret, frame = cap.read()

        if ret:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(
                Image.fromarray(frame)
            )

    cap.release()

    return frames

# =========================================
# SIDEBAR
# =========================================
st.sidebar.title("⚙️ System Settings")

mode = st.sidebar.radio(
    "Select Mode",
    ["Image", "Video"]
)

st.sidebar.markdown("""
---
### 🔍 Features
✅ Image Analysis  
✅ Video Analysis  
✅ Deepfake Detection  
✅ CNN + LSTM Model  
""")

# =========================================
# IMAGE MODE
# =========================================
if mode == "Image":

    st.markdown("""
    <h2 style='font-size:32px;'>
    📷 Upload Image
    </h2>
    """, unsafe_allow_html=True)

    uploaded_image = st.file_uploader(
        "",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        image = Image.open(
            uploaded_image
        ).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            width=350
        )

        if st.button("Analyze Image"):

            with st.spinner(
                "Analyzing image..."
            ):

                label, prob = predict_image(
                    image
                )

            if label is None:

                st.error(
                    "❌ No face detected"
                )

            else:

                if label == "REAL":

                    st.markdown(
                        f"""
                        <h1 style='color:green;'>
                        ✅ Prediction: {label}
                        </h1>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        f"""
                        <h1 style='color:red;'>
                        🚨 Prediction: {label}
                        </h1>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown(
                    f"""
                    <h2>
                    Fake Probability:
                    {prob:.2f}
                    </h2>
                    """,
                    unsafe_allow_html=True
                )

# =========================================
# VIDEO MODE
# =========================================
if mode == "Video":

    st.markdown("""
    <h2 style='font-size:32px;'>
    🎥 Upload Video
    </h2>
    """, unsafe_allow_html=True)

    uploaded_video = st.file_uploader(
        "",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video:

        st.video(
            uploaded_video,
            format="video/mp4"
        )

        tfile = tempfile.NamedTemporaryFile(
            delete=False
        )

        tfile.write(
            uploaded_video.read()
        )

        if st.button("Analyze Video"):

            with st.spinner(
                "Processing video..."
            ):

                frames = sample_frames(
                    tfile.name,
                    20
                )

                probs = []

                for frame in frames:

                    label, prob = predict_image(
                        frame
                    )

                    if prob is not None:

                        probs.append(prob)

                if len(probs) == 0:

                    st.error(
                        "❌ No faces detected"
                    )

                else:

                    avg_fake = sum(
                        probs
                    ) / len(probs)

                    final_label = (
                        "REAL VIDEO"
                        if avg_fake < 0.5
                        else "DEEPFAKE VIDEO"
                    )

                    if avg_fake < 0.5:

                        st.markdown(
                            f"""
                            <h1 style='color:green;'>
                            ✅ {final_label}
                            </h1>
                            """,
                            unsafe_allow_html=True
                        )

                    else:

                        st.markdown(
                            f"""
                            <h1 style='color:red;'>
                            🚨 {final_label}
                            </h1>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown(
                        f"""
                        <h2>
                        Average Fake Probability:
                        {avg_fake:.2f}
                        </h2>
                        """,
                        unsafe_allow_html=True
                    )
