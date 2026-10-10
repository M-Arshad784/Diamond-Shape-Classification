import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Diamond Shape AI",
    page_icon="💎",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

    /* Prediction Result Text */
    [data-testid="stAlert"] p {
        color: rgb(170, 180, 212) !important;
    }

    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #070b1a, #10162e, #0b1023);
        color: white;
    }

    /* Hide Streamlit Menu */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Main Title */
    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        margin-top: 10px;
        margin-bottom: 5px;
        background: linear-gradient(90deg, #8be9fd, #c7b8ff, #ffffff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #aab4d4;
        font-size: 18px;
        margin-bottom: 35px;
    }

    /* Diamond Icon */
    .diamond-icon {
        text-align: center;
        font-size: 70px;
        margin-bottom: -10px;
        filter: drop-shadow(0px 0px 20px #8be9fd);
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }

    /* Card Heading */
    .card-title {
        color: #8be9fd;
        font-size: 23px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    /* Result Card */
    .result-card {
        background: linear-gradient(
            135deg,
            rgba(139, 233, 253, 0.12),
            rgba(199, 184, 255, 0.12)
        );
        border: 1px solid rgba(139, 233, 253, 0.35);
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 0 30px rgba(139, 233, 253, 0.08);
    }

    .result-label {
        color: #9da8c7;
        font-size: 16px;
    }

    .result-name {
        color: white;
        font-size: 34px;
        font-weight: 800;
        margin: 8px 0;
    }

    .confidence {
        color: #8be9fd;
        font-size: 20px;
        font-weight: 600;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        border: none;
        padding: 12px;
        font-size: 17px;
        font-weight: 700;
        background: linear-gradient(90deg, #6c63ff, #8be9fd);
        color: white;
        transition: 0.3s;
    }

    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 25px rgba(139, 233, 253, 0.35);
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 15px;
        padding: 10px;
        color: black;
    }

    /* Footer */
    .custom-footer {
        text-align: center;
        color: #77809e;
        margin-top: 40px;
        padding: 20px;
        font-size: 14px;
    }

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = YOLO(os.path.join(BASE_DIR, "best.pt"))


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="diamond-icon">💎</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">Diamond Shape AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered Diamond Shape Classification using Computer Vision'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# MAIN COLUMNS
# --------------------------------------------------

col1, col2 = st.columns([1, 1], gap="large")


# --------------------------------------------------
# LEFT SIDE - UPLOAD
# --------------------------------------------------

with col1:

    st.markdown(
        """
        <div class="glass-card">
            <div class="card-title">💎 Upload Diamond</div>
            <p style="color:#aab4d4;">
                Upload a diamond image and let the AI identify
                its shape.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    upload_file = st.file_uploader(
        "Choose a Diamond Image",
        type=["jpg", "jpeg", "png"]
    )

    if upload_file is not None:

        image = Image.open(upload_file)

        st.image(
            image,
            caption="Uploaded Diamond",
            width="stretch"
        )

        st.markdown(
            """
            <p style="text-align:center;color:#77809e;">
                Image successfully uploaded ✓
            </p>
            """,
            unsafe_allow_html=True
        )

        classify = st.button("✨ Classify Diamond Shape")


# --------------------------------------------------
# RIGHT SIDE - RESULT
# --------------------------------------------------

with col2:

    st.markdown(
        """
        <div class="glass-card">
            <div class="card-title">🔬 AI Prediction</div>
            <p style="color:#aab4d4;">
                The trained YOLO model will analyze the uploaded
                image and predict the diamond shape.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if upload_file is not None and classify:

        # Run Prediction
        results = model(image)

        # Get Prediction
        result = results[0]

        class_id = result.probs.top1
        confidence = result.probs.top1conf.item()
        class_name = model.names[class_id]

        # Show Result
        st.success(f"💎 Predicted Diamond Shape: {class_name}")

        st.info(f"🎯 Confidence: {confidence * 100:.2f}%")

        st.progress(confidence)

    elif upload_file is None:

        st.info("👆 Upload a diamond image to start classification.")


# --------------------------------------------------
# PROJECT INFORMATION
# --------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

info1, info2, info3 = st.columns(3)

with info1:
    st.markdown(
        """
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:35px;">🤖</div>
            <b>AI Powered</b>
            <p style="color:#77809e;">
                Deep Learning based classification
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with info2:
    st.markdown(
        """
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:35px;">👁️</div>
            <b>Computer Vision</b>
            <p style="color:#77809e;">
                Image-based diamond analysis
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with info3:
    st.markdown(
        """
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:35px;">🎯</div>
            <b>Confidence Score</b>
            <p style="color:#77809e;">
                View model prediction confidence
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="custom-footer">
        💎 Diamond Shape Classification |
        Powered by YOLO • Python • Computer Vision • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)