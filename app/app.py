import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pickle
import requests
from io import BytesIO

# Load the pre-trained model
try:
    # model = load_model(
    #     r"C:\progremming ask zagzig\AI_0\doc\DOC_Emotion Prediction\app\models\modelF.keras"
    # )
    # Ensure this path is correct
    model = pickle.load(open("/mount/src/doc/app/models/model.pkl", "rb"))
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
# Emotion to emoji mapping
emotion_to_emoji = {
    "Angry": "😡",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😊",
    "Sad": "😞",
    "Surprise": "😲",
    "Neutral": "😐",
}

# Set a custom background and layout
st.markdown(
    """
    <style>
        .container {
            background-color: #333333;
            padding: 50px;
            border-radius: 20px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
        .header {
            font-size: 2em;
            color: #FF4B4B;
            text-align: center;
            margin-bottom: 20px;
        }
        .card {
            background-color: #444444;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.3);
        }
    </style>
""",
    unsafe_allow_html=True,
)

# App title with emoji
st.markdown(
    '<h1 class="title">Emotion Classification App 😄</h1>', unsafe_allow_html=True
)

# Upload or URL input
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
with col2:
    image_url = st.text_input("Or, Enter Image URL:")

# If an image is provided
if uploaded_file is not None or image_url:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("### Processing Image...")

    if uploaded_file:
        img = Image.open(uploaded_file)
    else:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))

    st.image(img, caption="Image for Prediction", use_container_width=True)

    # Image preprocessing
    img = img.resize((48, 48)).convert("L")  # Convert to grayscale
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_emotion = emotion_labels[np.argmax(prediction)]

    # Get corresponding emoji
    predicted_emotion_with_emoji = emotion_to_emoji.get(predicted_emotion, "😐")

    # Display prediction with emoji
    st.write(
        f"**Predicted Emotion:** {predicted_emotion_with_emoji} {predicted_emotion}"
    )
    st.markdown("</div>", unsafe_allow_html=True)
