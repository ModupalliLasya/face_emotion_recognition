import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from PIL import Image
import time

# Load the classifier and model
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default (2).xml')
classifier = load_model('model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_count = {emotion: 0 for emotion in emotion_labels}
emotion_quotes = {
    'Angry': "Anger 👺 is one letter short of danger ☠️.",
    'Disgust': "Don't let negativity get to you! 🤬",
    'Fear': "Fear 😰 is a reaction. Courage ⚡ is a decision.",
    'Happy': "Happiness 😀 is the best medicine.",
    'Neutral': "Keep calm and carry on 🙂.",
    'Sad': "😢 It's okay to feel sad. The sun will shine again 🌞.",
    'Surprise': "Surprises 🤩 make life interesting!"
}

emotion_images = {
    'Angry': 'anger.jpg',
    'Disgust': 'disgust.jpg',
    'Fear': 'fear.jpg',
    'Happy': 'happy.jpg',
    'Neutral': 'neutral.jpg',
    'Sad': 'sad.jpg',
    'Surprise': 'surprise.jpg'
}

# Function to detect faces and emotions
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    detected_emotions = []
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            detected_emotions.append(label)
            emotion_count[label] += 1
    
    return frame, detected_emotions

# Function to show emotion summary
def show_emotion_summary():
    emotions = list(emotion_count.keys())
    counts = list(emotion_count.values())

    fig, ax = plt.subplots()
    ax.bar(emotions, counts, color=['#FF6384', '#36A2EB', '#FFCE56', '#00CC99', '#FF9999', '#CCCCFF', '#FF6633'])
    ax.set_xlabel('Emotions')
    ax.set_ylabel('Count')
    ax.set_title('Emotion Detection Count')
    
    st.pyplot(fig)

    dominant_emotion = max(emotion_count, key=emotion_count.get)
    st.subheader(f"Most Detected Emotion: {dominant_emotion}")
    st.write(emotion_quotes.get(dominant_emotion, "Enjoy the moment!"))

    # Display corresponding image
    img_path = emotion_images.get(dominant_emotion)
    if img_path:
        st.image(img_path, width=300, caption=dominant_emotion)

# Function for gradient background
def apply_gradient():
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Animation function (loading effect)
def animate_loading():
    for _ in range(3):
        with st.empty():
            st.write("Loading...")
            time.sleep(0.5)
        with st.empty():
            st.write("Processing...")
            time.sleep(0.5)

# Streamlit UI
apply_gradient()

st.title("Emotion Detection Application", anchor="left")

# Show starting image
st.image('face_img.jpg', caption="Welcome! Let's detect your emotion.", use_column_width=True)

# Camera or image logic
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []

start_button = st.button("Start Emotion Detection")
stop_button = st.button("Stop Detection")

if start_button:
    uploaded_file = st.file_uploader("Upload a photo for emotion detection", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        animate_loading()
        
        frame, detected_emotions = detect_emotion(image)
        st.image(frame, caption="Processed Image", use_column_width=True)
        
        st.session_state.emotion_history.extend(detected_emotions)

# Display emotion summary
if stop_button:
    show_emotion_summary()
