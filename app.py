import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

# Load the classifier and model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classifier = load_model('model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
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

# Function to detect and crop face
def detect_and_crop_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None, None

    (x, y, w, h) = faces[0]
    cropped_face = gray[y:y + h, x:x + w]
    cropped_face = cv2.resize(cropped_face, (48, 48), interpolation=cv2.INTER_AREA)
    return cropped_face, faces[0]

# Function to predict emotion
def predict_emotion(cropped_face):
    if cropped_face is None:
        return None

    cropped_face = cropped_face.astype('float') / 255.0
    cropped_face = img_to_array(cropped_face)
    cropped_face = np.expand_dims(cropped_face, axis=0)

    prediction = classifier.predict(cropped_face)[0]
    return emotion_labels[prediction.argmax()]

# Streamlit UI
st.title("Live Emotion Detection")

# Display video feed
stframe = st.empty()

# Start the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        st.warning("Unable to access the camera.")
        break

    cropped_face, _ = detect_and_crop_face(frame)
    detected_emotion = predict_emotion(cropped_face)

    if detected_emotion:
        st.write(f"Detected Emotion: {detected_emotion}")
        st.write(emotion_quotes.get(detected_emotion, "No quote available."))

        img_path = emotion_images.get(detected_emotion)
        if img_path:
            st.image(img_path, width=300, caption=detected_emotion)

    # Display the video feed
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    stframe.image(frame, channels='RGB', use_column_width=True)

cap.release()
