import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import io

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

def detect_and_crop_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to gray for face detection
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return None, None

    # Use the first detected face
    (x, y, w, h) = faces[0]
    cropped_face = gray[y:y + h, x:x + w]
    cropped_face = cv2.resize(cropped_face, (48, 48), interpolation=cv2.INTER_AREA)
    return cropped_face, faces[0]

def predict_emotion(cropped_face):
    if cropped_face is None:
        return None

    cropped_face = cropped_face.astype('float') / 255.0
    cropped_face = img_to_array(cropped_face)
    cropped_face = np.expand_dims(cropped_face, axis=0)

    prediction = classifier.predict(cropped_face)[0]
    return emotion_labels[prediction.argmax()]

def show_emotion_summary(emotion):
    st.subheader(f"Detected Emotion: {emotion}")
    st.write(emotion_quotes.get(emotion, "Enjoy the moment!"))

    img_path = emotion_images.get(emotion)
    if img_path:
        st.image(img_path, width=300, caption=emotion)

# Streamlit UI
st.title("Emotion Detection Application")

# Start capturing video from webcam
camera = st.camera_input("Capture a photo")

if camera:
    # Read the uploaded image from the camera input
    image = Image.open(camera)
    frame = np.array(image)  # Convert PIL image to numpy array

    # Detect and crop face
    cropped_face, face_coords = detect_and_crop_face(frame)

    if cropped_face is not None:
        # Predict the emotion based on cropped face
        detected_emotion = predict_emotion(cropped_face)

        if detected_emotion:
            show_emotion_summary(detected_emotion)
        else:
            st.write("Emotion could not be detected.")
    else:
        st.write("No face detected in the image.")
