import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

# Load the classifier and model
try:
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    classifier = load_model('model.h5')
except Exception as e:
    st.error(f"Error loading model or classifier: {e}")
    st.stop()

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

def detect_and_crop_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        st.warning("No faces detected!")
        return None, None

    (x, y, w, h) = faces[0]
    cropped_face = gray[y:y + h, x:x + w]
    cropped_face = cv2.resize(cropped_face, (48, 48), interpolation=cv2.INTER_AREA)
    return cropped_face, faces[0]

def predict_emotion(cropped_face):
    if cropped_face is None:
        return None

    try:
        cropped_face = cropped_face.astype('float') / 255.0
        cropped_face = img_to_array(cropped_face)
        cropped_face = np.expand_dims(cropped_face, axis=0)

        prediction = classifier.predict(cropped_face)[0]
        return emotion_labels[prediction.argmax()]

    except Exception as e:
        st.error(f"Error in emotion prediction: {e}")
        return None

def show_emotion_summary(emotion):
    st.subheader(f"Detected Emotion: {emotion}")
    st.write(emotion_quotes.get(emotion, "Enjoy the moment!"))

    img_path = emotion_images.get(emotion)
    if img_path:
        st.image(img_path, width=300, caption=emotion)

# Streamlit UI
st.title("Emotion Detection Application")

# Show starting image
st.image('face_img.jpg', caption="Welcome! Let's detect your emotion.", use_column_width=True)

# Image Upload Logic
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        image = np.array(Image.open(uploaded_file).convert('RGB'))  # Convert to RGB
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Detect and crop face
        cropped_face, face_coords = detect_and_crop_face(image)

        if cropped_face is not None:
            # Predict the emotion based on cropped face
            detected_emotion = predict_emotion(cropped_face)

            if detected_emotion:
                show_emotion_summary(detected_emotion)
            else:
                st.write("Emotion could not be detected.")
        else:
            st.write("No face detected in the image.")

    except Exception as e:
        st.error(f"Error processing image: {e}")
