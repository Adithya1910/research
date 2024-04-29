import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Define a custom function to load the Keras model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("keras_model.h5")
    return model

# Define the class names
class_names = ["Class 1", "Class 2"]

# Function to preprocess and predict hand gestures
def predict_gesture(model, image):
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

# Streamlit UI
st.title("Hand Gesture Recognition")

uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if st.button("Predict") and uploaded_files:
    try:
        model = load_model()  # Load the model

        cols = st.columns(2)
        for i, uploaded_file in enumerate(uploaded_files):
            # Read and decode the uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display the image and perform the prediction
            with cols[i % 2]:
                st.image(image_rgb, width=200, caption=f'Image {i+1}: {uploaded_file.name}')
                class_name, confidence_score = predict_gesture(model, image)
                st.write(f"Predicted Gesture: {class_name}")
                st.write(f"Confidence Score: {confidence_score}")

                # Display the prediction result
                if class_name == 'Class 1':
                    st.markdown("<h3 style='text-align: center; color: blue;'>It's Class 1</h3>", unsafe_allow_html=True)
                elif class_name == 'Class 2':
                    st.markdown("<h3 style='text-align: center; color: green;'>It's Class 2</h3>", unsafe_allow_html=True)
    except OSError:
        st.error("Error loading the model. Please ensure that the model file exists and is accessible.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display a thank you message
st.markdown("<h4 style='text-align: center; color: orange;'>Thanks for using our Hand Gesture Recognition App!</h4>", unsafe_allow_html=True)
