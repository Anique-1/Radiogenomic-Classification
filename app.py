import streamlit as st
import tensorflow as tf
import numpy as np
import pydicom
import cv2
import os
from tensorflow import keras
from tensorflow.keras import layers

# --- Configuration ---
MODEL_PATH = 'radiogenomic_tnbc_transformer_classifier_1.keras'
IMAGE_SIZE = (224, 224) # Must match the size used during training

# --- Load the trained model ---
@st.cache_resource
def load_model():
    # It's important to pass custom objects during loading if your model contains custom layers
    custom_objects = {"Patches": Patches, "PatchEncoder": PatchEncoder}
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    return model

# --- Custom Layers for Model Loading ---
# These must be defined before loading the model if they are part of the model's architecture
@tf.keras.utils.register_keras_serializable()
class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

@tf.keras.utils.register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim # Store projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded_patches = self.projection(patches) + self.position_embedding(positions)
        return encoded_patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config

# Load the model only once
model = load_model()

# --- Preprocessing function (must match train_model.py) ---
def preprocess_dicom_image(dicom_file, image_size):
    try:
        # Read DICOM file from bytes
        dicom = pydicom.dcmread(dicom_file)
        image = dicom.pixel_array

        # Normalize image to 0-255 if not already
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Convert to 3-channel if grayscale (many CNNs expect 3 channels)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        image = cv2.resize(image, image_size)
        image = image / 255.0 # Normalize pixel values to [0, 1]
        return image
    except Exception as e:
        st.error(f"Error loading or processing DICOM file: {e}")
        return None

# --- Streamlit App ---
st.title("Radiogenomic Classification: TNBC vs. Other Subtypes")
st.write("Upload a mammogram DICOM image to predict if the tumor is Triple-Negative Breast Cancer (TNBC) or another subtype.")

uploaded_file = st.file_uploader("Choose a DICOM (.dcm) file...", type=["dcm"])

if uploaded_file is not None:
    # Read DICOM file from bytes for display and preprocessing
    try:
        dicom_data = pydicom.dcmread(uploaded_file)
        display_image = dicom_data.pixel_array

        # Normalize for display if needed
        if display_image.dtype != np.uint8:
            display_image = cv2.normalize(display_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Convert to 3-channel for display if grayscale
        if len(display_image.shape) == 2:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
        elif display_image.shape[2] == 1:
            display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)

        st.image(display_image, caption="Uploaded DICOM Image (Preview)", use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying DICOM image: {e}")
        display_image = None # Indicate that display failed

    # Preprocess the image for the model
    # Need to reset file pointer for pydicom.dcmread in preprocess_dicom_image
    uploaded_file.seek(0) 
    processed_image = preprocess_dicom_image(uploaded_file, IMAGE_SIZE)

    if processed_image is not None:
        # Reshape for model prediction (add batch dimension)
        input_image = np.expand_dims(processed_image, axis=0)

        # Make prediction
        st.write("Making prediction...")
        prediction = model.predict(input_image)
        
        # Get predicted class and probability
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = prediction[0][predicted_class]

        class_labels = {0: "Other Subtype (Gene Expression Present)", 1: "Triple-Negative Breast Cancer (No Gene Expression)"}
        
        st.subheader("Prediction Result:")
        st.write(f"The model predicts: **{class_labels[predicted_class]}**")
        st.write(f"Confidence: **{confidence:.2f}**")

        if predicted_class == 1:
            st.success("This mammogram shows a triple-negative cancer (no gene expression).")
        else:
            st.info("This mammogram shows another subtype (gene expression present).")

else:
    st.write("Please upload a DICOM file to get a prediction.")
