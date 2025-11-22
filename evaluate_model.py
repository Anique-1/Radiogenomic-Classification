import pandas as pd
import numpy as np
import os
import pydicom
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers

# --- Configuration ---
EXCEL_PATH = 'Balanced_TNBC_vs_Others.xlsx'
DICOM_DIR = 'ID-20251121T162146Z-1-001/ID/'
IMAGE_SIZE = (224, 224) # Standard size for many CNNs
MODEL_PATH = 'radiogenomic_tnbc_transformer_classifier_1.keras'

# --- Custom Layer Definitions (must match training script) ---
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
        self.projection_dim = projection_dim
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

# --- 1. Load Data and Prepare Labels ---
def load_labels(excel_path):
    df = pd.read_excel(excel_path)
    # Create a binary label: 1 for 'triple negative', 0 for others
    df['is_tnbc'] = (df['subtype'] == 'triple negative').astype(int)
    return df[['ID1', 'is_tnbc']]

# --- 2. Map Images to Labels ---
def map_images_to_labels(df_labels, dicom_dir):
    image_paths = []
    labels = []
    for index, row in df_labels.iterrows():
        patient_id = row['ID1']
        tnbc_label = row['is_tnbc']
        patient_dicom_dir = os.path.join(dicom_dir, patient_id)
        
        if os.path.exists(patient_dicom_dir):
            dicom_files = [os.path.join(patient_dicom_dir, f) for f in os.listdir(patient_dicom_dir) if f.endswith('.dcm')]
            for dicom_file in dicom_files:
                image_paths.append(dicom_file)
                labels.append(tnbc_label)
        else:
            print(f"Warning: DICOM directory not found for patient ID: {patient_id}")
    return image_paths, np.array(labels)

# --- 3. Load and Preprocess DICOM Images ---
def load_and_preprocess_dicom(dicom_path, image_size):
    try:
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array

        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        image = cv2.resize(image, image_size)
        return image
    except Exception as e:
        print(f"Error loading or processing {dicom_path}: {e}")
        return None

def create_dataset(image_paths, labels, image_size):
    images = []
    processed_labels = []
    for i, path in enumerate(image_paths):
        img = load_and_preprocess_dicom(path, image_size)
        if img is not None:
            images.append(img)
            processed_labels.append(labels[i])
    return np.array(images), np.array(processed_labels)

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting model evaluation...")

    # Load labels
    print(f"Loading labels from {EXCEL_PATH}...")
    df_labels = load_labels(EXCEL_PATH)
    print(f"Labels loaded. Total patients: {len(df_labels)}")

    # Map images to labels
    print(f"Mapping images from {DICOM_DIR} to labels...")
    image_paths, labels = map_images_to_labels(df_labels, DICOM_DIR)
    print(f"Found {len(image_paths)} images with corresponding labels.")

    # Create dataset
    print(f"Loading and preprocessing {len(image_paths)} DICOM images...")
    X, y = create_dataset(image_paths, labels, IMAGE_SIZE)
    print(f"Dataset created. X shape: {X.shape}, y shape: {y.shape}")

    # Normalize pixel values to [0, 1]
    X = X / 255.0

    # Convert labels to categorical (one-hot encoding) for model evaluation
    y_categorical = to_categorical(y, num_classes=2)

    # Split data into training and testing sets (must be consistent with training)
    print("Splitting data into training and testing sets...")
    _, X_test, _, y_test_categorical = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y)
    _, _, _, y_test_true = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Get true labels for F1 score

    print(f"Testing data shape: {X_test.shape}, {y_test_categorical.shape}")

    # Load the trained model with custom objects
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}. Please train the model first using train_model.py")
    else:
        # Load model with custom objects
        custom_objects = {
            'Patches': Patches,
            'PatchEncoder': PatchEncoder
        }
        model = load_model(MODEL_PATH, custom_objects=custom_objects)
        model.summary()

        # Evaluate the model
        print("Evaluating the model on the test set...")
        loss, accuracy = model.evaluate(X_test, y_test_categorical, verbose=0)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

        # Predict probabilities for the test set
        y_pred_probs = model.predict(X_test)
        # Convert probabilities to class labels
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Calculate F1 Score
        f1 = f1_score(y_test_true, y_pred, average='weighted') # Use 'weighted' for imbalanced classes
        print(f"Test F1 Score (weighted): {f1:.4f}")

    print("Model evaluation complete.")