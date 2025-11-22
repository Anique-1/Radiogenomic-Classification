import pandas as pd
import numpy as np
import os
import pydicom
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# --- Configuration (from existing train_model.py, potentially adjusted) ---
EXCEL_PATH = 'Balanced_TNBC_vs_Others.xlsx'
DICOM_DIR = 'ID-20251121T162146Z-1-001/ID/'
IMAGE_SIZE = (224, 224) # Standard size
BATCH_SIZE = 32
EPOCHS = 100 # Reduced for initial testing
NUM_CLASSES = 2 # TNBC or Not TNBC

# Transformer specific configurations
PATCH_SIZE = 16  # Size of the patches to be extracted from the input images
NUM_PATCHES = (IMAGE_SIZE[0] // PATCH_SIZE) ** 2
PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_LAYERS = 8
MLP_HEAD_UNITS = [2048, 1024]  # Size of the top MLP head

# --- 1. Load Data and Prepare Labels (reusing from train_model.py) ---
def load_labels(excel_path):
    df = pd.read_excel(excel_path)
    df['is_tnbc'] = (df['subtype'] == 'triple negative').astype(int)
    return df[['ID1', 'is_tnbc']]

# --- 2. Map Images to Labels (reusing from train_model.py) ---
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

# --- 3. Load and Preprocess DICOM Images (reusing from train_model.py) ---
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

# --- Transformer Model Components ---

# 4. Patch Extractor
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

# 5. Patch Encoder
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

# 6. Transformer Block (Multi-head Self-Attention + MLP)
def transformer_block(encoded_patches, projection_dim, num_heads):
    # Layer normalization 1
    x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # Multi-head self-attention
    attention_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim, dropout=0.1
    )(x1, x1)
    # Skip connection 1
    x2 = layers.Add()([attention_output, encoded_patches])
    # Layer normalization 2
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    # MLP
    x3 = keras.Sequential(
        [
            layers.Dense(units=projection_dim * 2, activation="relu"),
            layers.Dense(units=projection_dim),
        ]
    )(x3)
    # Skip connection 2
    encoded_patches = layers.Add()([x3, x2])
    return encoded_patches

# 7. Create Vision Transformer Model
def create_vit_classifier():
    inputs = layers.Input(shape=IMAGE_SIZE + (3,)) # (224, 224, 3)

    # Augment data if needed (can be integrated here or before training loop)
    # We will assume data is pre-augmented or augmentation is handled by tf.data or ImageDataGenerator

    # Create patches
    patches = Patches(PATCH_SIZE)(inputs)
    # Encode patches
    encoded_patches = PatchEncoder(NUM_PATCHES, PROJECTION_DIM)(patches)

    # Create multiple transformer blocks
    for _ in range(TRANSFORMER_LAYERS):
        encoded_patches = transformer_block(
            encoded_patches, PROJECTION_DIM, NUM_HEADS
        )

    # Create a [CLS] token equivalent by taking the mean of all encoded patches
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    # Add MLP head.
    for units in MLP_HEAD_UNITS:
        representation = layers.Dense(units, activation="relu")(representation)
        representation = layers.Dropout(0.5)(representation)
    logits = layers.Dense(NUM_CLASSES, activation="softmax")(representation)

    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting radiogenomic transformer classification model training...")

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

    # Convert labels to categorical (one-hot encoding)
    y = to_categorical(y, num_classes=NUM_CLASSES)

    # Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

    # Create and compile model
    print("Creating and compiling Vision Transformer model...")
    transformer_model = create_vit_classifier()
    transformer_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    transformer_model.summary()

    # Train the model
    print("Training the transformer model...")
    history = transformer_model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
    )

    # Evaluate the model
    print("Evaluating the model...")
    loss, accuracy = transformer_model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save the model
    model_save_path = 'radiogenomic_tnbc_transformer_classifier_1.keras'
    transformer_model.save(model_save_path)
    print(f"Transformer model saved to {model_save_path}")

    print("Transformer model training complete.")