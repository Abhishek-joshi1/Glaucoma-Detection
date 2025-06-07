import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K

# Custom layers and functions for segmentation model
class FixedDropout(layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)

def swish(x):
    return x * keras.activations.sigmoid(x)

# Load ensemble models for detection
detection_model_paths = ["models/models/ensemble/inceptionClaheNew1.keras", "models/models/ensemble/MobClaheNew1.keras", "models/models/ensemble/NasClahe1.keras", "models/models/ensemble/VggClahe1.keras"]
detection_models = [tf.keras.models.load_model(path) for path in detection_model_paths]

# Preprocessing functions
def preprocess_image(image, target_size=(256, 256)):
    image = np.array(image)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    image_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    image_resized = cv2.resize(image_clahe, target_size)
    image_array = np.expand_dims(image_resized, axis=0) / 255.0
    return image_array

def ensemble_prediction(image):
    predictions = [model.predict(preprocess_image(image)) for model in detection_models]
    avg_prediction = np.mean(predictions, axis=0)
    confidence = np.max(avg_prediction)  # get the confidence value as the max probability
    prediction = np.argmax(avg_prediction)
    return prediction, confidence

def preprocess_segmentation_image(image, target_size=(128, 128)):
    image_array = np.array(image)
    image_array = cv2.resize(image_array, target_size)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def load_segmentation_model(fundus_type):
    model_path = "models/models/segment/glaucomasegmentationNew.h5" if fundus_type == "Zoomed Fundus" else "models/models/segment/glaucomasegmentation.h5"
    
    return keras.models.load_model(model_path, compile=False, custom_objects={'swish': swish, 'FixedDropout': FixedDropout})

def calculate_vertical_diameter(mask, class_id):
    binary_mask = (mask == class_id).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    largest_contour = max(contours, key=cv2.contourArea)
    _, y_min, _, y_max = cv2.boundingRect(largest_contour)
    return abs(y_max - y_min)

def calculate_vertical_cdr(predicted_mask):
    disk_diameter = calculate_vertical_diameter(predicted_mask, 2)
    cup_diameter = calculate_vertical_diameter(predicted_mask, 1)
    if disk_diameter != 0 and cup_diameter != 0:
        print("Vertical Cup Diameter:", cup_diameter)
        print("Vertical Disk Diameter:", disk_diameter)
        if cup_diameter > disk_diameter:
            return round(disk_diameter / cup_diameter, 3)
        else:
            return round(cup_diameter / disk_diameter, 3)
            
        
    else:
        return None


def overlay_mask_on_image(image, predicted_mask, alpha=0.5):
    original_image = np.array(image)
    mask_resized = cv2.resize(predicted_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    color_mask = np.zeros_like(original_image, dtype=np.uint8)
    color_mask[mask_resized == 1] = [255, 0, 0]
    color_mask[mask_resized == 2] = [0, 0, 255]
    overlay_image = cv2.addWeighted(original_image, 1 - alpha, color_mask, alpha, 0)
    return overlay_image

def main():
    st.title("Glaucoma Detection with Ensemble Learning & Segmentation")
    fundus_type = st.radio("Select Fundus Type:", ["Zoomed Fundus", "Normal Fundus"])
    uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Detection with ensemble learning
        detection_result, confidence = ensemble_prediction(image)
        glaucoma_status = "Glaucoma Detected" if detection_result == 0 else "No Glaucoma Detected"
        st.subheader("Detection Result")
        st.write(f"{glaucoma_status} (Confidence: {confidence:.2f})")
        
        # Segmentation
        st.subheader("Segmentation and Vertical CDR Analysis")
        segmentation_model = load_segmentation_model(fundus_type)
        segmentation_input = preprocess_segmentation_image(image)
        segmentation_prediction = segmentation_model.predict(segmentation_input)
        predicted_mask = np.argmax(segmentation_prediction[0], axis=-1)

        # Calculate Vertical CDR
        vertical_cdr = calculate_vertical_cdr(predicted_mask)
        st.write(f"*Vertical Cup-to-Disc Ratio (CDR): {vertical_cdr if vertical_cdr is not None else 'Could not be calculated'}*")
        
        # Visualization
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Original Image")
            st.image(image, use_container_width=True)
        with col2:
            st.write("Predicted Mask")
            plt.figure(figsize=(5, 5))
            plt.imshow(predicted_mask, cmap='viridis')
            plt.axis('off')
            st.pyplot(plt)
        with col3:
            st.write("Overlay with Colors")
            overlay_image = overlay_mask_on_image(image, predicted_mask)
            st.image(overlay_image, use_container_width=True)

if __name__ == "__main__":
    main() 