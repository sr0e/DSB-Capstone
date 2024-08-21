import streamlit as st
import pickle
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tempfile
import os
from PIL import Image
from grad_cam import make_gradcam_heatmap, save_and_display_gradcam


st.title('Brain Tumor Detection with Grad-CAM')
upload_img = st.file_uploader('Place image here...', type=['jpg', 'jpeg', 'png'])

with open('tumor-detector.pkl', 'rb') as f:
    model = pickle.load(f)

def apply_modifications(model, custom_objects=None):
    # Save and reload the model to apply modifications
    model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.keras')
    try:
        model.save(model_path)
        return load_model(model_path, custom_objects=custom_objects)
    finally:
        os.remove(model_path)

def prepare_model_for_gradcam(model):
    # Modify the last layer's activation to linear (or another activation)
    model.layers[-1].activation = tensorflow.keras.activations.linear
    # Recompile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6),
        metrics=['accuracy']
    )
    # Apply the modifications
    return apply_modifications(model)

# Load and prepare your model
model2 = prepare_model_for_gradcam(model)


class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def preprocess_image(img, size):
    img = img.convert('RGB')
    img = img.resize(size)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


if upload_img is not None:
    img = Image.open(upload_img)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    img_array = preprocess_image(img, size=(256, 256))
    preds = model2.predict(img_array)
    predicted_class_index = np.argmax(preds[0])
    predicted_class_name = class_names[predicted_class_index]
    st.write(f"Predicted class: {predicted_class_name}")

    heatmap = make_gradcam_heatmap(img_array, model2, 'block14_sepconv2_bn')
    cam_buffer = save_and_display_gradcam(img, heatmap)
    st.image(cam_buffer, caption='Grad-CAM Visualization')