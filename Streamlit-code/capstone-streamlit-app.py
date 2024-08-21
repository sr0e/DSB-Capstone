import streamlit as st
import pickle
import tensorflow
import numpy as np
from PIL import Image
from tensorflow.keras.applications.xception import preprocess_input
from grad_cam import make_gradcam_heatmap, save_and_display_gradcam


st.title('🧠 Brain Tumor Detection with Grad-CAM')
upload_img = st.file_uploader('Place image here...', type=['jpg', 'jpeg', 'png'])

st.write('Your prediction will appear below')

model = pickle.load(open('tumor-detector.pkl', 'rb'))
# with open('tumor-detector.pkl', 'rb') as f:
#     model = pickle.load(f)

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
    model.layers[-1].activation = None
    preds = model.predict(img_array)
    predicted_class_index = np.argmax(preds[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = preds[0][predicted_class_index]
    st.write(f"Predicted class: {predicted_class_name} with confidence {confidence:.2f}")

    heatmap = make_gradcam_heatmap(img_array, model, 'block14_sepconv2_bn')
    cam_buffer = save_and_display_gradcam(img, heatmap)
    st.image(cam_buffer, caption='Grad-CAM Visualization')