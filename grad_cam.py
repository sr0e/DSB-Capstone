import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import tensorflow
from tensorflow import keras
from tensorflow.keras.utils import array_to_img
from IPython.display import Image, display
from PIL import Image
import io


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tensorflow.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tensorflow.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        preds = tensorflow.convert_to_tensor(preds)
        if pred_index is None:
            pred_index = tensorflow.argmax(preds[0])
        else:
            pred_index = tensorflow.convert_to_tensor(pred_index)

        class_channel = tensorflow.gather(preds[0], pred_index)

    grads = tape.gradient(class_channel, conv_output)

    print(f"Grads: {grads}")
    print(f"Grads mean: {tensorflow.reduce_mean(grads)}")
    print(f"Grads max: {tensorflow.reduce_max(grads)}")
    print(f"Grads min: {tensorflow.reduce_min(grads)}")

    if grads is None or tensorflow.reduce_all(tensorflow.math.is_nan(grads)):
        raise ValueError("Gradients are None or contain NaNs.")

    pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    heatmap = tensorflow.reduce_sum(conv_output * pooled_grads, axis=-1)

    print(f"heatmap values before normalization: {heatmap}")
    
    heatmap = tensorflow.maximum(heatmap, 0)
    heatmap = heatmap / tensorflow.reduce_max(heatmap)

    print(f"heatmap values after normalization: {heatmap}")

    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = np.array(img)
    img = np.uint8(img)

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap('jet')

    jet_heatmap = jet(heatmap)[:, :, :3]
    jet_heatmap = (jet_heatmap * 255).astype(np.uint8)

    jet_heatmap = Image.fromarray(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]), Image.BILINEAR)
    jet_heatmap = np.array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    superimposed_img = Image.fromarray(superimposed_img)
    return superimposed_img
    # buffer = io.BytesIO()
    # superimposed_img.save(buffer, format='JPEG')
    # buffer.seek(0)
    # return buffer
