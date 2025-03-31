# gradcam_densenet.py

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Model

# === Config
MODEL_PATH = "pneumonia_densenet121.keras"
TEST_DIR = "chest_xray/test"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
YOUden_THRESHOLD = 0.7653

# Load model
model = load_model(MODEL_PATH)

# Identify last conv layer in DenseNet121
last_conv_layer_name = "conv5_block16_2_conv"

# Build grad model
grad_model = Model(
    inputs=model.inputs,
    outputs=[model.get_layer(last_conv_layer_name).output, model.output]
)

# Grad-CAM function
def make_gradcam_heatmap(image):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        class_idx = 0  # for binary classification
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(image, heatmap, alpha=0.4):
    image = np.uint8(255 * image)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

def load_random_image(label):
    img_dir = os.path.join(TEST_DIR, label)
    img_name = random.choice(os.listdir(img_dir))
    path = os.path.join(img_dir, img_name)
    img = load_img(path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    return img_array, np.expand_dims(img_array, axis=0), img_name

# === Visualize 2 images from each class
plt.figure(figsize=(16, 6))

for i, label in enumerate(CLASS_NAMES):
    for j in range(2):
        raw_img, input_img, img_name = load_random_image(label)
        heatmap = make_gradcam_heatmap(input_img)
        overlay = display_gradcam(raw_img, heatmap)
        pred_prob = model.predict(input_img)[0][0]
        pred_label = CLASS_NAMES[int(pred_prob > YOUden_THRESHOLD)]

        plt.subplot(2, 4, i * 4 + j + 1)
        plt.imshow(overlay)
        plt.title(f"True: {label}\nPred: {pred_label}\nConf: {pred_prob:.2f}")
        plt.axis("off")

plt.suptitle("🔍 Grad-CAM Visualization (DenseNet121)", fontsize=16)
plt.tight_layout()
plt.show()
