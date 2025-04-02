# inference_predict.py

import os
import numpy as np
import matplotlib.pyplot as plt
import random
import gc
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paths and config
MODEL_PATH = "pneumonia_best_model.keras"
TEST_DIR = "chest_xray/test"
IMG_SIZE = (150, 150)
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# Load the trained CNN model
model = load_model(MODEL_PATH)

def load_random_test_images(test_dir, num_images=4):
    images = []
    labels = []
    for label in CLASS_NAMES:
        img_dir = os.path.join(test_dir, label)
        samples = random.sample(os.listdir(img_dir), num_images // 2)
        for img_name in samples:
            img_path = os.path.join(img_dir, img_name)
            img = load_img(img_path, target_size=IMG_SIZE)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(label)
    return np.array(images), labels

# Predict
images, true_labels = load_random_test_images(TEST_DIR)
preds = model.predict(images)
predicted_labels = [CLASS_NAMES[int(p > 0.5)] for p in preds]

# Plot predictions
plt.figure(figsize=(12, 6))
for i in range(len(images)):
    plt.subplot(1, len(images), i + 1)
    plt.imshow(images[i])
    plt.title(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Clear memory
del images, preds
gc.collect()
