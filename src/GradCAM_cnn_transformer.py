import os, random, numpy as np, matplotlib.pyplot as plt, cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# === Paths & Config ===
model_path = "cnn_transformer_best_v2.keras"
test_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray/test"
classes = ["NORMAL", "PNEUMONIA"]
img_size = (224, 224)

# === Load model
model = load_model(model_path)

# === Find last Conv2D layer dynamically
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            print(f"✅ Using last Conv2D layer: {layer.name}")
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")

# === Grad-CAM Generator
def make_gradcam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.input],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

# === Overlay function
def overlay_heatmap(heatmap, image, alpha=0.6):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(jet, alpha, image, 1 - alpha, 0)
    return overlayed

# === Image preprocessing
def preprocess_image(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)
    return img_array, input_tensor

# === Visualize Grad-CAMs
last_conv_name = find_last_conv_layer(model)

plt.figure(figsize=(10, 10))
for i, label in enumerate(classes):
    img_list = os.listdir(os.path.join(test_dir, label))
    for j in range(2):  # Two images per class
        img_path = os.path.join(test_dir, label, random.choice(img_list))
        img_array, input_tensor = preprocess_image(img_path)

        # Prediction
        pred_prob = model.predict(input_tensor)[0][0]
        pred_class = "PNEUMONIA" if pred_prob > 0.5 else "NORMAL"

        # Generate heatmap
        heatmap = make_gradcam_heatmap(model, input_tensor, last_conv_name)
        overlay = overlay_heatmap(heatmap, np.uint8(img_array * 255))

        # Plot
        plt.subplot(2, 2, i * 2 + j + 1)
        plt.imshow(overlay)
        plt.title(f"True: {label}\nPred: {pred_class}\nConf: {pred_prob:.2f}")
        plt.axis('off')

plt.suptitle("🔥 Grad-CAM Visualization (CNN + Transformer)", fontsize=16)
plt.tight_layout()
plt.show()
