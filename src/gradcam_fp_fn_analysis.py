import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve
import tensorflow as tf

# === Config
MODEL_PATH = "pneumonia_densenet121.keras"
TEST_DIR = "chest_xray/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
FP_DIR = "misclassified_gradcam_fp"
FN_DIR = "misclassified_gradcam_fn"
os.makedirs(FP_DIR, exist_ok=True)
os.makedirs(FN_DIR, exist_ok=True)

# === Load model and define Grad-CAM function
model = load_model(MODEL_PATH)
last_conv_layer_name = "conv5_block16_2_conv"
grad_model = Model(
    inputs=model.inputs,
    outputs=[model.get_layer(last_conv_layer_name).output, model.output]
)

def make_gradcam_heatmap(image):
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(image, heatmap, alpha=0.4):
    image = np.uint8(255 * image)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

# === Load test data
datagen = ImageDataGenerator(rescale=1./255)
test_gen = datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False
)
y_true = test_gen.classes
filepaths = test_gen.filepaths
y_probs = model.predict(test_gen)

# === Compute Youden’s threshold
fpr, tpr, thresholds = roc_curve(y_true, y_probs)
best_thresh = thresholds[np.argmax(tpr - fpr)]
y_pred = (y_probs > best_thresh).astype(int).flatten()

# === Identify FP and FN indices
fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]

# === Save Grad-CAM images
def process_and_save(indices, true_cls, pred_cls, folder):
    for idx in indices[:10]:
        path = filepaths[idx]
        filename = os.path.basename(path)
        confidence = y_probs[idx][0]
        
        img = load_img(path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        input_tensor = np.expand_dims(img_array, axis=0)
        
        heatmap = make_gradcam_heatmap(input_tensor)
        overlay = overlay_heatmap(img_array, heatmap)
        
        out_name = f"{true_cls}_pred_{pred_cls}_{confidence:.2f}_{filename}"
        out_path = os.path.join(folder, out_name)
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

process_and_save(fp_indices, "NORMAL", "PNEUMONIA", FP_DIR)
process_and_save(fn_indices, "PNEUMONIA", "NORMAL", FN_DIR)

# === Visualization
def show_gradcam_grid(image_dir, title, max_images=10):
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg"))])
    image_paths = [os.path.join(image_dir, f) for f in image_files[:max_images]]

    if not image_paths:
        print(f"❌ No images found in {image_dir}")
        return

    cols = 5
    rows = (len(image_paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < len(image_paths):
            path = image_paths[i]
            parts = os.path.basename(path).split("_")
            true_label = parts[0] if len(parts) > 0 else "?"
            pred_label = parts[2] if len(parts) > 2 else "?"
            confidence = parts[3] if len(parts) > 3 else "?"
            
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)
            ax.set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence}")

    plt.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.show()

# === Show results
show_gradcam_grid(FP_DIR, "❌ False Positives (NORMAL → PNEUMONIA)")
show_gradcam_grid(FN_DIR, "🚨 False Negatives (PNEUMONIA → NORMAL)")
