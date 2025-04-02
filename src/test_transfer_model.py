

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
MODEL_PATH = "pneumonia_densenet121.keras"
TEST_DIR = "chest_xray/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# Load model
model = load_model(MODEL_PATH)

# Prepare test generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Predict
y_probs = model.predict(test_generator)
y_true = test_generator.classes
y_pred = (y_probs > 0.5).astype(int).flatten()

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Confusion matrix
conf_mat = confusion_matrix(y_true, y_pred)

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_probs)

# Plotting
plt.figure(figsize=(16, 5))

# Confusion Matrix
plt.subplot(1, 3, 1)
sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

# ROC Curve
plt.subplot(1, 3, 2)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()

# Precision-Recall
plt.subplot(1, 3, 3)
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score

# Already predicted:
# y_probs = model.predict(test_generator)     # shape: (n_samples, 1)
# y_true = test_generator.classes             # shape: (n_samples,)

confidences = np.linspace(0.0, 1.0, 100)
precision_points = []
recall_points = []

for threshold in confidences:
    y_pred_thresh = (y_probs > threshold).astype(int).flatten()
    if np.sum(y_pred_thresh) == 0:
        precision_points.append(1.0)  # Define precision as perfect if no predictions
    else:
        precision_points.append(precision_score(y_true, y_pred_thresh))
    recall_points.append(recall_score(y_true, y_pred_thresh))

# === Plot Precision-Confidence and Recall-Confidence Curves
plt.figure(figsize=(10, 5))

plt.plot(confidences, precision_points, label='Precision', color='blue')
plt.plot(confidences, recall_points, label='Recall', color='green')
plt.axvline(x=0.5, color='red', linestyle='--', label='Default Threshold = 0.5')
plt.xlabel("Confidence Threshold")
plt.ylabel("Score")
plt.title("Precision & Recall vs Confidence Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



