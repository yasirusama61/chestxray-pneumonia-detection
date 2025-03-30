# 🫁 Pneumonia Chest X-Ray Classifier

This project implements a **Convolutional Neural Network (CNN)** to detect **pneumonia** from **chest X-ray images**. Built using TensorFlow/Keras and trained on the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset.

---

## 📌 Overview

Pneumonia is a serious lung infection that can be diagnosed using radiographic imaging like chest X-rays. This project aims to classify X-ray scans into:

- ✅ **Normal**
- ❌ **Pneumonia**

We use data augmentation, a clean CNN pipeline, and visualization techniques to build an interpretable and efficient model.

---

## 📁 Dataset

The dataset is structured into `train`, `val`, and `test` folders:



📦 Total images: ~5,800  
📊 Format: JPEG grayscale chest X-ray images

---

## 🖼️ Sample Images

<p align="center">
  <img src="assets/sample_train_grid.png" alt="Sample TRAIN Images" width="90%">
</p>

Above are sample chest X-ray images from the training dataset:
- The **top row** shows healthy (Normal) lungs.
- The **bottom row** shows lungs diagnosed with **Pneumonia**.

These images highlight the visual features used by the model to learn pathology patterns.

## 📊 Class Distribution

<p align="center">
  <img src="assets/class_distribution.png" alt="Class Distribution Across Splits" width="95%">
</p>

This bar chart illustrates the number of X-ray images per class (NORMAL vs. PNEUMONIA) across different dataset splits:

- **Training Set** shows moderate class imbalance (more pneumonia cases).
- **Validation Set** is balanced.
- **Test Set** also has more pneumonia cases.

Understanding class distribution is crucial for handling bias and ensuring balanced evaluation.


## 🧠 Model Architecture

A simple **CNN** with:

- 3 × Conv2D layers with ReLU
- MaxPooling after each conv
- Dense + Dropout
- Sigmoid output for binary classification

Want more power? You can upgrade to **EfficientNet**, **ResNet**, or **Transfer Learning**.

---

## 🚀 Training

```bash
python train_cnn.py
```
## 📈 Model Evaluation

We evaluated our trained CNN model on the test set from the **Chest X-Ray Pneumonia** dataset. The results demonstrate strong performance in detecting pneumonia from chest X-rays.

### 🧪 Performance Metrics

- **AUC (ROC):** 0.96
- **Precision, Recall, and F1-score** are also high, as shown below.

### 📊 Confusion Matrix, ROC Curve, and Precision-Recall Curve

The following plot summarizes model performance visually:

![Evaluation Metrics](assets/evaluation_metrics.png) <!-- Replace with actual path if hosted -->

- The model has high **true positive rate** and low **false positive rate**.
- Precision and recall trade-off is strong across thresholds.

### 🩻 Example Predictions on Test Images

Below are a few examples from the test set:

![Predictions](assets/sample_predictions.png) <!-- Replace with actual path if hosted -->

The predictions match the ground truth, showing the model's ability to distinguish between **NORMAL** and **PNEUMONIA** chest X-rays effectively.

---

## 📈 Model Comparison: Custom CNN vs DenseNet121 (Transfer Learning)

We trained and evaluated two models on the Chest X-ray Pneumonia dataset:

- ✅ A custom CNN trained from scratch  
- ✅ A transfer learning model using **DenseNet121** pretrained on ImageNet

---

### 🔬 Evaluation Metrics

| Metric              | Custom CNN       | DenseNet121 (Transfer Learning) |
|---------------------|------------------|---------------------------------|
| **Test Accuracy**   | ~96%             | **87%**                         |
| **ROC AUC**         | 0.9600           | **0.9481**                      |
| **F1-score (Normal)** | 0.85           | **0.81**                        |
| **F1-score (Pneumonia)** | 0.92       | **0.90**                        |
| **Recall (Pneumonia)**  | 0.96        | **0.94**                        |
| **Model Size**      | Small            | Larger                          |
| **Grad-CAM Support**| ✅ Enabled       | ✅ Planned                      |

---

### 📊 DenseNet121 – Evaluation Results

![DenseNet121 Evaluation](assets/densenet_eval.png)

![Classification Report](assets/densenet_classification_report.png)

- **Confusion Matrix** and ROC curve show reliable classification performance.
- **Classification Report** shows strong recall for PNEUMONIA class (0.94), with slightly lower recall on NORMAL class (0.75).
- F1-score balances out at **0.90 (Pneumonia)** and **0.81 (Normal)**.

---

### 🧠 Insight

Despite being a powerful pretrained architecture, DenseNet121 slightly underperformed the custom CNN on this dataset. This may be due to:
- The relatively small dataset size
- The model being frozen during training (not fine-tuned)

---

### 🛠️ Next Steps

- ✅ Grad-CAM visualization for DenseNet121
- 🔁 Fine-tune top layers of DenseNet for better generalization
- 📦 Package the best model into a web demo or API

### 🎯 Confidence Threshold Analysis

To better understand the model's behavior across different decision thresholds, we plotted **Precision and Recall vs Confidence Threshold**.

This helps identify optimal thresholds based on the application's needs — for example, prioritizing **recall** in a medical setting to minimize false negatives.

![Confidence Curve](assets/densenet_confidence_curve.png)

- As expected, **precision increases** and **recall decreases** as confidence threshold rises.
- The default threshold of 0.5 (dashed line) represents a good balance, but the curve allows threshold tuning for specific use cases.
