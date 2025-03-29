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
python train.py
```


