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


