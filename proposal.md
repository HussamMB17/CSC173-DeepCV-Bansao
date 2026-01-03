# CSC173 Deep Computer Vision Project Proposal

**Student:** Hussam Bansao
**Date:** December 11, 2025

## 1. Project Title
**DeepRice: Rice Leaf Disease Classification using Convolutional Neural Networks**

## 2. Problem Statement
Rice is a staple food for over half the world's population, particularly in Mindanao. However, diseases like **Bacterial Leaf Blight**, **Brown Spot**, and **Leaf Smut** can devastate crops if not detected early. Farmers often lack access to expert plant pathologists, leading to delayed or incorrect treatment.

## 3. Project Goal
The goal is to develop a Deep Computer Vision model that can automatically classify rice leaf diseases from images. This tool aims to assist farmers in early diagnosis, potentially running on mobile devices in the future.

## 4. Dataset
* **Source:** Rice Leaf Disease Image Dataset (likely from Kaggle or UCI).
* **Classes:** 3 classes (Bacterial leaf blight, Brown spot, Leaf smut).
* **Size:** 120 images total (based on your notebook output).
* **Preprocessing:** Resize to 128x128, Random Rotation, Horizontal Flip, Normalization.

## 5. Methodology (Architecture Sketch)
I will build a custom Convolutional Neural Network (CNN) called **RiceNet** using PyTorch.
* **Input:** 128x128 RGB images.
* **Backbone:** 3 Convolutional blocks with ReLU activation and Max Pooling.
    * Conv1: 32 filters
    * Conv2: 64 filters
    * Conv3: 128 filters
* **Classifier:** Fully connected layers (512 units -> 3 classes) with Dropout (0.5) to prevent overfitting.
* **Loss Function:** CrossEntropyLoss.
* **Optimizer:** Adam (lr=0.001).

## 6. Expected Results
* A classification accuracy of >70%.
* Visual explanation of model decisions using Saliency Maps to show which part of the leaf the AI focuses on.