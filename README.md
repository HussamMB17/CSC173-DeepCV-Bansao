# DeepRice: Rice Leaf Disease Classification

**CSC173 Intelligent Systems Final Project** *Mindanao State University - Iligan Institute of Technology*

**Student:** Hussam Bansao  
**Semester:** AY 2025-2026 Sem 1  

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org)

---

## Abstract
Rice diseases such as Bacterial Leaf Blight, Brown Spot, and Leaf Smut significantly reduce agricultural yield in the Philippines. Identifying these diseases early is difficult for non-experts. This project, **DeepRice**, implements a deep computer vision system using a custom Convolutional Neural Network (RiceNet) to classify these diseases automatically. Trained on a dataset of 120 field images, the model utilizes data augmentation techniques to achieve a classification accuracy of **75%**. Furthermore, this project implements interpretability techniques using Saliency Maps, proving that the model correctly identifies disease lesions on the leaves rather than background noise.

## Table of Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
- [Experiments & Results](#experiments--results)
- [Demo](#demo)
- [Conclusion](#conclusion)
- [Installation](#installation)

---

## Introduction
Rice farming is a critical economic activity in Mindanao. Traditional disease diagnosis relies on visual inspection by experts, who are not always available. This project aims to bridge that gap by providing an automated classification tool.

## Methodology
The solution uses **PyTorch** to implement a custom CNN named `RiceNet`.

### 1. Dataset
The dataset consists of 120 images divided into three classes:
1.  Bacterial leaf blight
2.  Brown spot
3.  Leaf smut

Preprocessing includes resizing to 128x128 pixels and normalizing RGB channels to (0.5, 0.5, 0.5).

### 2. Network Architecture
`RiceNet` consists of three convolutional blocks:
* **Conv1**: 3 input channels -> 32 filters (3x3 kernel) + ReLU + MaxPool
* **Conv2**: 32 -> 64 filters + ReLU + MaxPool
* **Conv3**: 64 -> 128 filters + ReLU + MaxPool
* **FC Layers**: Flatten -> 512 neurons -> Dropout(0.5) -> 3 output classes.

## Experiments & Results
The model was trained for **40 epochs** using the Adam optimizer (lr=0.001).

* **Final Loss:** ~0.17
* **Test Accuracy:** **75.00%**

### Interpretability (Saliency Map)
To ensure trust, I implemented Saliency Maps. The visualization below shows the original leaf (left) and the heatmap (right), indicating that the neural network focuses on the discolored lesions to make its prediction.

*(Note: Insert a screenshot of your notebook's last cell output here)*

## Demo
A video demonstration of the project can be found in the `demo/` folder:
`demo/CSC173_Bansao_Final.mp4`

## Conclusion
DeepRice successfully demonstrates that a lightweight CNN can classify rice diseases with reasonable accuracy (75%) even on a small dataset. Future work would involve expanding the dataset and deploying the model to a mobile app for field testing.

## Installation
1. Clone the repo:
   ```bash
   git clone [https://github.com/yourusername/csc173-deeprice.git](https://github.com/yourusername/csc173-deeprice.git)