# DeepRice Project Proposal

**Student:** Hussam Bansao  

## 1. Project Title
**DeepRice: Rice Leaf Disease Classification using Residual Neural Networks**

## 2. Problem Statement
Rice farmers in Mindanao face significant yield losses due to diseases like **Bacterial Leaf Blight** and **Brown Spot**. Traditional diagnosis relies on expert visual inspection, which is slow and error-prone. An automated, explainable AI tool is needed to assist in early detection.

## 3. Objectives
* Develop a Deep Learning model to classify 3 common rice diseases.
* Implement a **Residual Network (ResNet)** architecture for improved feature extraction.
* Achieve >80% accuracy on field data.
* Provide **Explainability (Grad-CAM)** to visualize disease localization on the leaf.

## 4. Dataset
* **Classes:** Bacterial leaf blight, Brown spot, Leaf smut.
* **Processing:** Images will be resized to 128x128 and normalized.
* **Augmentation:** Heavy augmentation (Rotation, Color, Flip) will be used to handle data scarcity.

## 5. Methodology
I will build **RiceResNet** using PyTorch.
* **Input:** 128x128 RGB Images.
* **Backbone:** Custom Residual Blocks with Skip Connections to prevent vanishing gradients.
* **Regularization:** Batch Normalization and Dropout.
* **Optimizer:** Adam with Learning Rate Scheduling.

## 6. Deliverables
1.  Full Source Code (`.ipynb` and Python scripts).
2.  Trained Model Weights (`best_ricenet.pth`).
3.  Documentation proving the model's focus (Heatmaps).
4.  Demo Video.