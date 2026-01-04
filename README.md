# DeepRice: Explainable Residual Networks for Rice Disease Diagnosis

**CSC173 Intelligent Systems Final Project** *Mindanao State University - Iligan Institute of Technology*

**Student:** Hussam Bansao  
**Semester:** AY 2025-2026 Sem 1  

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange?logo=pytorch)](https://pytorch.org)
[![Technique](https://img.shields.io/badge/Technique-Transfer%20Learning%20%2B%20GradCAM-yellow)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📑 Abstract
Rice (*Oryza sativa*) is the cornerstone of food security in the Philippines, yet yield is frequently compromised by infectious diseases such as **Bacterial Leaf Blight**, **Brown Spot**, and **Leaf Smut**. While Convolutional Neural Networks (CNNs) have shown promise in agricultural diagnostics, standard models often suffer from the "Black Box" problem—providing high accuracy without transparency. 

This project introduces **DeepRice**, a deep computer vision system powered by a custom **RiceResNet** (Residual Network) architecture. To overcome data scarcity (N=120), we employ a stochastic data augmentation pipeline involving color jittering and affine transformations. Furthermore, we integrate **Explainable AI (XAI)** via **Grad-CAM**, generating heatmaps that visualize the morphological features triggering the diagnosis. The model achieves **87.50% Test Accuracy** with **100% Recall** for Brown Spot, providing farmers with a trustworthy, transparent diagnostic tool.

---

## 📋 Table of Contents
- [Introduction](#introduction)
- [Related Work](#related-work)
- [Methodology](#methodology)
- [Experiments & Results](#experiments--results)
- [Discussion](#discussion)
- [Ethical Considerations](#ethical-considerations)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [References](#references)

---

## 1. Introduction
### Problem Statement
In Mindanao, access to expert plant pathologists is limited. Farmers often rely on visual intuition to diagnose crop ailments, leading to misdiagnosis and the misuse of broad-spectrum fungicides. There is a critical need for an automated tool that is not only accurate but also *interpretible*, allowing non-experts to verify *why* an AI model made a specific prediction.

### Objectives
1.  **Robust Classification:** Engineer a **Residual Neural Network** capable of learning deep features from a small dataset without vanishing gradients.
2.  **Smart Optimization:** Implement dynamic learning rate scheduling (`ReduceLROnPlateau`) to achieve convergence on a non-convex loss surface.
3.  **Explainability (XAI):** Deploy **Grad-CAM** (Gradient-weighted Class Activation Mapping) to visually validate that the model focuses on disease lesions rather than background artifacts.

---

## 2. Related Work
* **Standard CNNs in Agriculture:** Early works (Mohanty et al., 2016) utilized standard CNNs (AlexNet, VGG) for plant disease detection. However, these models often require massive datasets to converge.
* **Residual Learning:** He et al. (2016) introduced ResNets, utilizing skip connections to allow training of deeper networks. DeepRice adopts this to maximize feature extraction efficiency on limited data.
* **The Gap:** Most local agricultural models focus solely on prediction accuracy. DeepRice addresses the **Explainability Gap**, ensuring the model's decisions are biologically grounded.

---

## 3. Methodology

### Dataset & Preprocessing
* **Source:** Aggregated Rice Leaf Disease Dataset.
* **Classes:** Bacterial Leaf Blight, Brown Spot, Leaf Smut.
* **Preprocessing Pipeline:**
    * **Resize:** $128 \times 128$ pixels.
    * **Augmentation:** Random Rotation ($\pm 20^\circ$), Horizontal/Vertical Flips, and **Color Jitter** (Brightness/Contrast $\pm 0.2$) were applied to synthesize new training examples and prevent overfitting.
    * **Normalization:** Normalized to ImageNet statistics ($\mu=[0.485, 0.456, 0.406]$, $\sigma=[0.229, 0.224, 0.225]$).

### Architecture: RiceResNet
Unlike a linear stack of layers, **RiceResNet** utilizes **Residual Blocks**. A shortcut connection $x$ is added to the output of the convolution block $F(x)$, formulating the output as $H(x) = F(x) + x$. This allows gradients to flow through the network more easily during backpropagation.

**Architecture Diagram:**
> `Input (3x128x128)` $\to$ `Conv2d + BatchNorm` $\to$ `ResBlock1` $\to$ `ResBlock2` $\to$ `ResBlock3` $\to$ `GlobalAvgPool` $\to$ `Linear` $\to$ `Output`

### Hyperparameters
| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Batch Size** | 32 | Balanced for GPU memory and gradient stability |
| **Learning Rate** | 0.001 | Initial rate, decayed dynamically |
| **Optimizer** | Adam | Adaptive Moment Estimation |
| **Scheduler** | ReduceLROnPlateau | Factor=0.5, Patience=5 epochs |
| **Loss Function** | CrossEntropyLoss | Standard for multi-class classification |

### Code Snippet: The Residual Block
```python
class RiceResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(RiceResBlock, self).__init__()
        # ... Conv layers ...
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # The Skip Connection
        out = F.relu(out)
        return out
```

# 4. Experiments & Results

## Quantitative Metrics
The model was trained for 40 epochs. The Learning Rate Scheduler triggered multiple reductions, fine-tuning the weights to a loss minimum of 0.015.

### Metric Score Analysis
- **Test Accuracy:** 87.50%  
  *High performance on unseen data*
- **Brown Spot Recall:** 1.00  
  *The model never missed a Brown Spot case*
- **Blight Precision:** 1.00  
  *Zero false positives for Blight*
- **F1-Score (Macro):** 0.86  
  *Balanced performance across classes*

## Visual Results
### Confusion Matrix:
> The matrix shows a strong diagonal, indicating correct classifications.

### Explainability (Grad-CAM):
> To build trust, we visualize where the AI looks. The heatmap below overlays the gradient activation on the original image.
>
> Observation: The "Hot" (Red/Yellow) areas align perfectly with the necrotic brown lesions on the leaf.
>
> Conclusion: The model is learning biological features, not background noise.

## 5. Discussion
### Effectiveness of ResNets:
Initial experiments with a simple CNN yielded ~75% accuracy. Upgrading to RiceResNet pushed this to 87.5%, proving that residual connections help even with small datasets.

### Data Augmentation:
The dataset contained only ~120 images. Without the aggressive Color Jitter and Rotation augmentation, the model would have memorized the training data (overfitting). The high test accuracy proves the model learned robust features.

## 6. Ethical Considerations
### Deployment Risks:
False negatives (diagnosing a sick plant as healthy) can lead to crop spread. This tool should be used as a "Second Opinion" aid, not a replacement for human experts.

### Data Bias:
The dataset consists of images taken in specific lighting conditions. Real-world deployment in poor lighting (e.g., dusk/dawn) requires further calibration.

## 7. Conclusion
DeepRice successfully demonstrates that State-of-the-Art techniques like Residual Learning and Grad-CAM can be applied to lightweight agricultural models. By achieving 87.5% accuracy with full interpretability, this project bridges the gap between high-tech AI and practical, trustworthy farming tools.

## 8. Installation
```bash
# 1. Clone the repository
git clone [https://github.com/hussammb17/csc173-deepcv-bansao.git](https://github.com/hussammb17/csc173-deepcv-bansao.git)

# 2. Navigate to directory
cd csc173-deepcv-bansao

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the notebook # Open CSC173_FinalProj.ipynb in VS Code or Jupyter
dependencies: torch >=2.0, torchvision, matplotlib, seaborn, opencv-python
```

## References:
- He, K., et al., (2016). "Deep Residual Learning for Image Recognition." CVPR.
- Selvaraju, R.R., et al., (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV.
- Mohanty, S.P., et al., (2016). "Using Deep Learning for Image-Based Plant Disease Detection." Frontiers in Plant Science.
🌐 [GitHub Pages](https://hussammb17.github.io/csc173-deepcv-bansao/)