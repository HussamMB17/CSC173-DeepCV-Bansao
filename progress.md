# Project Development Log

**Student:** Hussam Bansao

## Phase 1: Data Preparation [Week 1]
- [x] Collected dataset for 3 classes: Bacterial leaf blight, Brown spot, Leaf smut.
- [x] **Challenge:** Dataset was extremely small (approx 120 images).
- [x] **Solution:** Implemented robust PyTorch `transforms` including Rotation, Color Jitter, and Flipping to synthetically expand the dataset diversity.

## Phase 2: Architecture Design [Week 2]
- [x] Initial design: Simple 3-Layer CNN (`RiceNet`).
- [x] **Upgrade:** Refactored architecture to `RiceResNet` using **Residual Blocks** and **Batch Normalization** to improve training stability and depth.
- [x] Implemented Global Average Pooling to reduce parameter count and prevent overfitting.

## Phase 3: Training & Optimization [Week 3]
- [x] **Baseline:** Initial training yielded ~75% accuracy.
- [x] **Optimization:** Added `ReduceLROnPlateau` scheduler.
    - *Observation:* Learning rate dropped from 0.001 to 0.000031 during training, allowing the model to fine-tune weights.
- [x] **Result:** Final Test Accuracy improved to **87.50%**.
- [x] **Metric:** Achieved 100% Recall on "Brown Spot" class.

## Phase 4: Explainability & Finalization [Week 4]
- [x] Implemented **Grad-CAM** (Gradient-weighted Class Activation Mapping).
- [x] Verified that the model focuses on disease spots and not background artifacts.
- [x] **Engineering:** Migrated code from Google Colab to a modular VS Code structure.
- [x] Ensured **Reproducibility** by setting random seeds (`42`).