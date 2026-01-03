# Project Progress Log

**Student:** Hussam Bansao

## Week 1: Data Preparation & Exploration
* [x] Set up Google Colab environment and mounted Google Drive.
* [x] Loaded the Rice Leaf Disease dataset from `/content/drive/MyDrive/DeepRice_Project/rice_data`.
* [x] Implemented data augmentation: `RandomRotation(15)` and `RandomHorizontalFlip()` to handle the small dataset size (120 images).
* [x] Split data into **80% Training** and **20% Testing**.

## Week 2: Model Architecture (RiceNet)
* [x] Designed `RiceNet`, a custom CNN with 3 convolutional layers.
* [x] Added `Dropout(0.5)` in the fully connected layer to reduce overfitting.
* [x] Implemented the training loop with `CrossEntropyLoss` and `Adam` optimizer.

## Week 3: Training & Evaluation
* [x] Trained the model for 40 epochs.
* [x] **Results:** Achieved a final test accuracy of **75.00%**.
* [x] Saved the trained model weights to `rice_model.pth`.

## Week 4: Explainability
* [x] Implemented **Saliency Maps** to visualize gradients.
* [x] Validated that the model is looking at the disease spots (lesions) rather than the background.