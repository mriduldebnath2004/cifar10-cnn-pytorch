# CIFAR-10 Image Classification with a Custom ResNet (PyTorch)

## Overview
This project implements and trains a **custom ResNet-style convolutional neural network** from scratch using **PyTorch** to classify images from the **CIFAR-10 dataset**.

The goal of this project was not only to achieve strong classification accuracy, but to demonstrate a **complete and well-structured deep learning workflow**, including:
- Proper data preprocessing and augmentation
- Residual network design
- Stable training with regularization and learning-rate scheduling
- Thorough evaluation and error analysis

This project builds upon a simple baseline CNN and improves performance through modern deep learning techniques.

---

## Dataset
**CIFAR-10** consists of **60,000 color images (32×32)** across 10 classes:

`plane, car, bird, cat, deer, dog, frog, horse, ship, truck`

- **Training set:** 50,000 images  
- **Test set:** 10,000 images  
- **Train / Validation split:** 40,000 / 10,000  

### Normalization
Dataset-wide statistics were computed directly from the training data:
- **Mean:** `[0.4914, 0.4822, 0.4465]`
- **Std:** `[0.2470, 0.2435, 0.2616]`

---

## Data Preprocessing & Augmentation
**Training data augmentation:**
- Random horizontal flip
- Random crop (32×32 with padding=4)
- Color jitter (brightness, contrast, saturation)

**Validation & test data:**
- Tensor conversion
- Normalization only (no augmentation)

This ensures strong generalization while avoiding data leakage.

---

## Model Architecture
The model is a **custom ResNet-18–inspired architecture**, adapted for small 32×32 images.

### Key Components
- Residual blocks with:
  - 3×3 convolutions
  - Batch Normalization
  - ReLU activation
  - Identity skip connections
- Downsampling via stride and 1×1 convolutions when needed
- Adaptive average pooling
- Final fully connected classification layer

### Architecture Summary
- Initial convolution: 3 → 16 channels  
- Residual stages:
  - Stage 1: 16 channels × 2 blocks
  - Stage 2: 32 channels × 2 blocks (stride=2)
  - Stage 3: 64 channels × 2 blocks (stride=2)
- Adaptive average pooling
- Fully connected layer (64 → 10)

This design allows deeper networks to train effectively by mitigating vanishing gradients.

---

## Training Setup
- **Loss function:** CrossEntropyLoss  
- **Optimizer:** AdamW  
- **Learning rate:** 0.0005  
- **Weight decay:** 0.01  
- **LR scheduler:** StepLR (gamma=0.5 every 10 epochs)  
- **Batch size:** 128  
- **Max epochs:** 150  
- **Early stopping:** Patience = 15 epochs  
- **Device:** GPU if available  

The model with the lowest validation loss is saved automatically.

---

## Results

### Accuracy
| Metric | Accuracy |
|------|---------|
| Training Accuracy | **92.49%** |
| Test Accuracy | **87.06%** |

### Per-Class Accuracy
| Class | Accuracy |
|------|---------|
| Plane | 90.3% |
| Car | 92.3% |
| Bird | 82.3% |
| Cat | 76.0% |
| Deer | 89.2% |
| Dog | 81.0% |
| Frog | 86.2% |
| Horse | 90.3% |
| Ship | 90.1% |
| Truck | 92.9% |

Cats and dogs show lower accuracy due to high visual similarity, while vehicles perform strongly due to clearer structural features.

---

## Loss Curves
Training and validation loss curves show smooth convergence and no severe overfitting, indicating effective regularization and data augmentation.

---

## Baseline Comparison
| Model | Test Accuracy |
|------|--------------|
| Baseline CNN | ~69% |
| Custom ResNet (this project) | **87.06%** |

This represents an **~18% absolute improvement** over the baseline model.

---

## Key Takeaways
- Residual connections significantly improve training stability for deeper networks
- Data augmentation is critical for CIFAR-10 generalization
- Learning rate scheduling and early stopping prevent overfitting
- Detailed evaluation (confusion matrix, per-class accuracy) provides insight beyond a single metric

---

## Future Improvements
- Cosine annealing or OneCycleLR scheduling  
- MixUp / CutMix augmentation  
- Label smoothing  
- Deeper residual architectures  
- Error visualization on misclassified samples  

---

## Conclusion
This project demonstrates an end-to-end deep learning pipeline using PyTorch, progressing from a simple baseline CNN to a carefully designed residual network. It highlights both **theoretical understanding** and **practical implementation skills**, making it suitable as a portfolio project or foundation for further experimentation.

---

## Technologies Used
- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- scikit-learn
