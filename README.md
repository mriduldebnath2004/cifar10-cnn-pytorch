# CIFAR-10 Image Classification with PyTorch

## Overview
This project implements a **baseline Convolutional Neural Network** using PyTorch to classify images from the **CIFAR-10 dataset** into 10 categories.  
The project demonstrates a complete workflow: data preprocessing, model definition, training, and evaluation.

---

## Dataset
- **CIFAR-10**: 60,000 color images (32×32 pixels) across 10 classes:
  - `plane`, `car`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`
- **Train/Validation Split**: 40,000 training / 10,000 validation
- **Normalization**: Images converted to tensors and normalized using dataset mean and standard deviation:
  - Mean: `[0.4914, 0.4822, 0.4465]`
  - Std: `[0.2470, 0.2435, 0.2616]`

---

## Data Preprocessing
- Converted images to PyTorch tensors
- Normalized each channel (R, G, B) using computed mean and standard deviation
- Created **DataLoaders** for training and validation sets with batch size 128

---

## Model Architecture
The CNN consists of:

1. **Convolutional Layer 1**: 3 input channels → 32 output channels, 3×3 kernel, padding=1  
2. **ReLU Activation**  
3. **MaxPooling**: 2×2 kernel, stride=2  
4. **Convolutional Layer 2**: 32 → 64 channels, 3×3 kernel, padding=1  
5. **ReLU Activation**  
6. **MaxPooling**: 2×2 kernel, stride=2  
7. **Flatten Layer**: Converts 2D feature maps to 1D vector  
8. **Fully Connected Layer 1**: 64×8×8 → 128 neurons  
9. **ReLU Activation**  
10. **Fully Connected Layer 2**: 128 → 10 (output logits for each class)

---

## Training
- **Loss Function**: CrossEntropyLoss  
- **Optimizer**: Adam (learning rate = 0.001)  
- **Epochs**: 20  
- **Batch Size**: 128  
- **Device**: GPU if available

---

## Evaluation
- **Metric**: Accuracy on validation set  
- **Validation Accuracy**: **69.14%**

> Accuracy is sufficient for this baseline model as CIFAR-10 is a balanced dataset.  
> Further improvements can be added using techniques like Batch Normalization, Data Augmentation, or deeper CNN architectures.

---

## Results
| Model | Validation Accuracy |
|-------|-------------------|
| Baseline CNN | 69.14% |

---

## Future Work
- Add **Batch Normalization** to improve convergence and stability  
- Apply **Data Augmentation** (random crop, horizontal flip)  
- Experiment with **learning rate scheduling**  
- Explore **deeper CNN architectures** like ResNet for higher accuracy  

---

## Conclusion
This project demonstrates the end-to-end process of building a CNN in PyTorch for image classification, including:
- Data preprocessing
- Model design
- Training
- Evaluation

It serves as a strong baseline for CIFAR-10 classification and a foundation for further experimentation.
