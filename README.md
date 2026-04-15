# Automated Bone Age Estimation Using Deep Learning

A deep learning-based system for automated pediatric bone age estimation using hand X-ray images with Xception architecture and transfer learning.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Performance Metrics](#performance-metrics)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Team](#team)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

Bone age assessment is a critical diagnostic tool for evaluating skeletal maturity and diagnosing growth disorders in children. Traditional manual methods are time-consuming and subject to inter-observer variability (±6-12 months). This project implements an automated bone age prediction system using deep learning to achieve clinically acceptable accuracy.

**Key Achievements:**
- 📊 **R² Score: 0.9169** (target: ≥0.90)
- 📉 **MAE: 9.04 months** (target: ≤12 months)
- ⚖️ **Low Gender Bias**: <2 months MAE difference between genders
- 🎯 **91.54% Classification Accuracy** for developmental stages

## ✨ Key Features

- **Transfer Learning**: Leverages Xception architecture pre-trained on ImageNet
- **Gender-Aware Prediction**: Incorporates patient sex as input for improved accuracy
- **Model Explainability**: Grad-CAM visualization shows anatomically-correct attention on carpal bones and growth plates
- **Comprehensive Analysis**: 
  - Regression performance metrics
  - Gender bias analysis
  - Developmental stage classification (Child/Adolescent/Adult)
  - Error pattern analysis
- **Mixed Precision Training**: Optimized for RTX 3060 12GB GPU
- **Robust Preprocessing**: Architecture-specific preprocessing with data augmentation

## 📊 Performance Metrics

### Regression Performance
| Metric | Value |
|--------|-------|
| R² Score | 0.9169 |
| MAE (Mean Absolute Error) | 9.04 months |
| RMSE | 11.75 months |
| Within ±12 months | 91.5% |

### Classification Performance
| Metric | Value |
|--------|-------|
| Overall Accuracy | 91.54% |
| Quadratic Weighted Kappa (QWK) | 0.8248 |
| Child (0-10y) Recall | 95% |
| Adolescent (10-18y) Recall | 91% |

### Gender Bias Analysis
- MAE Difference: <2 months between male/female samples
- R² Difference: <0.01
- **Fair predictions across both genders** ✓

## 📁 Dataset

**RSNA Pediatric Bone Age Challenge Dataset**
- **Total Images**: 12,611 hand X-ray images
- **Age Range**: 1-228 months (0-19 years)
- **Modality**: Left hand radiographs
- **Metadata**: Ground truth bone age + patient sex
- **Split**: 70% training / 15% validation / 15% test (stratified by age)

## 🏗️ Model Architecture

### Xception Base Model
- **Architecture**: 36 convolutional layers with depthwise separable convolutions
- **Pre-training**: ImageNet weights
- **Innovation**: Efficient feature extraction through extreme Inception modules
- **Training Strategy**: All layers trainable from start (not frozen)

### Regression Head
```
Xception Base (256×256×3 input)
    ↓
GlobalMaxPooling2D
    ↓
Flatten
    ↓
Dense(10, ReLU)
    ↓
Dense(1, linear) → Bone Age (months)
```

### Training Configuration
- **Input Size**: 256×256 pixels
- **Preprocessing**: `xception.preprocess_input()` ([-1, 1] scaling)
- **Batch Size**: 4 (memory constraint)
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Early Stopping**: Patience=7 epochs
- **Epochs**: 50 max (converged at epoch 18)
- **Mixed Precision**: FP16 for memory efficiency

## 🛠️ Installation

### Prerequisites
```bash
Python 3.8+
CUDA 11.2+ (for GPU support)
```

### Dependencies
```bash
pip install tensorflow>=2.8.0
pip install numpy pandas matplotlib seaborn
pip install scikit-learn opencv-python
pip install jupyter notebook
```

### Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/bone-age-estimation.git
cd bone-age-estimation
```

## 🚀 Usage

### Training the Model
```python
# Open the main notebook
jupyter notebook Bone_Age_Prediction_Xception_FINAL.ipynb

# Or run directly
python train_model.py  # If you create a standalone script
```

### Making Predictions
```python
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
import cv2

# Load model
model = load_model('best_model.h5')

# Load and preprocess image
img = cv2.imread('xray_image.png')
img = cv2.resize(img, (256, 256))
img = preprocess_input(img)
img = np.expand_dims(img, axis=0)

# Predict (with gender: 0=female, 1=male)
gender = 1  # male
prediction = model.predict([img, np.array([[gender]])])
print(f"Predicted Bone Age: {prediction[0][0]:.2f} months")


## 📈 Results

### Approach Comparison
We experimented with multiple approaches before achieving optimal results:

| Approach | R² Score | MAE (months) | Notes |
|----------|----------|--------------|-------|
| EfficientNet-B4 (frozen) | 0.70 | 33 | Insufficient adaptation |
| Xception + CLAHE | -0.01 | N/A | Preprocessing broke transfer learning |
| Xception + ROI crop | 0.68 | 28 | Lost contextual information |
| **Our Final Model** | **0.9169** | **9.04** | **All layers trainable** ✓ |

### Key Insights
1. **Architecture-specific preprocessing is critical** - Using `xception.preprocess_input()` was essential
2. **Training all layers from start** outperformed gradual unfreezing approach
3. **Contrast enhancement (CLAHE) breaks transfer learning** from ImageNet weights
4. **Simple regression heads** (10 units) work well with strong base models

### Visualizations

**Predicted vs Actual Age**
- Strong linear correlation (R²=0.9169)
- Most predictions within ±12 months clinical threshold

**Residual Plot**
- Errors centered around zero
- Slight heteroscedasticity at age extremes

**Grad-CAM Attention**
- Model correctly focuses on carpal bones and epiphyseal growth plates
- Validates medically-correct feature learning

**Gender Bias Analysis**
- Minimal performance difference between male/female samples
- Fair and unbiased predictions



**Guided By:**  
Dr. Umarani Jayaraman, Assistant Professor

## 🎓 Acknowledgments

- **RSNA** for providing the Pediatric Bone Age Challenge dataset
- **Keras Applications** for pre-trained Xception weights
- **TensorFlow** team for the deep learning framework
- Course instructors and peers for valuable feedback


## 📚 References

1. Halabi, S. S., et al. (2019). "The RSNA Pediatric Bone Age Machine Learning Challenge." *Radiology*.
2. Chollet, F. (2017). "Xception: Deep Learning with Depthwise Separable Convolutions." *CVPR*.
3. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." *ICCV*.


---

**⭐ If you find this project useful, please consider giving it a star!**
