# EnhancedDenseNet201-KOA

# EnhancedDenseNet201-KOA 🦵📊

This repository contains the complete implementation of the **EnhancedDenseNet201+** model for automatic **Kellgren-Lawrence (KL) grading of knee osteoarthritis** from radiographic images.

## 🔬 Project Description

Knee osteoarthritis (KOA) is a progressive joint disease assessed via the Kellgren-Lawrence scale. This project proposes a hybrid deep learning-classical machine learning approach that combines:

- 🧠 DenseNet201 backbone
- 🎯 Multi-scale feature pooling
- 🔍 CBAM attention mechanism
- 📈 Statistical & frequency-based deep feature engineering
- 🧮 Classical classifiers: SVM, Random Forest, Gradient Boosting

## 🗂️ Folder Structure

```bash
EnhancedDenseNet201-KOA/
│
├── dataset/                 # Preprocessed and augmented KOA dataset (not included)
│   ├── Grade_0/
│   ├── Grade_1/
│   ├── Grade_2/
│   ├── Grade_3/
│   └── Grade_4/
│
├── models/                  # Trained models and feature extractors
├── results/                 # CSV files and result logs
├── visualizations/          # GradCAM, t-SNE, ROC curves, confusion matrix
│
├── knee_classifier.py       # Full implementation of EnhancedDenseNet201+ pipeline
├── knee.ipynb               # Colab-compatible notebook
├── requirements.txt         # Dependencies and versions
└── README.md                # Project overview and usage
