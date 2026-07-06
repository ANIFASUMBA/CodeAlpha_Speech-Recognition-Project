# Speech Emotion Recognition (SER) Using 1D CNN

An end-to-end Machine Learning pipeline that classifies human emotions from raw audio waveforms using deep learning. This project extracts acoustic features from vocal signals and processes them through a custom Convolutional Neural Network (CNN) to achieve high-accuracy emotion classification.

## 🚀 Project Overview & Pipeline
Raw audio signals are too chaotic for neural networks to process directly. This project implements a structured three-phase machine learning workflow:

1. **Feature Extraction:** Uses `librosa` to compute **Mel-Frequency Cepstral Coefficients (MFCCs)** from raw `.wav` files, compressing audio signals into mathematical vectors that mimic human auditory perception
2. **Data Loading:** Traverses the RAVDESS dataset, maps file naming conventions to structural targets (`happy`, `sad`, `angry`, `neutral`, etc.), and encodes them into one-hot target arrays.
3. **Deep Learning Classification:** Feeds the formatted vectors into a **1D Convolutional Neural Network (CNN)** built with TensorFlow/Keras to capture localized time-series frequency pattern shifts.

## 📊 Performance Results
* **Training Epochs:** 50
* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Final Test Accuracy:** **92.36%** 🌟

## 🛠️ Project Structure
```text
Speech_Emotion_Project/
├── emotion_recognition.py   # Main pipeline (Extraction, Architecture, Training)
├── predict_live.py          # Presentation Live Demo Inference Script
├── speech_emotion_model.h5  # Pre-trained CNN model weights (saved brain)
└── .gitignore               # Excludes virtual environments and raw datasets
