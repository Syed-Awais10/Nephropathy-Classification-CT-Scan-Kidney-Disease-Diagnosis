# CT-Scan Kidney Disease Classification (Nephropathy Classification)

A deep learning-based medical imaging system designed to classify kidney CT scans into four clinically significant categories: Normal, Cyst, Tumor, and Stone.
Built with TensorFlow, utilizing transfer learning (VGG16) and a custom CNN model, and deployed as an interactive Streamlit web application for real-time clinical use.

## 👤 My Contribution

This project was developed as a **team-based academic project**.

My role focused on **research and documentation**, including:
- Researching and evaluating suitable deep learning models for kidney disease classification using CT scans
- Comparing CNN architectures and model performance based on accuracy and feasibility
- Contributing to project documentation, explanations, and reporting
- Assisting the team in selecting appropriate approaches for implementation

This repository is forked to showcase my contribution and learning as part of the team.

<img width="1506" height="673" alt="image" src="https://github.com/user-attachments/assets/c3427a92-2dc3-45e3-bd78-314c0009a69e" />
<img width="1016" height="700" alt="image" src="https://github.com/user-attachments/assets/88894796-3ea6-4e23-92a0-c8887c4c379a" />
<img width="1000" height="563" alt="image" src="https://github.com/user-attachments/assets/fa4432c5-bcb4-4be5-a725-449705658067" />



## Features
 
Multi-Class Classification: Distinguishes between four kidney conditions using 2D CT scan slices.
Deep Learning Models:
Custom CNN: Lightweight architecture achieving 81.2% test accuracy.
VGG16 Transfer Learning: Pre-trained on ImageNet, fine-tuned for medical imaging.
Real-Time Web App: Interactive Streamlit interface for instant predictions and visual feedback.
Data Augmentation: Comprehensive augmentation pipeline (rotation, flipping, zoom, brightness) to combat overfitting.
=Model Interpretability: Confidence scores and prediction probabilities displayed for clinical transparency.



## Model Performance

| Model        | Training Accuracy | Test Accuracy | Note                          |
|--------------|-------------------|---------------|-------------------------------|
| VGG16     | 99.1%          | N/A       | High training accuracy, used with augmentation |
| Custom CNN| ~95%           | 81.2%     | Generalizes well to unseen data |

> Note: The custom CNN's 81.2% test accuracy represents a robust, deployable performance on a limited medical imaging dataset.



## 🛠️ Tech Stack

Deep Learning: TensorFlow 2.x, Keras
Computer Vision: OpenCV, Pillow
Web Framework: Streamlit
Data Processing: Pandas, NumPy, Scikit-learn
Visualization: Matplotlib, Seaborn
Environment: Python 3.8+

Original repository: [<link>](https://github.com/MuhammadAmanQazi2/Nephropathy-Classification-CT-Scan-Kidney-Disease-Diagnosis)
