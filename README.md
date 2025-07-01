# 😃 Face Detection and Recognition using CNN + VGGFace

## 📘 Course Assignment  
**Applied Machine Learning**  
Author: Angelica Lozano  

---

## 📌 Overview  
This project builds a facial recognition model using a pre-trained **VGGFace (ResNet50)** convolutional neural network (CNN) architecture to identify three known individuals:
- Alec Bohm  
- Bryson Stott  
- Brandon Marsh  

The pipeline includes image preprocessing, face detection using **MTCNN**, transfer learning using **VGGFace**, and model evaluation using prediction confidence.

---

## 🧪 Methodology

### 🔹 Image Preprocessing
- Face images loaded and resized to 224×224
- Preprocessed using `vgg_preprocess`
- Labels encoded manually (Bohm=0, Stott=1, Marsh=2)

### 🔹 Model Architecture
- **Base model**: VGGFace with ResNet50 backbone (`include_top=False`)
- Unfroze last 10 layers for fine-tuning
- Added custom dense layers with dropout
- Optimizer: Adam  
- Loss: Sparse Categorical Crossentropy

### 🔹 Training & Evaluation
- Used `ImageDataGenerator` for augmentation
- Applied early stopping on validation accuracy
- Final model saved as `model.h5`
- Predictions saved in `prediction_df.pkl`

---

## 📈 Key Results
- High validation accuracy on a limited dataset
- MTCNN and transfer learning helped overcome small sample size
- Accurate predictions on 13 unseen test images

---

## 📁 Project Files
face-recognition-cnn/
- Face Detection and Recognition.ipynb # Full training and evaluation notebook
- prediction_df.pkl # Pickled prediction DataFrame for CodeGrade
- README.md # Project documentation
