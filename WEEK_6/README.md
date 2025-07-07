# 🐶 Dog Breed Identification – Major Assignment 2

This project is part of the **Deep Learning Specialization – Week 6 (CNNs)** and focuses on building an image classification model to identify dog breeds using **Convolutional Neural Networks (CNNs)** and **Transfer Learning with MobileNetV2**.

---

## 📁 Files Included

| File                     | Description                                      |
|--------------------------|--------------------------------------------------|
| `cnn_model_full.h5`      | Trained CNN model using full training dataset    |
| `submission.csv`         | Kaggle-ready CSV with predicted test labels      |
| `dog_breed_identification_model.ipynb` | Complete notebook with all steps (data, training, prediction) |
| `README.md`              | Project overview and instructions                |

---

## 🧠 Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet, frozen)
- **Custom Classification Head**:
  - `GlobalAveragePooling2D`
  - `Dense(128, activation='relu')`
  - `Dense(num_classes, activation='softmax')`

- **Transfer Learning Strategy**:
  - Base model used for feature extraction
  - Only top layers trained

---

## 🔁 Dataset Overview

- **Source**: [Dog Breed Identification - Kaggle](https://www.kaggle.com/c/dog-breed-identification/)
- **Classes**: 120 dog breeds
- **Training Images**: ~10,000
- **Test Images**: ~10,000 (unlabeled)
- **Format**: `.jpg` files with filenames as image IDs
- **Labels**: Provided in `labels.csv`

---

## ⚙️ Training Setup

| Parameter        | Value         |
|------------------|---------------|
| Input Image Size | 224 × 224     |
| Batch Size       | 32            |
| Epochs Trained   | 10            |
| Optimizer        | Adam          |
| Loss Function    | Categorical Crossentropy |
| Data Augmentation | Preprocessing using MobileNetV2 |

---

## 📊 Results

- **Final Validation Accuracy**: **78.00%**
- **Training/Validation Split**: 80/20 using `train_test_split`
- **Model Evaluation**: Accuracy plotted per epoch

---

## 📌 How to Run

1. Upload the zipped dataset to Google Colab.
2. Run the notebook `dog_breed_identification_model.ipynb`.
3. Model will:
   - Extract images
   - Preprocess data
   - Train using MobileNetV2
   - Save predictions (`submission.csv`) and model (`cnn_model_full.h5`)

---

## ✅ Dependencies

- Python 3.x  
- TensorFlow 2.x  
- Pandas, NumPy  
- scikit-learn  
- Matplotlib

---



