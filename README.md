# 📰 Fake News Detection

## 📌 Project Overview
The spread of fake news through online platforms and social media has become a major challenge in the digital era. Fake or misleading information can influence public opinion, create panic, and spread misinformation rapidly.

This project focuses on **Fake News Detection** using **supervised machine learning techniques** to classify news articles as **Fake** or **Real**. Multiple supervised learning models are trained and their accuracies are compared to determine the most effective approach for fake news classification.

---

## 🎯 Problem Statement
To design and implement a machine learning-based system that can automatically detect fake news articles by analyzing textual content and comparing the performance of different supervised learning algorithms.

---

## 🧠 Methodology
1. Data collection from an existing fake news dataset  
2. Text preprocessing and cleaning  
3. Feature extraction  
4. Training supervised machine learning models  
5. Comparing model performance using accuracy  



## ⚙️ Supervised Machine Learning Models Used

| Model | Description |
|------|------------|
| Random Forest | Ensemble-based classifier |
| Logistic Regression | Baseline linear classifier for text data |
| Naive Bayes | Probabilistic model |
| Support Vector Machine (Linear SVM) | Effective for high-dimensional text classification |
| K-Nearest Neighbors (KNN) | Distance-based supervised learning algorithm |

---

## 📊 Evaluation Metric
- **Accuracy Score**  

Model Accuracy Comparison:
Random Forest: 0.92
Logistic Regression: 0.89
Naive Bayes: 0.87
SVM: 0.93
KNN: 0.82

---

## Deployment 
Used streamlit in a simple UI , IN app.py 


## 📁 Project Structure

fake-news-detection/
│
├── app.py
├── Fake News Detection Dataset.csv
└── venv/

Run the app.py for the final output
