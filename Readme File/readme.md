# Diabetes Prediction Using Supervised Machine Learning Models

## 📌 Project Overview

This project presents a comparative study of supervised machine learning models for predicting diabetes using the PIMA Indians Diabetes Dataset. The implementation follows a structured experimental framework including data preprocessing, model training, evaluation, and visualization.

The objective is to identify the most effective model for early diabetes detection based on multiple performance metrics.

---

## 🎯 Objectives

* To implement and compare multiple supervised learning algorithms
* To evaluate models using standardized preprocessing techniques
* To analyze performance using multiple evaluation metrics
* To identify the best-performing model for diabetes prediction

---

## 🧠 Models Implemented

The following machine learning models are used:

* Logistic Regression (LR)
* Decision Tree (DT)
* Support Vector Machine (SVM)
* Random Forest (RF)

These models represent different learning approaches:

* Linear model → Logistic Regression
* Tree-based model → Decision Tree
* Kernel-based model → SVM
* Ensemble model → Random Forest

---

## 📊 Dataset

* Dataset: **PIMA Indians Diabetes Dataset**
* Source: UCI Machine Learning Repository
* Records: 768
* Features: 8 input attributes

### Features include:

* Pregnancies
* Glucose
* Blood Pressure
* Skin Thickness
* Insulin
* BMI
* Diabetes Pedigree Function
* Age

Target variable:

* Outcome (0 = Non-diabetic, 1 = Diabetic)

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Replace zero values with missing values
* Handle missing values using median imputation
* Feature scaling using StandardScaler

### 2. Data Splitting

* Train-Test Split: 80% training, 20% testing
* Stratified sampling to maintain class balance

### 3. Model Training

* All models trained under identical conditions
* Fixed random seed for reproducibility

### 4. Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score

### 5. Visualization

* Confusion Matrix
* ROC Curve
* Precision–Recall Curve
* Accuracy Comparison Graph

---

## 📁 Project Structure

```
diabetes-prediction-ml/
│── data/
│   └── diabetes.csv
│
│── src/
│   ├── preprocess.py
│   ├── train_models.py
│   ├── evaluate.py
│   └── main.py
│
│── outputs/
│   ├── accuracy_plot.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── pr_curve.png
│
│── requirements.txt
│── README.md
```

---

## 🛠 Installation

Install required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Step 1: Navigate to source folder

```bash
cd src
```

### Step 2: Run the program

```bash
python main.py
```

---

## 📈 Output

The following outputs will be generated in the `outputs/` folder:

* Accuracy comparison graph
* Confusion matrix
* ROC curve
* Precision–Recall curve

Additionally, model performance metrics will be printed in the console.

---

## 📊 Expected Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~79%     |
| Decision Tree       | ~83%     |
| SVM                 | ~86%     |
| Random Forest       | ~91%     |

👉 Random Forest performs best due to ensemble learning.

---

## ⚠️ Limitations

* Dataset is relatively small and population-specific
* Missing values handled using basic imputation
* Only four models evaluated
* No cross-validation used

---

## 🚀 Future Work

* Implement cross-validation techniques
* Apply hyperparameter tuning (GridSearchCV)
* Use advanced models (Gradient Boosting, Neural Networks)
* Deploy model using a web application

---

## 👨‍💻 Authors

* Ayush Bhrouta
* Saksham Vashisht
* Rupindeep Singh

Supervisor: Dr. Gifty Gupta

---

## 📚 Reference

Based on the research paper:
**“Diabetes Prediction Using Supervised Machine Learning Models”**
