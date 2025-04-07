# SC5010 – A Double Analytics Framework for Robust Diabetes Prediction with Incomplete Clinical Data

This project investigates the effectiveness of a **double analytics approach** for predicting diabetes risk in patients using the Pima Indians Diabetes Dataset. The approach leverages machine learning to handle incomplete data (specifically, missing `SkinThickness` values) before performing classification. This is particularly valuable in clinical settings where testing resources are limited.

## 🔍 Problem Statement

Can we achieve better diabetes prediction accuracy using a two-stage "double analytics" approach (regression + classification) versus a single-stage classification model alone?

In real-world healthcare scenarios, missing diagnostic values are common. Our hypothesis is that **predicting missing values using regression before classification can lead to better accuracy and reliability**, especially for identifying high-risk patients.

## 🎯 Motivation

Testing an entire population for diabetes using clinical methods is costly and resource-intensive. This project aims to create a machine learning pipeline that:
- Works even when some diagnostic features are missing
- Identifies high-risk patients more effectively
- Reduces the burden on healthcare systems

## 🧪 Methodology

### Phase 1 – Data Preparation
- Cleaned dataset by removing rows with `0` values in **BloodPressure**, **BMI**, or **Glucose**

### Phase 2 – Five Comparative Experiments

| Experiment | Description |
|-----------|-------------|
| 1. Cleaned Dataset | Predict diabetes using original dataset with `SkinThickness` containing zeroes |
| 2. No SkinThickness | Predict diabetes with no `SkinThickness` attributes |
| 3. No Zero SkinThickness | Remove rows where `SkinThickness == 0` and predict diabetes |
| 4. Predicted SkinThickness | Use Random Forest Regressor to impute missing `SkinThickness` then predict diabetes |
| 5. Predicted SkinThickness (Hypertuned) | Use GridSearch optimized RFR to impute missing `SkinThickness` then predict diabetes |

All experiments use the **same 80% of the data** for training and the remaining **20% as the evaluation set**.

## ⚙️ Notebooks

| Notebook | Purpose |
|---------|---------|
| `Data_Cleaning.ipynb` | Prepare datasets with and without zero values or SkinThickness |
| `Predict_SkinThickness.ipynb` | Predict `SkinThickness` using RFR |
| `Predict_SkinThickness_Hypertune.ipynb` | Predict `SkinThickness` using RFR with GridSearch |
| `Diabetes_Prediction_no_SkinThickness.ipynb` | Classify diabetes without using `SkinThickness` attribute |
| `Diabetes_Prediction_without_zero_SkinThickness.ipynb` | Classify diabetes without zero SkinThickness dataset |
| `Diabetes_Prediction_predicted_SkinThickness.ipynb` | Classify diabetes with RFR-imputed SkinThickness |
| `Diabetes_Prediction_hyptertune_SkinThickness.ipynb` | Classify diabetes with  hyperparameter-tuned-RFR-imputed SkinThickness |

## 📂 Datasets 

| File | Description |
|------|-------------|
| `diabetes.csv` | Original dataset |
| `diabetes_no_zeros.csv` | Removed rows with 0 in BloodPressure, BMI, or Glucose |
| `diabetes_no_SkinThickness.csv` | Dropped the entire SkinThickness column |
| `diabetes_without_zero_SkinThickness.csv` | Dropped rows where SkinThickness = 0 |
| `diabetes_prediction_SkinThickness.csv` | Imputed SkinThickness using RFR |
| `diabetes_prediction_SkinThickness_Hypertune.csv` | Imputed SkinThickness using hypertuned RFR |

## 📊 Results

Visual comparisons of the classification performance (e.g., accuracy, precision, recall, F1-score) across different models:

| File Name | Description |
|-----------|-------------|
| `model_metrics_plot_no_SkinThickness.png` | Diabetes prediction without the `SkinThickness` feature |
| `model_metrics_plot_without_zero_SkinThickness.png` | Diabetes prediction using only rows with valid `SkinThickness` values |
| `model_metrics_plot_predicted_SkinThickness.png` | Diabetes prediction using `SkinThickness` predicted |
| `model_metrics_plot_hypertune_predicted_SkinThickness.png` | Diabetes prediction using `SkinThickness` predicted via hypertuned |

These visualizations help illustrate how different approaches to handling missing values affect model performance, and support the value of our double analytics framework.

## 📈 Evaluation Metrics (Result)

To compare the effectiveness of the approaches, we use:
- **Accuracy** (macro & weighted)
- **Precision**
- **Recall** (emphasized)
- **F1 Score**

> Since our goal is to flag high-risk individuals, **Recall and False Negatives** are particularly important.

## 📌 Key Takeaways

- The **double analytics approach** (regression + classification) provides **better prediction accuracy** than models that ignore or drop missing values, but still perform **worse** than models without the attributes.
- **Hyperparameter tuning** further improves the regression step and boosts the final classification results.
- The model is better suited for **early screening**, where conservative detection (higher recall) is more critical than overall accuracy.

## 🧠 Future Improvements

- Explore deep learning models (e.g., autoencoders) for feature reconstruction

## 👥 Authors

- Pham Nguyen Hung (U2140410D)  
- Shi Lisheng (U2320567L)  
- Kho Kah Chun (U2120491D)  
- Kribalan Pugalenthi (U2340115D)

---

📌 *This project was developed as part of the SC5010 final group project at NTU.*
