# Network Anomaly Detection Platform

## ✨ Live Demos ✨

### 🚀 Interactive Web App Demo (Streamlit)

[![Streamlit App Demo](https://github.com/RitikaChoudhary-debug/celebal-internship-projects/raw/main/celebal-internship-projects/Final_Project_Anomaly_Detection/App_Demo.gif)](https://github.com/RitikaChoudhary-debug/celebal-internship-projects/raw/main/celebal-internship-projects/Final_Project_Anomaly_Detection/App_Demo.gif)

_This GIF showcases the user interface of the Streamlit application, demonstrating how users can input network parameters and receive real-time anomaly predictions._

### 📊 Model Training & Evaluation Pipeline Demo (Console Output)

[![Model Pipeline Demo](https://github.com/RitikaChoudhary-debug/celebal-internship-projects/raw/main/celebal-internship-projects/Final_Project_Anomaly_Detection/Model_demo.gif)](https://github.com/RitikaChoudhary-debug/celebal-internship-projects/raw/main/celebal-internship-projects/Final_Project_Anomaly_Detection/Model_demo.gif)

_This GIF provides a quick view of the backend machine learning pipeline executing, including data loading, preprocessing, tuning, and final evaluation metrics printed to the console._

---

## 🚀 Project Overview

This project implements an unsupervised machine learning solution for detecting anomalies in network traffic data. Leveraging the Isolation Forest algorithm, the platform identifies unusual patterns that could signify security breaches, system malfunctions, or other critical deviations from normal network behavior.

**Key Highlights:**
* **Unsupervised Learning:** Detects anomalies without requiring pre-labeled attack data for training.
* **Isolation Forest:** An efficient algorithm for anomaly detection, particularly effective in high-dimensional datasets.
* **Enhanced Feature Engineering:** Custom-engineered features derived from raw network attributes to provide richer, more discriminative information to the model.
* **Robust Hyperparameter Tuning:** Systematic optimization of model parameters for improved performance.
* **Streamlit Web Application:** A user-friendly interface for interactive anomaly prediction.
* **Modular Codebase:** Organized into distinct Python modules for maintainability and scalability.

**Dataset:** KDD Cup 99 - A widely recognized benchmark dataset for intrusion detection systems.

**Developed by:** Ritika Choudhary
**Date:** July 23, 2025

---

## 📂 Project Structure

---

## 🛠️ Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/RitikaChoudhary-debug/celebal-internship-projects.git](https://github.com/RitikaChoudhary-debug/celebal-internship-projects.git)
    cd celebal-internship-projects/Final_Project_Anomaly_Detection
    ```
    _You need to navigate into the specific project folder after cloning._

2.  **Create a Conda Environment (Recommended):**
    ```bash
    conda create -n anomaly_env python=3.10 # Or python=3.11, 3.12
    conda activate anomaly_env
    ```
3.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn streamlit
    ```
4.  **Place the Dataset:**
    Download the `kddcup.data_10_percent_corrected` dataset and place it directly into the `Final_Project_Anomaly_Detection/` directory (next to `app.py` and `src/`).
    _The `file_path` variable in `src/anomaly_detection_project.py` is configured to find it here._

---

## 🚀 How to Run the Project

### 1. Run the Main ML Pipeline (Training & Evaluation)

This script will perform data loading, preprocessing, feature engineering, hyperparameter tuning, model training, and evaluation. It will also save the trained model and preprocessing objects.

Navigate to the `Final_Project_Anomaly_Detection/` directory in your terminal (and activate your conda environment):

```bash
cd celebal-internship-projects/Final_Project_Anomaly_Detection
conda activate anomaly_env # if you created one
python -m src.anomaly_detection_project

## 💡 Project Reflection and Future Directions

This project successfully implemented and evaluated an unsupervised Isolation Forest model for anomaly detection on a subset of the KDD Cup 99 dataset. Through systematic data preprocessing, enhanced feature engineering, and robust hyperparameter tuning, we have significantly improved the model's ability to identify unusual network patterns.

**Current Model Performance (after all optimizations):**
* **Precision:** 56.91%
* **Recall:** 18.82%
* **F1 Score:** **28.29%**
* **Accuracy:** 77.03%
* **ROC AUC:** **0.9004**

The ROC AUC of 0.9004 demonstrates strong underlying discriminative power, indicating the model is highly effective at ranking anomalies. The F1-Score of 28.29% reflects the balanced performance between precision (minimizing false alarms) and recall (minimizing missed anomalies) achieved through contamination tuning.

### Suggestions for enhancing this anomaly detection project:

#### 1. Advanced Feature Engineering and Selection:
    - **Iterative Refinement:** Continue exploring more complex interaction terms, polynomial features, or time-series-based features if the raw data allows for it (e.g., aggregating features over different sliding windows for specific source/destination IPs).
    - **Feature Selection Algorithms:** Apply methods like Recursive Feature Elimination (RFE) or feature importance from other tree-based models (e.g., RandomForestClassifier if used in a supervised context) to identify and retain only the most impactful features, potentially reducing noise and improving generalization.

#### 2. Exploring Other Machine Learning Approaches:
    - **Supervised Learning (Highly Recommended):** Since this dataset has labels, framing it as a highly imbalanced supervised classification problem would likely yield significantly better results.
      - **Algorithms:** Experiment with Gradient Boosting Machines (XGBoost, LightGBM, CatBoost) or Deep Neural Networks.
      - **Imbalance Handling:** Implement techniques like SMOTE (Synthetic Minority Over-sampling Technique) for oversampling the minority class, or using `class_weight` parameters in classifiers.
      - **Evaluation:** Focus on Precision-Recall curves and Average Precision (AP) score, which are more appropriate for imbalanced datasets than ROC AUC.
    - **Other Unsupervised Methods:** Explore models like One-Class SVM (OCSVM), Local Outlier Factor (LOF), or Autoencoders to compare their performance against Isolation Forest.

#### 3. Hyperparameter Tuning Refinement:
    - **Expanded Search Space:** For `n_estimators` and `max_features`, explore a wider range of values, or use `RandomizedSearchCV` for a more efficient search of larger spaces before fine-tuning with `GridSearchCV`.
    - **Contamination Robustness:** Instead of a fixed contamination, consider adaptive methods or using a threshold based on a specific precision/recall target if such a target is known.

#### 4. Model Interpretability and Explainability:
    - For a detected anomaly, investigate techniques (e.g., SHAP values, LIME) to understand *why* it was flagged by the Isolation Forest. This can involve analyzing the feature contributions to the anomaly score, which is crucial for security analysts to respond effectively.

#### 5. Real-world Deployment Considerations:
    - Discuss how such a model could be integrated into a real-time network monitoring system (e.g., through stream processing frameworks). Consider model latency and resource requirements for high-throughput environments.

#### 6. Leveraging Newer Datasets:
    - Validate the approach on more contemporary anomaly detection datasets (e.g., NSL-KDD, CICIDS2017, UNSW-NB15) to demonstrate its applicability to current threat landscapes.

By focusing on these points, you will demonstrate a deep understanding of unsupervised learning for IDS, a proactive approach to problem-solving, and the ability to connect your work to current research trends in network security. This will undoubtedly help you stand out for your PPO opportunity!
