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

celebal-internship-projects/
├── README.md                   # Project overview, setup, and documentation (This file)
├── Final_Project_Anomaly_Detection/ # Main project directory
│   ├── App_Demo.gif              # GIF demonstrating the Streamlit web application
│   ├── Model_demo.gif            # GIF demonstrating the model training pipeline
│   ├── app.py                    # Streamlit web application for interactive predictions
│   ├── kddcup.data_10_percent_corrected # The raw dataset file
│   ├── project_outputs_isolation_forest/ # Directory for saved plots and the trained model
│   │   ├── iso_forest_anomaly_model.pkl # The trained Isolation Forest model
│   │   └── ... (evaluation plots: confusion matrix, ROC curve, tuning plots)
│   ├── cached_data/              # Directory for cached preprocessed data and preprocessing objects
│   │   ├── scaler.pkl            # Trained StandardScaler object
│   │   ├── label_encoders.pkl    # Trained LabelEncoder objects for categorical features
│   │   └── ... (cached X_train, X_test, y_train_binary, y_test_binary)
│   └── src/                      # Source code for the core ML pipeline
│       └── anomaly_detection_project.py # Main script for data processing, model training, and evaluation
└── ... (other project folders within 'celebal-internship-projects')
