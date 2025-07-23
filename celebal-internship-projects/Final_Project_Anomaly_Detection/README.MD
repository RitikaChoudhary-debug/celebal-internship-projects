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
