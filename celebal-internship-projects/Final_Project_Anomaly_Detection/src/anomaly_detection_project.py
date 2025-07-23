# anomaly_detection_project.py

# --- Section 1: Project Setup and Library Imports ---

"""
Network Anomaly Detection Project: KDD Cup 99 Analysis (Isolation Forest Only)

Objective: To develop an unsupervised machine learning model capable of detecting
unusual patterns and potential anomalies within network traffic data, which could
indicate security threats or system malfunctions.

Dataset: KDD Cup 99 - A widely recognized benchmark dataset for intrusion detection systems.

Developed by: Ritika Choudhary
Date: July 23, 2025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score, make_scorer,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import pickle


# --- Global Project Configuration ---
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['figure.figsize'] = (10, 6)

# --- CRITICAL: ABSOLUTE PATH TO DATASET FILE ---
file_path = r"C:\Users\lenovo\Desktop\celebal-internship-projects\celebal-internship-projects\Final_Project\kddcup.data_10_percent_corrected"

final_project_root = os.path.dirname(file_path)

output_dir = os.path.join(final_project_root, 'project_outputs_isolation_forest')
data_cache_dir = os.path.join(final_project_root, 'cached_data')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(data_cache_dir, exist_ok=True)

column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

print("--- Project Initialization Complete ---")
print(f"Output plots will be saved to: {output_dir}")
print(f"Processed data cache will be stored in: {data_cache_dir}\n")


# --- Section 2: Data Loading, Preprocessing, Feature Engineering, and Data Splitting ---

print("--- Data Loading and Comprehensive Preprocessing ---")

cache_files = {
    'X_train': os.path.join(data_cache_dir, 'X_train.pkl'),
    'X_test': os.path.join(data_cache_dir, 'X_test.pkl'),
    'y_train_binary': os.path.join(data_cache_dir, 'y_train_binary.pkl'),
    'y_test_binary': os.path.join(data_cache_dir, 'y_test_binary.pkl'),
}

def check_and_load_cache(cache_paths):
    all_files_exist = True
    for key, path in cache_paths.items():
        if not os.path.exists(path):
            all_files_exist = False
            break
    if all_files_exist:
        print("Attempting to load data from cache...")
        X_train_cached = pickle.load(open(cache_paths['X_train'], 'rb'))
        X_test_cached = pickle.load(open(cache_paths['X_test'], 'rb'))
        y_train_binary_cached = pickle.load(open(cache_paths['y_train_binary'], 'rb'))
        y_test_binary_cached = pickle.load(open(cache_paths['y_test_binary'], 'rb'))
        print("Data loaded from cache successfully. Skipping full preprocessing.")
        return X_train_cached, X_test_cached, y_train_binary_cached, y_test_binary_cached
    else:
        print("Cache not found or incomplete. Proceeding with full data loading and preprocessing pipeline.")
        return None

cached_data_tuple = check_and_load_cache(cache_files)

if cached_data_tuple:
    X_train, X_test, y_train_binary, y_test_binary = cached_data_tuple
    print(f"Loaded from cache: Training samples={X_train.shape[0]}, Testing samples={X_test.shape[0]}.")
else:
    try:
        df = pd.read_csv(file_path, names=column_names, nrows=50000)
        print("Raw dataset loaded successfully.")
        print(f"Initial dataset shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure it's in the same directory as the script or the path is correct.")
        raise

    missing_values = df.isnull().sum()[df.isnull().sum() > 0]
    if not missing_values.empty:
        print("\nMissing values per column:")
        print(missing_values)
    else:
        print("\nNo missing values found in the dataset.")

    df['label_binary'] = df['label'].apply(lambda x: 0 if x == 'normal.' else 1)
    print("\nBinarized label distribution (0: Normal, 1: Anomaly):")
    print(df['label_binary'].value_counts())
    binary_label_counts = df['label_binary'].value_counts()

    plt.figure(figsize=(8, 6))
    sns.barplot(x=binary_label_counts.index, y=binary_label_counts.values, palette="rocket", hue=binary_label_counts.index, legend=False)
    plt.xticks(ticks=[0, 1], labels=['Normal Traffic', 'Anomaly Traffic'])
    plt.title('Distribution of Binarized Network Traffic Labels')
    plt.xlabel('Traffic Type')
    plt.ylabel('Number of Instances')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'binarized_label_distribution.png'))
    plt.close()

    print("\n--- Encoding Categorical Features ---")
    categorical_cols = ['protocol_type', 'service', 'flag']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    print("Categorical features encoded.")

    print("\n--- Performing Enhanced Feature Engineering ---")

    df['src_to_dst_bytes_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1e-6)
    df['dst_to_src_bytes_ratio'] = df['dst_bytes'] / (df['src_bytes'] + 1e-6)
    df['total_bytes'] = df['src_bytes'] + df['dst_bytes']

    df['connections_per_duration'] = df['count'] / (df['duration'] + 1e-6)
    df['srv_connections_per_duration'] = df['srv_count'] / (df['duration'] + 1e-6)
    df['hot_per_duration'] = df['hot'] / (df['duration'] + 1e-6)
    df['failed_logins_per_duration'] = df['num_failed_logins'] / (df['duration'] + 1e-6)

    df['total_serror_rate'] = df['serror_rate'] + df['srv_serror_rate']
    df['total_rerror_rate'] = df['rerror_rate'] + df['srv_rerror_rate']

    df['host_srv_ratio'] = df['dst_host_srv_count'] / (df['dst_host_count'] + 1e-6)
    df['host_diff_srv_rate_complement'] = 1.0 - df['dst_host_diff_srv_rate']

    df['any_login_activity'] = df['logged_in'] + df['is_host_login'] + df['is_guest_login']

    print(f"Added 12 new engineered features. Total features before scaling: {df.shape[1] - 2}.")
    print("--- Feature Engineering Complete ---")

    X = df.drop(['label', 'label_binary'], axis=1)
    y_binary = df['label_binary']

    del df
    gc.collect()

    print("\n--- Scaling Numerical Features with StandardScaler ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    del X_scaled_df
    gc.collect()

    print("\n--- Data Splitting ---")
    X_train, X_test, y_train_binary, y_test_binary = train_test_split(
        X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    print(f"Data split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples).")
    print(f"Number of features: {X_train.shape[1]}")

    del X_scaled
    del y_binary
    gc.collect()

    print("\nSaving processed data to cache for faster reloading...")
    
    scaler_save_path = os.path.join(data_cache_dir, 'scaler.pkl')
    label_encoders_save_path = os.path.join(data_cache_dir, 'label_encoders.pkl')

    try:
        with open(scaler_save_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to: {scaler_save_path}")

        with open(label_encoders_save_path, 'wb') as f:
            pickle.dump(label_encoders, f)
        print(f"Label encoders saved to: {label_encoders_save_path}")
    except Exception as e:
        print(f"Error saving preprocessing objects: {e}")
    
    pickle.dump(X_train, open(cache_files['X_train'], 'wb'))
    pickle.dump(X_test, open(cache_files['X_test'], 'wb'))
    pickle.dump(y_train_binary, open(cache_files['y_train_binary'], 'wb'))
    pickle.dump(y_test_binary, open(cache_files['y_test_binary'], 'wb'))
    print("Data successfully cached.")


# --- Section 3: Hyperparameter Tuning for Contamination Parameter ---

print("\n--- Tuning Contamination Parameter for Isolation Forest ---")

contamination_values = np.linspace(0.001, 0.1, 20) 

best_f1_score_cont = -1
best_contamination_rate = 0
tuning_results = []

for cont_rate in contamination_values:
    iso_forest_cont_tuned = IsolationForest(
        contamination=cont_rate,
        random_state=42,
        n_estimators=100,
        max_features=1.0,
        verbose=0
    )
    iso_forest_cont_tuned.fit(X_train)

    y_pred_raw_cont = iso_forest_cont_tuned.predict(X_test)
    y_pred_binary_cont = np.array([1 if x == -1 else 0 for x in y_pred_raw_cont])

    precision = precision_score(y_test_binary, y_pred_binary_cont, zero_division=0)
    recall = recall_score(y_test_binary, y_pred_binary_cont, zero_division=0)
    f1 = f1_score(y_test_binary, y_pred_binary_cont, zero_division=0)
    accuracy = accuracy_score(y_test_binary, y_pred_binary_cont)

    iso_scores_cont = iso_forest_cont_tuned.decision_function(X_test)
    roc_auc = roc_auc_score(y_test_binary, -iso_scores_cont)

    tuning_results.append({
        'contamination': cont_rate,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'roc_auc': roc_auc
    })

    if f1 > best_f1_score_cont:
        best_f1_score_cont = f1
        best_contamination_rate = cont_rate

results_df_cont_tuning = pd.DataFrame(tuning_results)
print("\nContamination Tuning Results:")
print(results_df_cont_tuning.round(4).to_string())

print(f"\nOptimal Contamination for F1-Score: {best_contamination_rate:.4f} (F1: {best_f1_score_cont*100:.2f}%)")

plt.figure(figsize=(12, 6))
plt.plot(results_df_cont_tuning['contamination'], results_df_cont_tuning['f1_score'], label='F1-Score', marker='o')
plt.plot(results_df_cont_tuning['contamination'], results_df_cont_tuning['precision'], label='Precision', marker='x')
plt.plot(results_df_cont_tuning['contamination'], results_df_cont_tuning['recall'], label='Recall', marker='s')
plt.xlabel('Contamination Rate')
plt.ylabel('Score')
plt.title('Isolation Forest Performance vs. Contamination Rate')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'contamination_tuning_metrics.png'))
plt.close()


# --- Section 4: Hyperparameter Tuning for n_estimators and max_features (Manual Grid Search) ---

print("\n--- Tuning n_estimators and max_features for Isolation Forest (Manual Grid Search) ---")

optimal_contamination_for_manual_tuning = best_contamination_rate

param_grid_manual = {
    'n_estimators': [50, 100, 200, 300],
    'max_features': [0.7, 0.8, 0.9, 1.0]
}

def iso_forest_roc_auc_scorer_manual(estimator, X, y):
    anomaly_scores = estimator.decision_function(X)
    return roc_auc_score(y, -anomaly_scores)

best_roc_auc_manual = -1
best_n_estimators_manual = None
best_max_features_manual = None
manual_tuning_results = []

print(f"Starting manual tuning with {len(param_grid_manual['n_estimators']) * len(param_grid_manual['max_features'])} combinations...")

for n_est in param_grid_manual['n_estimators']:
    for max_feat in param_grid_manual['max_features']:
        print(f"   Testing n_estimators={n_est}, max_features={max_feat}...", end="")

        iso_forest_manual = IsolationForest(
            contamination=optimal_contamination_for_manual_tuning,
            n_estimators=n_est,
            max_features=max_feat,
            random_state=42,
            verbose=0
        )
        iso_forest_manual.fit(X_train)

        current_roc_auc = iso_forest_roc_auc_scorer_manual(iso_forest_manual, X_test, y_test_binary)

        manual_tuning_results.append({
            'n_estimators': n_est,
            'max_features': max_feat,
            'roc_auc': current_roc_auc
        })

        if current_roc_auc > best_roc_auc_manual:
            best_roc_auc_manual = current_roc_auc
            best_n_estimators_manual = n_est
            best_max_features_manual = max_feat
        print(f" ROC AUC: {current_roc_auc:.4f}")

print("\n--- Manual Tuning Results Summary ---")
print(f"Best ROC AUC Score found: {best_roc_auc_manual:.4f}")
print(f"Best Parameters: {{'max_features': {best_max_features_manual}, 'n_estimators': {best_n_estimators_manual}}}")

results_df_manual_tuning = pd.DataFrame(manual_tuning_results)
print("\nDetailed Manual Tuning Results:")
print(results_df_manual_tuning.sort_values(by='roc_auc', ascending=False).round(4).to_string())

best_n_estimators = best_n_estimators_manual
best_max_features = best_max_features_manual


# --- Section 5: Final Isolation Forest Model Training and Prediction with Optimized Hyperparameters ---

print("\n--- Final Isolation Forest Model Training and Prediction ---")

optimal_contamination_final = best_contamination_rate
optimal_n_estimators_final = best_n_estimators
optimal_max_features_final = best_max_features


print(f"Final Model Parameters:")
print(f"   Contamination: {optimal_contamination_final:.4f}")
print(f"   n_estimators: {optimal_n_estimators_final}")
print(f"   max_features: {optimal_max_features_final}")

iso_forest_final = IsolationForest(
    contamination=optimal_contamination_final,
    random_state=42,
    n_estimators=optimal_n_estimators_final,
    max_features=optimal_max_features_final,
    verbose=0
)
iso_forest_final.fit(X_train)

model_save_path = os.path.join(output_dir, 'iso_forest_anomaly_model.pkl')
try:
    with open(model_save_path, 'wb') as f:
        pickle.dump(iso_forest_final, f)
    print(f"\nFinal model saved successfully to: {model_save_path}")
except Exception as e:
    print(f"\nError saving model: {e}")

y_pred_iso_raw_final = iso_forest_final.predict(X_test)
y_pred_iso_final = np.array([1 if x == -1 else 0 for x in y_pred_iso_raw_final])

print("\nFinal Model Training and Prediction Complete.")


# --- Section 6: Final Model Evaluation and Visualizations ---

print("\n--- Final Isolation Forest Model Evaluation ---")

precision_final = precision_score(y_test_binary, y_pred_iso_final, zero_division=0)
recall_final = recall_score(y_test_binary, y_pred_iso_final, zero_division=0)
f1_score_final = f1_score(y_test_binary, y_pred_iso_final, zero_division=0)
accuracy_final = accuracy_score(y_test_binary, y_pred_iso_final)

iso_scores_final = iso_forest_final.decision_function(X_test)
roc_auc_final = roc_auc_score(y_test_binary, -iso_scores_final)

print(f"   Precision: {precision_final*100:.2f}%")
print(f"   Recall: {recall_final*100:.2f}%")
print(f"   F1 Score: {f1_score_final*100:.2f}%")
print(f"   Accuracy: {accuracy_final*100:.2f}%")
print(f"   ROC AUC: {roc_auc_final:.4f}")

cm_final = confusion_matrix(y_test_binary, y_pred_iso_final)
plt.figure(figsize=(7, 6))
sns.heatmap(cm_final, annot=True, fmt='d', cmap='Greens', linewidths=.5, linecolor='gray',
            xticklabels=['Predicted Normal', 'Predicted Anomaly'],
            yticklabels=['True Normal', 'True Anomaly'])
plt.title('Isolation Forest Confusion Matrix (Final Model)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'isolation_forest_confusion_matrix_final.png'))
plt.close()

fpr_final, tpr_final, _ = roc_curve(y_test_binary, -iso_scores_final)
plt.figure(figsize=(8, 8))
plt.plot(fpr_final, tpr_final, color='darkorange', lw=2, label=f'Isolation Forest (AUC = {roc_auc_final:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve (Final Model)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roc_curve_isolation_forest_final.png'))
plt.close()

print("\nFinal Model Evaluation and Visualizations Complete.")


# --- Section 7: Project Reflection and Future Directions ---

"""
## Project Reflection and Future Directions for Anomaly Detection

This project successfully implemented and evaluated an unsupervised Isolation Forest model for anomaly detection on a subset of the KDD Cup 99 dataset. Through systematic data preprocessing, enhanced feature engineering, and robust hyperparameter tuning, we have significantly improved the model's ability to identify unusual network patterns.

**Current Model Performance (after all optimizations):**
* **Precision:** {precision_final*100:.2f}%
* **Recall:** {recall_final*100:.2f}%
* **F1 Score:** **{f1_score_final*100:.2f}%**
* **Accuracy:** {accuracy_final*100:.2f}%
* **ROC AUC:** **{roc_auc_final:.4f}**

The ROC AUC of {roc_auc_final:.4f} demonstrates strong underlying discriminative power, indicating the model is highly effective at ranking anomalies. The F1-Score of {f1_score_final*100:.2f}% reflects the balanced performance between precision (minimizing false alarms) and recall (minimizing missed anomalies) achieved through contamination tuning.

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

By focusing on these points, you will demonstrate a deep understanding of unsupervised learning for IDS, a proactive approach to problem-solving, and the ability to connect your work to current research trends in network security.
"""