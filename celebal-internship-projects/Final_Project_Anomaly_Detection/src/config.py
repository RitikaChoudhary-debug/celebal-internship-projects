# src/config.py

import os

# Define output directories for saving plots and cached data
OUTPUT_DIR = 'project_outputs_isolation_forest'
DATA_CACHE_DIR = 'cached_data'

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Path to the KDD Cup 99 dataset file
# IMPORTANT: Verify this path is correct on your system
FILE_PATH = r"C:\Users\lenovo\Desktop\celebal-internship-projects\celebal-internship-projects\Final_Project\kddcup.data_10_percent_corrected"

# Define column names for the KDD Cup 99 dataset
COLUMN_NAMES = [
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

# Optimal Model Hyperparameters derived from tuning
OPTIMAL_CONTAMINATION = 0.0792
OPTIMAL_N_ESTIMATORS = 50
OPTIMAL_MAX_FEATURES = 0.9