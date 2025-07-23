# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# --- Configuration and File Paths ---
DATA_CACHE_DIR = 'cached_data'
OUTPUT_DIR = 'project_outputs_isolation_forest'

MODEL_PATH = os.path.join(OUTPUT_DIR, 'iso_forest_anomaly_model.pkl')
SCALER_PATH = os.path.join(DATA_CACHE_DIR, 'scaler.pkl')
LABEL_ENCODERS_PATH = os.path.join(DATA_CACHE_DIR, 'label_encoders.pkl')

# Define column names (must match the order used during model training)
COLUMN_NAMES_EXCEPT_LABEL = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
]

# Names of categorical columns
CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']


# --- Load Saved Model and Preprocessing Objects ---
@st.cache_resource
def load_model_and_preprocessors():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(LABEL_ENCODERS_PATH, 'rb') as f:
            label_encoders = pickle.load(f)
        return model, scaler, label_encoders
    except FileNotFoundError as e:
        st.error(f"Error: Required file not found. Please ensure all .pkl files are in '{DATA_CACHE_DIR}' and '{OUTPUT_DIR}'. Details: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading files: {e}")
        st.stop()

model, scaler, label_encoders = load_model_and_preprocessors()


# --- Function to preprocess new input data ---
def preprocess_input(input_data_dict, scaler, label_encoders, column_names_ordered, categorical_cols):
    input_df = pd.DataFrame([input_data_dict])

    for col in categorical_cols:
        if col in input_df.columns:
            try:
                input_df[col] = label_encoders[col].transform(input_df[col])
            except ValueError:
                input_df[col] = 0 
                st.warning(f"Warning: Unseen categorical value '{input_data_dict[col]}' for '{col}'. Assigned to default (0).")

    # Perform Feature Engineering
    input_df['src_to_dst_bytes_ratio'] = input_df['src_bytes'] / (input_df['dst_bytes'] + 1e-6)
    input_df['dst_to_src_bytes_ratio'] = input_df['dst_bytes'] / (input_df['src_bytes'] + 1e-6)
    input_df['total_bytes'] = input_df['src_bytes'] + input_df['dst_bytes']
    input_df['connections_per_duration'] = input_df['count'] / (input_df['duration'] + 1e-6)
    input_df['srv_connections_per_duration'] = input_df['srv_count'] / (input_df['duration'] + 1e-6)
    input_df['hot_per_duration'] = input_df['hot'] / (input_df['duration'] + 1e-6)
    input_df['failed_logins_per_duration'] = input_df['num_failed_logins'] / (input_df['duration'] + 1e-6)
    input_df['total_serror_rate'] = input_df['serror_rate'] + input_df['srv_serror_rate']
    input_df['total_rerror_rate'] = input_df['rerror_rate'] + input_df['srv_rerror_rate']
    input_df['host_srv_ratio'] = input_df['dst_host_srv_count'] / (input_df['dst_host_count'] + 1e-6)
    input_df['host_diff_srv_rate_complement'] = 1.0 - input_df['dst_host_diff_srv_rate']
    input_df['any_login_activity'] = input_df['logged_in'] + input_df['is_host_login'] + input_df['is_guest_login']
    
    final_feature_columns = column_names_ordered + [
        'src_to_dst_bytes_ratio', 'dst_to_src_bytes_ratio', 'total_bytes',
        'connections_per_duration', 'srv_connections_per_duration', 'hot_per_duration',
        'failed_logins_per_duration', 'total_serror_rate', 'total_rerror_rate',
        'host_srv_ratio', 'host_diff_srv_rate_complement', 'any_login_activity'
    ]
    
    input_df = input_df[final_feature_columns]

    scaled_input = scaler.transform(input_df)
    return scaled_input

# --- Streamlit App Interface ---
st.set_page_config(page_title="Network Anomaly Detector", layout="wide")
st.title("🛡️ Network Anomaly Detection Platform")
st.markdown("""
    This application uses an unsupervised Isolation Forest model to detect unusual patterns
    in network traffic data, which could indicate potential security breaches.
    **Enter connection parameters below and click 'Predict' to check for anomalies.**
""")

# Define default values for input fields (representing a 'normal' connection or common values)
default_vals = {
    "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
    "src_bytes": 100, "dst_bytes": 100, "land": 0, "wrong_fragment": 0, "urgent": 0,
    "hot": 0, "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
    "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
    "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0,
    "is_guest_login": 0, "count": 10, "srv_count": 10, "serror_rate": 0.0,
    "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
    "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
    "dst_host_count": 50, "dst_host_srv_count": 50, "dst_host_same_srv_rate": 1.0,
    "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 1.0,
    "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
    "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0
}

# Example data for "Load Example" buttons
NORMAL_EXAMPLE_DATA = {
    "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
    "src_bytes": 45, "dst_bytes": 45, "land": 0, "wrong_fragment": 0, "urgent": 0,
    "hot": 0, "num_failed_logins": 0, "logged_in": 1, "num_compromised": 0,
    "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
    "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0,
    "is_guest_login": 0, "count": 2, "srv_count": 2, "serror_rate": 0.0,
    "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
    "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
    "dst_host_count": 255, "dst_host_srv_count": 255, "dst_host_same_srv_rate": 1.0,
    "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 1.0,
    "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
    "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0
}

ANOMALY_EXAMPLE_DATA = {
    "duration": 0, "protocol_type": "icmp", "service": "ecr_i", "flag": "SF",
    "src_bytes": 1032, "dst_bytes": 0, "land": 0, "wrong_fragment": 0, "urgent": 0,
    "hot": 0, "num_failed_logins": 0, "logged_in": 0, "num_compromised": 0,
    "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
    "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0,
    "is_guest_login": 0, "count": 140, "srv_count": 140, "serror_rate": 0.0,
    "srv_serror_rate": 0.0, "rerror_rate": 0.0, "srv_rerror_rate": 0.0,
    "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
    "dst_host_count": 255, "dst_host_srv_count": 255, "dst_host_same_srv_rate": 1.0,
    "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 1.0,
    "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
    "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
    "dst_host_srv_rerror_rate": 0.0
}


# Initialize session state for all inputs if not already present
if 'input_state' not in st.session_state:
    st.session_state['input_state'] = default_vals.copy()

if 'form_key' not in st.session_state:
    st.session_state['form_key'] = 'anomaly_prediction_form_initial'

def update_form_with_example(example_data):
    st.session_state['input_state'] = example_data.copy()
    st.session_state['form_key'] = str(np.random.rand()) 

def get_categorical_options(le):
    return list(le.classes_)

# --- Load Example Buttons ---
st.markdown("### Load Example Data:")
col_ex1, col_ex2, _ = st.columns([0.2, 0.2, 0.6])
with col_ex1:
    if st.button("Load Normal Example", help="Fills form with parameters of a typical normal HTTP connection."):
        update_form_with_example(NORMAL_EXAMPLE_DATA)
        st.rerun()
with col_ex2:
    if st.button("Load Anomaly Example", help="Fills form with parameters of a typical ICMP Echo (ping) scan anomaly."):
        update_form_with_example(ANOMALY_EXAMPLE_DATA)
        st.rerun()

st.markdown("---")


# --- Input Fields Layout ---
st.header("Connection Parameters")

with st.form(key=st.session_state.form_key):
    input_values = {}

    def get_current_input_value(key_name):
        return st.session_state['input_state'].get(key_name)

    col1, col2, col3 = st.columns(3)
    with col1:
        input_values["duration"] = st.number_input(
            "Duration (seconds)", min_value=0, value=get_current_input_value("duration"), 
            help="Length of the connection in seconds."
        )
        input_values["protocol_type"] = st.selectbox(
            "Protocol Type", options=get_categorical_options(label_encoders['protocol_type']), 
            index=get_categorical_options(label_encoders['protocol_type']).index(get_current_input_value("protocol_type")),
            help="Type of the protocol (e.g., tcp, udp, icmp)."
        )
    with col2:
        input_values["service"] = st.selectbox(
            "Service", options=get_categorical_options(label_encoders['service']), 
            index=get_categorical_options(label_encoders['service']).index(get_current_input_value("service")),
            help="Network service on the destination (e.g., http, telnet, ftp)."
        )
        input_values["flag"] = st.selectbox(
            "Flag", options=get_categorical_options(label_encoders['flag']), 
            index=get_categorical_options(label_encoders['flag']).index(get_current_input_value("flag")),
            help="Normal or error status of the connection (e.g., SF, S0, REJ)."
        )
    with col3:
        input_values["src_bytes"] = st.number_input(
            "Source Bytes", min_value=0, value=get_current_input_value("src_bytes"), 
            help="Number of data bytes from source to destination."
        )
        input_values["dst_bytes"] = st.number_input(
            "Destination Bytes", min_value=0, value=get_current_input_value("dst_bytes"), 
            help="Number of data bytes from destination to source."
        )

    with st.expander("Advanced Connection Details"):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        with adv_col1:
            input_values["land"] = st.number_input("Land (1=connection from/to same host/port)", min_value=0, max_value=1, value=get_current_input_value("land"), help="1 if connection is to/from the same host/port.")
            input_values["wrong_fragment"] = st.number_input("Wrong Fragment", min_value=0, value=get_current_input_value("wrong_fragment"), help="Number of 'wrong' fragments.")
            input_values["urgent"] = st.number_input("Urgent", min_value=0, value=get_current_input_value("urgent"), help="Number of urgent packets.")
            input_values["hot"] = st.number_input("Hot (indicators of hot activity)", min_value=0, value=get_current_input_value("hot"), help="Number of 'hot' indicators (e.g., accessing sensitive directories).")
            input_values["num_failed_logins"] = st.number_input("Failed Logins", min_value=0, value=get_current_input_value("num_failed_logins"), help="Number of failed login attempts.")
            input_values["logged_in"] = st.number_input("Logged In (1 if successful login)", min_value=0, max_value=1, value=get_current_input_value("logged_in"), help="1 if successfully logged in.")
            input_values["num_compromised"] = st.number_input("Num Compromised", min_value=0, value=get_current_input_value("num_compromised"), help="Number of compromised conditions.")
        with adv_col2:
            input_values["root_shell"] = st.number_input("Root Shell (1 if root shell)", min_value=0, max_value=1, value=get_current_input_value("root_shell"), help="1 if root shell is obtained.")
            input_values["su_attempted"] = st.number_input("SU Attempted (1 if SU attempted)", min_value=0, max_value=1, value=get_current_input_value("su_attempted"), help="1 if 'su root' command is attempted.")
            input_values["num_root"] = st.number_input("Num Root", min_value=0, value=get_current_input_value("num_root"), help="Number of root accesses.")
            input_values["num_file_creations"] = st.number_input("File Creations", min_value=0, value=get_current_input_value("num_file_creations"), help="Number of file creation operations.")
            input_values["num_shells"] = st.number_input("Num Shells", min_value=0, value=get_current_input_value("num_shells"), help="Number of shell prompts.")
            input_values["num_access_files"] = st.number_input("Access Files", min_value=0, value=get_current_input_value("num_access_files"), help="Number of operations on access control files.")
            input_values["num_outbound_cmds"] = st.number_input("Outbound Cmds", min_value=0, value=get_current_input_value("num_outbound_cmds"), help="Number of outbound commands (always 0 in KDD Cup 99).")
        with adv_col3:
            input_values["is_host_login"] = st.number_input("Is Host Login (1 if host login)", min_value=0, max_value=1, value=get_current_input_value("is_host_login"), help="1 if the login is 'host login'.")
            input_values["is_guest_login"] = st.number_input("Is Guest Login (1 if guest login)", min_value=0, max_value=1, value=get_current_input_value("is_guest_login"), help="1 if the login is 'guest login'.")
            input_values["count"] = st.number_input("Count (connections to same host in 2s)", min_value=0, value=get_current_input_value("count"), help="Number of connections to the same host as the current connection in the past 2 seconds.")
            input_values["srv_count"] = st.number_input("Srv Count (connections to same service in 2s)", min_value=0, value=get_current_input_value("srv_count"), help="Number of connections to the same service as the current connection in the past 2 seconds.")
            input_values["serror_rate"] = st.number_input("Serror Rate", min_value=0.0, max_value=1.0, value=get_current_input_value("serror_rate"), help="Percentage of connections with SYN errors (source-side).")
            input_values["srv_serror_rate"] = st.number_input("Srv Serror Rate", min_value=0.0, max_value=1.0, value=get_current_input_value("srv_serror_rate"), help="Percentage of connections to the same service with SYN errors.")
            input_values["rerror_rate"] = st.number_input("Rerror Rate", min_value=0.0, max_value=1.0, value=get_current_input_value("rerror_rate"), help="Percentage of connections with RST errors (source-side).")
            input_values["srv_rerror_rate"] = st.number_input("Srv Rerror Rate", min_value=0.0, max_value=1.0, value=get_current_input_value("srv_rerror_rate"), help="Percentage of connections to the same service with RST errors.")

    with st.expander("Destination Host & Service Details"):
        dest_col1, dest_col2, dest_col3 = st.columns(3)
        with dest_col1:
            input_values["same_srv_rate"] = st.number_input("Same Srv Rate (Source)", min_value=0.0, max_value=1.0, value=get_current_input_value("same_srv_rate"), help="Percentage of connections to the same service among all connections to the current host.")
            input_values["diff_srv_rate"] = st.number_input("Diff Srv Rate (Source)", min_value=0.0, max_value=1.0, value=get_current_input_value("diff_srv_rate"), help="Percentage of connections to different services from the same source.")
            input_values["srv_diff_host_rate"] = st.number_input("Srv Diff Host Rate", min_value=0.0, max_value=1.0, value=get_current_input_value("srv_diff_host_rate"), help="Percentage of connections to different hosts from the same service.")
            input_values["dst_host_count"] = st.number_input("Dst Host Count", min_value=0, value=get_current_input_value("dst_host_count"), help="Number of connections having the same destination host IP address.")
        with dest_col2:
            input_values["dst_host_srv_count"] = st.number_input("Dst Host Srv Count", min_value=0, value=get_current_input_value("dst_host_srv_count"), help="Number of connections having the same destination host IP address and using the same service.")
            input_values["dst_host_same_srv_rate"] = st.number_input("Dst Host Same Srv Rate", min_value=0.0, max_value=1.0, value=get_current_input_value("dst_host_same_srv_rate"), help="Percentage of connections having the same destination host and using the same service.")
            input_values["dst_host_diff_srv_rate"] = st.number_input("Dst Host Diff Srv Rate", min_value=0.0, max_value=1.0, value=get_current_input_value("dst_host_diff_srv_rate"), help="Percentage of connections having the same destination host and using different services.")
        with dest_col3:
            input_values["dst_host_same_src_port_rate"] = st.number_input("Dst Host Same Src Port Rate", min_value=0.0, max_value=1.0, value=get_current_input_value("dst_host_same_src_port_rate"), help="Percentage of connections having the same destination host and source port.")
            input_values["dst_host_srv_diff_host_rate"] = st.number_input("Dst Host Srv Diff Host Rate", min_value=0.0, max_value=1.0, value=get_current_input_value("dst_host_srv_diff_host_rate"), help="Percentage of connections having the same destination host and using a different service rate.")
            input_values["dst_host_serror_rate"] = st.number_input("Dst Host Serror Rate", min_value=0.0, max_value=1.0, value=get_current_input_value("dst_host_serror_rate"), help="Percentage of connections having the same destination host with SYN errors.")
            input_values["dst_host_srv_serror_rate"] = st.number_input("Dst Host Srv Serror Rate", min_value=0.0, max_value=1.0, value=get_current_input_value("dst_host_srv_serror_rate"), help="Percentage of connections having the same destination host and using the same service with SYN errors.")
            input_values["dst_host_rerror_rate"] = st.number_input("Dst Host Rerror Rate", min_value=0.0, max_value=1.0, value=get_current_input_value("dst_host_rerror_rate"), help="Percentage of connections having the same destination host with RST errors.")
            input_values["dst_host_srv_rerror_rate"] = st.number_input("Dst Host Srv Rerror Rate", min_value=0.0, max_value=1.0, value=get_current_input_value("dst_host_srv_rerror_rate"), help="Percentage of connections having the same destination host and using the same service with RST errors.")

    submitted = st.form_submit_button("Predict Anomaly")

    if submitted:
        is_valid_input = True
        error_messages = []

        for key, val in input_values.items():
            if isinstance(val, (int, float)) and val < 0:
                if key not in CATEGORICAL_COLS:
                    error_messages.append(f"'{key}' cannot be negative. Please enter a non-negative value.")
                    is_valid_input = False
            
            if key in ['serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
                        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']:
                if not (0.0 <= val <= 1.0):
                    error_messages.append(f"'{key}' must be between 0.0 and 1.0.")
                    is_valid_input = False
        
        if not is_valid_input:
            st.error("Input Validation Errors:")
            for msg in error_messages:
                st.write(f"- {msg}")
        else:
            processed_input = preprocess_input(input_values, scaler, label_encoders, COLUMN_NAMES_EXCEPT_LABEL, CATEGORICAL_COLS)

            prediction_raw = model.predict(processed_input)
            prediction_score = model.decision_function(processed_input)

            result_label = "Anomaly Detected! 🚨" if prediction_raw == -1 else "Normal Traffic 👍"

            st.subheader("Prediction Result:")
            if prediction_raw == -1:
                st.error(f"{result_label} (Anomaly Score: {prediction_score[0]:.4f})")
                st.write("This connection exhibits characteristics consistent with anomalous network behavior.")
                st.warning("Further investigation may be required.")
            else:
                st.success(f"{result_label} (Anomaly Score: {prediction_score[0]:.4f})")
                st.write("This connection appears to be normal network traffic.")

            st.info("Note: Lower anomaly scores indicate a higher likelihood of being an anomaly.")

st.sidebar.markdown("---")
st.sidebar.markdown("Project by: Ritika Choudhary")
st.sidebar.markdown("Date: July 23, 2025")