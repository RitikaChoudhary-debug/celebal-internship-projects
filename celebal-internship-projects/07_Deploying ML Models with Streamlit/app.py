import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# --- 1. Streamlit Page Configuration ---
st.set_page_config(
    page_title="Iris Species Predictor",
    page_icon="🌸",
    layout="wide", # Use wide layout for more space
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS for Aesthetics (Minimal, Streamlit handles most) ---
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5em;
        color: #4CAF50; /* A nice green color */
        text-align: center;
        margin-bottom: 0.5em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .subheader {
        font-size: 1.8em;
        color: #333333;
        margin-top: 1.5em;
        border-bottom: 2px solid #eee;
        padding-bottom: 0.5em;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        font-size: 1.2em;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .stAlert {
        border-radius: 8px;
    }
    .prediction-box {
        background-color: #e6ffe6; /* Light green */
        border-left: 8px solid #4CAF50;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Data Loading and Model Training (Cached for Performance) ---
@st.cache_resource # Use st.cache_resource for models and large objects
def load_data_and_train_model():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    target_names = iris.target_names

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train a RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred)

    # Return iris.feature_names along with other variables
    return X, y, target_names, model, accuracy, report, conf_mat, X_test, y_test, iris.feature_names

X, y, target_names, model, accuracy, report, conf_mat, X_test, y_test, feature_names = load_data_and_train_model()

# Add target names to the main DataFrame for visualizations
X['species'] = y
X['species_name'] = X['species'].map({i: name for i, name in enumerate(target_names)})

# --- 4. Sidebar for Model Information ---
st.sidebar.header("Model Information 📊")
st.sidebar.write("This app uses a `RandomForestClassifier` trained on the famous Iris dataset.")
st.sidebar.metric("Model Accuracy", f"{accuracy:.2f}")

st.sidebar.subheader("Dataset Features:")
# Use the 'feature_names' variable returned from the function
for i, feature in enumerate(feature_names):
    st.sidebar.write(f"- {feature.replace('_', ' ').title()}")

st.sidebar.markdown("---")
st.sidebar.info("Developed with ❤️ using Streamlit and Scikit-learn.")


# --- 5. Main Content Area ---
st.markdown("<h1 class='main-header'>🌸 Iris Species Predictor 🌸</h1>", unsafe_allow_html=True)
st.write("""
    Welcome to the interactive Iris Species Prediction app!
    Input the measurements of an Iris flower below, and our machine learning model
    will predict its species. You can also explore the model's performance and
    dataset characteristics.
""")

# --- 6. User Input Section ---
st.markdown("<h2 class='subheader'>📏 Input Flower Measurements</h2>", unsafe_allow_html=True)

# Create columns for a cleaner input layout
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.5, step=0.1)
    sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.0, step=0.1)
with col2:
    petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=4.0, step=0.1)
    petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.3, step=0.1)

# Create a DataFrame for the input
input_data = pd.DataFrame({
    'sepal length (cm)': [sepal_length],
    'sepal width (cm)': [sepal_width],
    'petal length (cm)': [petal_length],
    'petal width (cm)': [petal_width]
})

st.subheader("Your Input:")
st.dataframe(input_data, hide_index=True)

# --- 7. Make Prediction ---
if st.button("Predict Species ✨"):
    with st.spinner('Predicting...'):
        try:
            prediction_proba = model.predict_proba(input_data)[0]
            predicted_class_idx = np.argmax(prediction_proba)
            predicted_species = target_names[predicted_class_idx]

            st.markdown(f"""
                <div class="prediction-box">
                    <h3>Prediction Result:</h3>
                    <p style="font-size: 1.5em; font-weight: bold;">
                        The predicted Iris species is: <span style="color: #4CAF50;">{predicted_species.capitalize()}</span>
                    </p>
                    <p>Confidence:</p>
                    <ul>
            """, unsafe_allow_html=True)
            for i, prob in enumerate(prediction_proba):
                st.markdown(f"<li>{target_names[i].capitalize()}: {prob*100:.2f}%</li>", unsafe_allow_html=True)
            st.markdown("</ul></div>", unsafe_allow_html=True)
            st.snow() # A nice visual effect for success

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# --- 8. Model Output Understanding & Visualizations ---
st.markdown("<h2 class='subheader'>📊 Model Insights & Data Exploration</h2>", unsafe_allow_html=True)

# Dataset overview
with st.expander("Explore the Iris Dataset 📖"):
    st.write("A glimpse of the dataset used for training the model:")
    st.dataframe(X.head())
    st.write(f"Total samples: {len(X)}")
    st.write("Species distribution:")
    st.bar_chart(X['species_name'].value_counts())

# Pair Plot
st.subheader("Feature Relationships (Pair Plot)")
st.write("Visualize the relationships between all features, colored by species. This helps understand data separability.")
# Use the 'feature_names' variable returned from the function
fig_pair = px.scatter_matrix(X, dimensions=feature_names, color="species_name",
                             title="Pair Plot of Iris Features by Species",
                             height=700, width=1000)
st.plotly_chart(fig_pair, use_container_width=True)


# Feature Importance
st.subheader("Feature Importance 💡")
st.write("Understanding which features contributed most to the model's predictions.")
# Use the 'feature_names' variable returned from the function
feature_importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
# Changed `palette='viridis'` to `color='#4CAF50'` to address the FutureWarning
sns.barplot(x=feature_importances.values, y=feature_importances.index, ax=ax_importance, color='#4CAF50')
ax_importance.set_title("Feature Importances")
ax_importance.set_xlabel("Importance")
ax_importance.set_ylabel("Features")
st.pyplot(fig_importance)
plt.close(fig_importance)

# Model Performance Details (Confusion Matrix & Classification Report)
st.subheader("Detailed Model Performance 📈")
with st.expander("View Confusion Matrix"):
    st.write("The confusion matrix shows how many instances of each species were correctly/incorrectly classified.")
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
    ax_cm.set_xlabel("Predicted Label")
    ax_cm.set_ylabel("True Label")
    ax_cm.set_title("Confusion Matrix")
    st.pyplot(fig_cm)
    plt.close(fig_cm)

with st.expander("View Classification Report"):
    st.write("Precision, Recall, and F1-score for each species:")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

st.markdown("---")
st.markdown("Thank you for using the Iris Species Predictor!")
